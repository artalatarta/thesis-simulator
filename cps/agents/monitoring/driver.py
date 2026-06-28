"""Monitoring driver that runs the diagnosis-and-recovery loop over the simulation."""

import logging
import threading
from collections.abc import Iterable, Mapping

import simpy

from cps.agents.contracts import Conflict, DetectsConflicts, EvidenceWindow, MonitoringReport, ResolutionDecision, ResolvesConflicts, executed_action_to_record
from cps.agents.monitoring import (
	MonitoringAgent,
	monitoring_context_from_machines,
	run_monitoring_agents,
)
from cps.agents.monitoring.actions import execute_report_actions_with_reports, reports_selected_for_action
from cps.agents.monitoring.debug_log import debug, report_summary
from cps.config import AGENT_POLL_INTERVAL
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.reporting import EventReporter
from cps.types import ActionOutcome, ProcessGenerator

MAX_AGENT_DECISION_HISTORY = 5


class MonitoringDriver:
	"""Drive recovery from the role-specific LLM monitoring agents.

	Each polling cycle the driver snapshots recent events plus controller
	status, runs the monitoring agents to produce reports, detects conflicts
	between reports covering the same machine and window, and executes the
	recommended action for each report through the responsible monitoring agent.

	Detected conflicts are retained as structured Conflict records in the conflicts list
	so downstream resolution layers (and tests) can inspect the
	disagreements rather than scraping them from the log.
	"""

	# Assigned in attach(); the handlers below only run from the process started there.
	_kpi_tracker: KPITracker

	def __init__(
		self,
		*,
		agents: Iterable[MonitoringAgent],
		detector: DetectsConflicts,
		resolver: ResolvesConflicts,
		event_reporter: EventReporter,
		interval: float = AGENT_POLL_INTERVAL,
		require_sensor_operational: bool = False,
	) -> None:
		self._agents = tuple(agents)
		self._interval = interval
		self._require_sensor_operational = require_sensor_operational
		self._detector = detector
		self._resolver = resolver
		self._event_reporter = event_reporter
		self.conflicts: list[Conflict] = []
		self.reports: list[MonitoringReport] = []
		self.report_ledger: list[MonitoringReport] = []
		self.resolution_decisions: list[ResolutionDecision] = []
		self.executed_actions: list[dict[str, object]] = []
		self._agent_decision_history: dict[str, list[dict[str, object]]] = {}
		self._active_report_keys: set[tuple[object, ...]] = set()
		self._active_action_keys: set[tuple[object, ...]] = set()
		self._closed = False
		# _resolve_conflict runs on resolver worker threads when a cycle has
		# multiple conflicts; this guards the KPI counters and decision list.
		self._resolution_lock = threading.Lock()

	def attach(
		self,
		env: simpy.Environment,
		machines: dict[str, Machine],
		kpi_tracker: KPITracker,
	) -> None:
		self._kpi_tracker = kpi_tracker
		for agent in self._agents:
			agent.start(env)
		env.process(self._run(env, machines))

	def _run(self, env: simpy.Environment, machines: dict[str, Machine]) -> ProcessGenerator:
		"""Polling loop that detects conflicts and turns reports into recovery actions."""
		event_reporter = self._event_reporter
		cursor = len(event_reporter.events)
		window_start = 0.0
		logging.info("MONITORING agent process started.", extra={"component": "MonitoringAgents"})
		while True:
			yield env.timeout(self._interval)
			if self._closed:
				return
			new_events = event_reporter.events[cursor:]
			cursor = len(event_reporter.events)
			window_end = float(env.now)
			debug("MONITORING_CYCLE", window=f"{window_start:.2f}-{window_end:.2f}", new_events=[event.identifier for event in new_events])
			if not new_events:
				self._active_report_keys.clear()
				self._active_action_keys.clear()
				window_start = window_end
				continue
			context = monitoring_context_from_machines(
				machines.values(),
				new_events,
				window_start=window_start,
				window_end=window_end,
				agent_decision_history=self._agent_decision_history,
			)
			window_start = window_end
			reports = tuple({_report_key(report): report for report in run_monitoring_agents(context, self._agents)}.values())
			self.report_ledger.extend(reports)
			debug("MONITORING_CYCLE_REPORTS", reports=[report_summary(report) for report in reports])
			new_reports = self._new_reports(reports)
			self.reports.extend(new_reports)
			detector_window = EvidenceWindow(
				start_time=context.window.start_time,
				end_time=context.window.end_time,
				events=context.observable_events(),
			)
			try:
				conflicts = self._detector.detect(reports, window=detector_window)
			except Exception:
				# LLM detector traces its own failures; this guard protects other detector implementations.
				logging.exception(
					"Conflict detector failed for evidence window %.1f-%.1f.",
					context.window.start_time,
					context.window.end_time,
					extra={"component": "MonitoringAgents"},
				)
				conflicts = ()
			for conflict in conflicts:
				self._handle_conflict(conflict)
			action_reports, resolver_selected_report_ids = reports_selected_for_action(reports, conflicts, resolve_conflict=self._resolve_conflict)
			new_action_reports = tuple(report for report in action_reports if _action_key(report) not in self._active_action_keys)
			debug(
				"MONITORING_ACTION_FILTER",
				action_reports=[report.report_id for report in action_reports],
				active_action_keys=sorted(str(key) for key in self._active_action_keys),
				new_action_reports=[report.report_id for report in new_action_reports],
			)
			action_results = execute_report_actions_with_reports(
				self._agents,
				new_action_reports,
				require_sensor_operational=self._require_sensor_operational,
				active_root_fault_keys=frozenset(self._kpi_tracker.open_faults),
			)
			# Keys stay active while their report keeps recurring: keep the keys of
			# reports suppressed this cycle, not just this cycle's executions, so a
			# recurring report is not re-executed every other cycle.
			suppressed_keys = {_action_key(report) for report in action_reports} & self._active_action_keys
			self._active_action_keys = suppressed_keys | {_action_key(report) for report, ok in action_results if ok == "succeeded"}
			debug(
				"MONITORING_ACTION_RESULTS",
				results=[(report.report_id, ok) for report, ok in action_results],
				active_action_keys=sorted(str(key) for key in self._active_action_keys),
			)
			for report, ok in action_results:
				action_record = executed_action_to_record(
					report,
					ok,
					selected_by_resolver=report.report_id in resolver_selected_report_ids,
				)
				if ok is not None:
					self._kpi_tracker.track_agent_action(ok)
				self.executed_actions.append(action_record)
			recent_action_records = self.executed_actions[-len(action_results) :] if action_results else []
			history_reports = _reports_for_decision_history(new_reports, action_results)
			self._remember_agent_decisions(
				history_reports,
				action_records={str(record["report_id"]): record for record in recent_action_records},
				resolver_selected_report_ids=resolver_selected_report_ids,
			)
			debug("MONITORING_EXECUTED_ACTION_RECORDS", records=recent_action_records)

	def _new_reports(self, reports: Iterable[MonitoringReport]) -> tuple[MonitoringReport, ...]:
		current = {_report_key(report): report for report in reports}
		new_reports = tuple(report for key, report in current.items() if key not in self._active_report_keys)
		self._active_report_keys = set(current)
		return new_reports

	def _remember_agent_decisions(
		self,
		reports: Iterable[MonitoringReport],
		*,
		action_records: Mapping[str, Mapping[str, object]] | None = None,
		resolver_selected_report_ids: frozenset[str] = frozenset(),
	) -> None:
		action_records = action_records or {}
		for report in reports:
			history = self._agent_decision_history.setdefault(report.agent_name, [])
			history.append(
				_decision_history_entry(
					report,
					action_record=action_records.get(report.report_id),
					selected_by_resolver=report.report_id in resolver_selected_report_ids,
				)
			)
			del history[:-MAX_AGENT_DECISION_HISTORY]

	def close(self) -> None:
		"""Stop monitoring after the current synchronous cycle returns."""
		self._closed = True

	def _handle_conflict(self, conflict: Conflict) -> None:
		logging.info(
			"CONFLICT %s on %s over %s across %d reports.",
			conflict.conflict_id,
			conflict.machine_id or "line",
			", ".join(conflict.conflict_types),
			len(conflict.reports),
			extra={"component": "MonitoringAgents"},
		)
		self.conflicts.append(conflict)
		self._kpi_tracker.track_conflict_detected()

	def _resolve_conflict(self, conflict: Conflict) -> ResolutionDecision | None:
		try:
			with self._resolution_lock:
				self._kpi_tracker.track_resolver_attempt()
			decision = self._resolver.resolve(conflict)
			with self._resolution_lock:
				self._kpi_tracker.track_resolver_success()
				self.resolution_decisions.append(decision)
			logging.info(
				"RESOLUTION %s selected %s / %s for conflict %s.",
				decision.decision_id,
				decision.selected_diagnosis,
				decision.selected_action,
				decision.conflict_id,
				extra={"component": "MonitoringAgents"},
			)
			return decision
		except Exception:
			with self._resolution_lock:
				self._kpi_tracker.track_resolver_failure()
			logging.exception(
				"Resolver failed for conflict %s.",
				conflict.conflict_id,
				extra={"component": "MonitoringAgents"},
			)
			return None


def _report_key(report: MonitoringReport) -> tuple[object, ...]:
	return (
		report.agent_name,
		report.machine_id,
		report.diagnosis_id,
		report.recommended_action,
		report.confidence,
		report.evidence,
	)


def _action_key(report: MonitoringReport) -> tuple[object, ...]:
	return (
		report.agent_name,
		report.machine_id,
		report.diagnosis_id,
		report.recommended_action,
	)


def _reports_for_decision_history(
	new_reports: Iterable[MonitoringReport],
	action_results: Iterable[tuple[MonitoringReport, ActionOutcome | None]],
) -> tuple[MonitoringReport, ...]:
	reports_by_id = {report.report_id: report for report in new_reports}
	for report, _outcome in action_results:
		reports_by_id.setdefault(report.report_id, report)
	return tuple(reports_by_id.values())


def _decision_history_entry(
	report: MonitoringReport,
	*,
	action_record: Mapping[str, object] | None = None,
	selected_by_resolver: bool = False,
) -> dict[str, object]:
	entry: dict[str, object] = {
		"time": report.time,
		"evidence": list(report.evidence),
		"diagnosis": report.diagnosis,
		"recommended_action": report.recommended_action,
		"confidence": report.confidence,
		"selected_by_resolver": selected_by_resolver,
		"execution_attempted": False,
		"execution_outcome": "not_executed",
		"execution_succeeded": False,
		"failure_reason": None,
	}
	if report.diagnosis_id is not None:
		entry["diagnosis_id"] = report.diagnosis_id
	if action_record is not None:
		entry.update(
			{
				"selected_by_resolver": bool(action_record["selected_by_resolver"]),
				"execution_attempted": bool(action_record["execution_attempted"]),
				"execution_outcome": action_record["execution_outcome"],
				"execution_succeeded": bool(action_record["execution_succeeded"]),
				"failure_reason": action_record["failure_reason"],
			}
		)
	return entry
