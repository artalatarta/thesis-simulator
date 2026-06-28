import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from cps.agents.identifiers import parse_identifier
from cps.agents.contracts import Conflict, MonitoringReport, ResolutionDecision
from cps.agents.monitoring.base import MonitoringAgent
from cps.agents.report_selection import PASSIVE_ACTIONS
from cps.types import ActionOutcome

ResolveConflict = Callable[[Conflict], ResolutionDecision | None]
logger = logging.getLogger(__name__)


class ActionSelection(NamedTuple):
	"""Reports cleared for action plus the ids a resolver decision selected."""

	reports: tuple[MonitoringReport, ...]
	resolver_selected_ids: frozenset[str]


def reports_selected_for_action(
	reports: Iterable[MonitoringReport],
	conflicts: Iterable[Conflict],
	*,
	resolve_conflict: ResolveConflict | None = None,
) -> ActionSelection:
	"""Return reports allowed to mutate simulator state.

	Reports involved in a conflict are withheld. If a resolver is available, one
	report matching each resolver decision is allowed through (tracked in
	``resolver_selected_ids``), along with supplemental direct-repair reports
	that are safe to execute regardless of the decision.
	"""
	report_tuple = tuple(reports)
	conflict_tuple = tuple(conflicts)
	conflicted_report_ids = {report.report_id for conflict in conflict_tuple for report in conflict.reports}
	selected = [report for report in report_tuple if report.report_id not in conflicted_report_ids]
	logger.debug(
		"ACTION_SELECTION_START reports=%s conflicts=%s initially_selected=%s conflicted=%s",
		[report.report_id for report in report_tuple],
		[conflict.conflict_id for conflict in conflict_tuple],
		[report.report_id for report in selected],
		sorted(conflicted_report_ids),
		extra={"component": "MonitoringAgents"},
	)
	if resolve_conflict is None:
		return ActionSelection(tuple(selected), frozenset())
	resolver_selected_ids: set[str] = set()
	selected_report_ids = {report.report_id for report in selected}
	decisions = _resolve_conflicts_concurrently(resolve_conflict, conflict_tuple)
	for conflict, decision in zip(conflict_tuple, decisions, strict=True):
		if decision is None:
			logger.debug("ACTION_SELECTION_NO_DECISION conflict_id=%s", conflict.conflict_id, extra={"component": "MonitoringAgents"})
			continue
		report = _report_selected_by_decision(conflict, decision)
		if report is not None and report.report_id not in selected_report_ids:
			selected.append(report)
			selected_report_ids.add(report.report_id)
			resolver_selected_ids.add(report.report_id)
			logger.debug(
				"ACTION_SELECTION_RESOLVER_REPORT conflict_id=%s decision_action=%s report_id=%s",
				conflict.conflict_id,
				decision.selected_action,
				report.report_id,
				extra={"component": "MonitoringAgents"},
			)
		for supplemental_report in _supplemental_enabling_reports(conflict):
			if supplemental_report.report_id in selected_report_ids:
				continue
			selected.append(supplemental_report)
			selected_report_ids.add(supplemental_report.report_id)
			logger.debug(
				"ACTION_SELECTION_SUPPLEMENTAL_REPORT conflict_id=%s report_id=%s action=%s diagnosis_id=%s",
				conflict.conflict_id,
				supplemental_report.report_id,
				supplemental_report.recommended_action,
				supplemental_report.diagnosis_id,
				extra={"component": "MonitoringAgents"},
			)
	logger.debug("ACTION_SELECTION_DONE selected=%s", [report.report_id for report in selected], extra={"component": "MonitoringAgents"})
	return ActionSelection(tuple(selected), frozenset(resolver_selected_ids))


def _resolve_conflicts_concurrently(
	resolve_conflict: ResolveConflict,
	conflicts: tuple[Conflict, ...],
) -> tuple[ResolutionDecision | None, ...]:
	"""Fetch every conflict's decision up front; each one is an independent model call."""
	if len(conflicts) <= 1:
		return tuple(resolve_conflict(conflict) for conflict in conflicts)
	with ThreadPoolExecutor(max_workers=len(conflicts), thread_name_prefix="conflict-resolver") as executor:
		return tuple(executor.map(resolve_conflict, conflicts))


def execute_report_actions_with_reports(
	agents: Sequence[MonitoringAgent],
	reports: Iterable[MonitoringReport],
	*,
	require_sensor_operational: bool = False,
	active_root_fault_keys: set[str] | frozenset[str] = frozenset(),
) -> list[tuple[MonitoringReport, ActionOutcome | None]]:
	"""Execute each report's action, retaining reports without a matching handler."""
	results: list[tuple[MonitoringReport, ActionOutcome | None]] = []
	for report in reports:
		if report.recommended_action in PASSIVE_ACTIONS:
			if _passive_report_fails_active_root_fault(report, active_root_fault_keys):
				results.append((report, "failed"))
			continue
		outcome = _execute_report_action(agents, report, require_sensor_operational=require_sensor_operational)
		logger.debug(
			"ACTION_EXECUTION report_id=%s agent_name=%s diagnosis_id=%s action=%s outcome=%s require_sensor_operational=%s",
			report.report_id,
			report.agent_name,
			report.diagnosis_id,
			report.recommended_action,
			outcome,
			require_sensor_operational,
			extra={"component": "MonitoringAgents"},
		)
		results.append((report, outcome))
	return results


def _report_selected_by_decision(
	conflict: Conflict,
	decision: ResolutionDecision,
) -> MonitoringReport | None:
	selected_index = decision.metadata.get("selected_report_index")
	if isinstance(selected_index, int) and not isinstance(selected_index, bool) and 1 <= selected_index <= len(conflict.reports):
		report = conflict.reports[selected_index - 1]
		if _report_matches_decision(report, decision):
			return report
	candidates = [
		report
		for report in conflict.reports
		if _report_matches_decision(report, decision)
	]
	if len(candidates) == 1:
		return candidates[0]
	return None


def _report_matches_decision(report: MonitoringReport, decision: ResolutionDecision) -> bool:
	return report.recommended_action == decision.selected_action and (
		decision.selected_diagnosis_id is None or report.diagnosis_id == decision.selected_diagnosis_id
	)


def _supplemental_enabling_reports(conflict: Conflict) -> tuple[MonitoringReport, ...]:
	return tuple(report for report in conflict.reports if _is_direct_repair_report(report))


def _is_direct_repair_report(report: MonitoringReport) -> bool:
	diagnosis_id = report.diagnosis_id or ""
	if diagnosis_id.startswith("sensor:"):
		return report.recommended_action in {"fix_stuck", "fix_no_signal"}
	if diagnosis_id.startswith("actuator:"):
		return report.recommended_action in {"fix_stuck", "fix_slow_response"}
	return False


def _execute_report_action(
	agents: Sequence[MonitoringAgent],
	report: MonitoringReport,
	*,
	require_sensor_operational: bool,
) -> ActionOutcome | None:
	if report.recommended_action in PASSIVE_ACTIONS:
		return None
	for agent in agents:
		agent_name = getattr(agent, "name", "")
		agent_identity = getattr(agent, "identity_name", agent_name)
		if report.agent_name not in {agent_identity, agent_name}:
			continue
		outcome = agent.execute_action(report, require_sensor_operational=require_sensor_operational)
		if outcome is not None:
			return outcome
	return None


def _passive_report_fails_active_root_fault(
	report: MonitoringReport,
	active_root_fault_keys: set[str] | frozenset[str],
) -> bool:
	if report.recommended_action not in PASSIVE_ACTIONS or not isinstance(report.diagnosis_id, str):
		return False
	fault_key = _active_fault_key_for_report(report)
	return fault_key is not None and fault_key in active_root_fault_keys


def _active_fault_key_for_report(report: MonitoringReport) -> str | None:
	if not isinstance(report.diagnosis_id, str):
		return None
	parsed = parse_identifier(report.diagnosis_id)
	if parsed.kind == "sensor" and len(parsed.parts) == 4:
		return f"{parsed.parts[1]}-{parsed.parts[2]}"
	if parsed.kind == "actuator" and len(parsed.parts) == 3:
		return f"{parsed.parts[1]}-Actuator"
	if parsed.kind == "machine" and len(parsed.parts) == 3:
		return f"{parsed.parts[1]}-Machine"
	if parsed.kind == "belt" and len(parsed.parts) == 4:
		return f"{parsed.parts[1]}->{parsed.parts[2]}-Belt"
	if parsed.kind == "network" and len(parsed.parts) == 2:
		return f"network-{parsed.parts[1]}"
	return None
