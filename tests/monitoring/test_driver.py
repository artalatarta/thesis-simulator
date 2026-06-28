import threading
from dataclasses import replace
from typing import cast

import simpy

from cps.agents.monitoring import MonitoringAgent
from cps.agents.monitoring.setup import configure_monitoring
from cps.agents.contracts import ActionLabel, ConfidenceLevel, Conflict, DiagnosisLabel, EvidenceWindow, MonitoringReport, ResolutionDecision
from cps.agents.diagnosis import component_label_for_identifier, diagnosis_label_for_catalog_id
from tests.fakes import MockLLMClient, RuleBasedDetector
from cps.agents.monitoring import monitoring_agents_for_machines
from cps.agents.monitoring.driver import (
	MAX_AGENT_DECISION_HISTORY,
	MonitoringDriver,
	_reports_for_decision_history,
	execute_report_actions_with_reports,
	reports_selected_for_action,
)
from cps.config import AGENT_POLL_INTERVAL
from cps.components.actuators import ACTUATOR_REPAIR_MAX_TIME
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from cps.core.reporting import EventReporter

reporter = EventReporter()

# The execute-action tests build agents only to route recovery actions, never to
# generate reports, so the LLM client is required but never reached.
_UNUSED_LLM_CLIENT = MockLLMClient(['{"reports": []}'])


def _agent(agent: object) -> MonitoringAgent:
	return cast(MonitoringAgent, agent)


def _make_machine(schedule: list[tuple[str, float]] | None = None) -> tuple[simpy.Environment, KPITracker, Machine, Network]:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	network = Network(env, reporter)
	machine = Machine(env, "M1", schedule or [], network, kpi_tracker)
	machine.start()
	network.start_observation_monitor()
	return env, kpi_tracker, machine, network


def _report(
	action: ActionLabel,
	*,
	diagnosis_id: str | None,
	machine_id: str | None,
	evidence: tuple[str, ...] = (),
	agent_name: str = "MachineHealth",
	confidence: ConfidenceLevel = "medium",
) -> MonitoringReport:
	return MonitoringReport(
		report_id=f"test-{action}",
		agent_role="machine_health",
		machine_id=machine_id,
		time=10.0,
		diagnosis=diagnosis_label_for_catalog_id(diagnosis_id) if diagnosis_id is not None else "unknown",
		recommended_action=action,
		confidence=confidence,
		evidence=evidence,
		diagnosis_id=diagnosis_id,
		component=component_label_for_identifier(diagnosis_id) if diagnosis_id is not None else "Line",
		agent_name=agent_name,
	)


def _resolution_decision(
	conflict: Conflict,
	*,
	diagnosis: DiagnosisLabel,
	action: ActionLabel,
	diagnosis_id: str,
	confidence: ConfidenceLevel = "high",
) -> ResolutionDecision:
	return ResolutionDecision(
		decision_id=f"resolution-{conflict.conflict_id}",
		conflict_id=conflict.conflict_id,
		selected_diagnosis=diagnosis_label_for_catalog_id(diagnosis_id),
		selected_action=action,
		confidence=confidence,
		supporting_report_ids=tuple(report.report_id for report in conflict.reports),
		explanation="test resolver",
		selected_diagnosis_id=diagnosis_id,
	)


def test_execute_report_actions_runs_actions_on_bound_components() -> None:
	_, kpi_tracker, machine, network = _make_machine()
	machine.power_sensor.inject_fault("stuck")
	machine.actuator.inject_fault("stuck")
	network.inject_fault("packet_loss")
	agents = monitoring_agents_for_machines([machine], network, kpi_tracker, _UNUSED_LLM_CLIENT)
	reports = [
		_report("fix_stuck", diagnosis_id="sensor:M1:Power:stuck", machine_id="M1", agent_name="PowerSensor"),
		_report(
			"fix_stuck",
			diagnosis_id="actuator:M1:stuck",
			machine_id="M1",
			evidence=("sensor:M1:ActuatorSensor:actuator_stuck_detected",),
			agent_name="ActuatorSensor",
		),
		_report("fix_packet_loss", diagnosis_id="network:packet_loss", machine_id=None, agent_name="Network"),
		_report("wait_for_more_evidence", diagnosis_id="belt:M1:M2:persistent_queue_pressure", machine_id="M1->M2"),
	]

	results = execute_report_actions_with_reports(agents, reports, require_sensor_operational=False)

	assert [(report.report_id, ok) for report, ok in results] == [
		("test-fix_stuck", "succeeded"),
		("test-fix_stuck", "succeeded"),
		("test-fix_packet_loss", "succeeded"),
	]
	assert machine.power_sensor.fault_type is None
	assert machine.power_sensor.pending_repair is None
	assert machine.actuator.fault_type is None
	assert machine.actuator.pending_repair is None
	assert network.pending_repairs == set()


def test_execute_report_actions_counts_passive_action_as_failed_when_root_fault_active_on_component() -> None:
	_, kpi_tracker, machine, network = _make_machine()
	agents = monitoring_agents_for_machines([machine], network, kpi_tracker, _UNUSED_LLM_CLIENT)
	reports = [
		_report("wait_for_more_evidence", diagnosis_id="sensor:M1:Power:stuck", machine_id="M1", agent_name="PowerSensor"),
		_report("wait_for_more_evidence", diagnosis_id="belt:M1:M2:persistent_queue_pressure", machine_id="M1->M2"),
	]

	results = execute_report_actions_with_reports(
		agents,
		reports,
		active_root_fault_keys=frozenset({"M1-Machine", "M1-Power"}),
	)

	assert [(report.report_id, ok) for report, ok in results] == [
		("test-wait_for_more_evidence", "failed"),
	]


def test_execute_report_actions_retains_unhandled_actions_from_wrong_monitoring_agent() -> None:
	_, kpi_tracker, machine, network = _make_machine()
	machine.power_sensor.inject_fault("stuck")
	machine.actuator.inject_fault("stuck")
	network.inject_fault("packet_loss")
	agents = monitoring_agents_for_machines([machine], network, kpi_tracker, _UNUSED_LLM_CLIENT)
	reports = [
		_report("fix_stuck", diagnosis_id="sensor:M1:Power:stuck", machine_id="M1", agent_name="MachineHealth"),
		_report(
			"fix_stuck",
			diagnosis_id="actuator:M1:stuck",
			machine_id="M1",
			evidence=("sensor:M1:ActuatorSensor:actuator_stuck_detected",),
			agent_name="PowerSensor",
		),
		_report("fix_packet_loss", diagnosis_id="network:packet_loss", machine_id=None, agent_name="BeltSegment"),
		_report("replace_battery", diagnosis_id="battery:M1:low_battery", machine_id="M1", agent_name="TemperatureSensor"),
		_report("start_cooling", diagnosis_id="temperature:M1:overheating", machine_id="M1", agent_name="PowerSensor"),
	]

	results = execute_report_actions_with_reports(agents, reports)

	assert [report for report, _outcome in results] == reports
	assert all(outcome is None for _report, outcome in results)
	assert machine.power_sensor.fault_type == "stuck"
	assert machine.actuator.fault_type == "stuck"
	assert network.pending_repairs == set()


def test_monitoring_driver_starts_with_empty_conflict_log() -> None:
	driver = MonitoringDriver(agents=(), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter)
	assert driver.conflicts == []
	driver.close()


def test_monitoring_driver_passes_only_observable_events_to_detector() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()

	class _RecordingDetector:
		def __init__(self):
			self.windows = []

		def detect(self, reports, *, window):
			del reports
			self.windows.append(window)
			return ()

	detector = _RecordingDetector()
	driver = MonitoringDriver(agents=(), detector=detector, resolver=_NeverCalledResolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)

	env.run(until=0.1)
	reporter.root_fault("sensor:M1:Power:stuck", context={"time": 0.2})
	reporter.physical_state("battery:M1:low_battery", component="M1", context={"time": 0.3})
	reporter.observation("sensor:M1:Power:sensor_stuck_detected", component="M1", context={"time": 0.4})
	reporter.derived_issue("machine:M1:production_blocked", component="M1", context={"time": 0.5})
	env.run(until=1.1)

	assert [[event.kind for event in window.events] for window in detector.windows] == [["observation", "derived_issue"]]
	assert [[event.identifier for event in window.events] for window in detector.windows] == [
		["sensor:M1:Power:sensor_stuck_detected", "machine:M1:production_blocked"]
	]
	driver.close()


def test_monitoring_driver_deduplicates_active_reports_until_a_quiet_window() -> None:
	driver = MonitoringDriver(agents=(), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter)
	report = _report(
		"fix_stuck",
		diagnosis_id="sensor:M1:Power:stuck",
		machine_id="M1",
		agent_name="PowerSensor",
	)

	assert driver._new_reports((report, report)) == (report,)
	assert driver._new_reports((report,)) == ()
	assert driver._new_reports(()) == ()
	assert driver._new_reports((report,)) == (report,)
	driver.close()


def test_monitoring_driver_remembers_passive_agent_decisions() -> None:
	driver = MonitoringDriver(agents=(), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter)
	wait_report = _report(
		"wait_for_more_evidence",
		diagnosis_id="sensor:M1:Power:stuck",
		machine_id="M1",
		agent_name="PowerSensor@M1",
	)

	driver._remember_agent_decisions((wait_report,))

	assert driver._agent_decision_history["PowerSensor@M1"] == [
		{
			"time": 10.0,
			"evidence": [],
			"diagnosis": "stuck",
			"diagnosis_id": "sensor:M1:Power:stuck",
			"recommended_action": "wait_for_more_evidence",
			"confidence": "medium",
			"selected_by_resolver": False,
			"execution_attempted": False,
			"execution_outcome": "not_executed",
			"execution_succeeded": False,
			"failure_reason": None,
		}
	]
	driver.close()


def test_monitoring_driver_remembers_bounded_agent_decisions_with_execution_state() -> None:
	driver = MonitoringDriver(agents=(), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter)
	wait_report = _report(
		"wait_for_more_evidence",
		diagnosis_id="sensor:M1:Power:stuck",
		machine_id="M1",
		agent_name="PowerSensor@M1",
	)
	action_reports = [
		_report(
			"fix_stuck",
			diagnosis_id="sensor:M1:Power:stuck",
			machine_id="M1",
			evidence=(f"sensor:M1:Power:stuck_{index}",),
			agent_name="PowerSensor@M1",
		)
		for index in range(MAX_AGENT_DECISION_HISTORY + 2)
	]

	last_report = action_reports[-1]
	driver._remember_agent_decisions(
		(wait_report, *action_reports),
		action_records={
			last_report.report_id: {
				"report_id": last_report.report_id,
				"selected_by_resolver": True,
				"execution_attempted": True,
				"execution_outcome": "already_resolved",
				"execution_succeeded": False,
				"failure_reason": None,
			}
		},
		resolver_selected_report_ids=frozenset({last_report.report_id}),
	)

	history = driver._agent_decision_history["PowerSensor@M1"]
	assert len(history) == MAX_AGENT_DECISION_HISTORY
	assert history[0]["evidence"] == ["sensor:M1:Power:stuck_2"]
	assert history[-1] == {
		"time": 10.0,
		"evidence": [f"sensor:M1:Power:stuck_{MAX_AGENT_DECISION_HISTORY + 1}"],
		"diagnosis": "stuck",
		"diagnosis_id": "sensor:M1:Power:stuck",
		"recommended_action": "fix_stuck",
		"confidence": "medium",
		"selected_by_resolver": True,
		"execution_attempted": True,
		"execution_outcome": "already_resolved",
		"execution_succeeded": False,
		"failure_reason": None,
	}
	driver.close()


def test_monitoring_driver_remembers_execution_state_for_recurring_deduped_report() -> None:
	driver = MonitoringDriver(agents=(), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter)
	first_report = replace(
		_report(
			"fix_stuck",
			diagnosis_id="sensor:M1:Power:stuck",
			machine_id="M1",
			evidence=("sensor:M1:Power:sensor_stuck_detected",),
			agent_name="PowerSensor@M1",
		),
		report_id="first-report",
	)
	recurring_report = replace(first_report, report_id="recurring-report", time=11.0)

	new_reports = driver._new_reports((first_report,))
	driver._remember_agent_decisions(
		_reports_for_decision_history(new_reports, ()),
	)
	new_reports = driver._new_reports((recurring_report,))
	assert new_reports == ()

	driver._remember_agent_decisions(
		_reports_for_decision_history(new_reports, ((recurring_report, "succeeded"),)),
		action_records={
			recurring_report.report_id: {
				"report_id": recurring_report.report_id,
				"selected_by_resolver": True,
				"execution_attempted": True,
				"execution_outcome": "succeeded",
				"execution_succeeded": True,
				"failure_reason": None,
			}
		},
		resolver_selected_report_ids=frozenset({recurring_report.report_id}),
	)

	history = driver._agent_decision_history["PowerSensor@M1"]
	assert [entry["execution_outcome"] for entry in history] == ["not_executed", "succeeded"]
	assert history[-1]["selected_by_resolver"] is True
	assert history[-1]["time"] == 11.0
	driver.close()


def test_monitoring_driver_report_ledger_covers_conflict_report_ids_across_deduped_cycles() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()

	class _ConflictAgent:
		name = "ConflictAgent"

		def __init__(self):
			self.calls = 0

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			self.calls += 1
			return (
				replace(
					_report(
						"wait_for_more_evidence",
						diagnosis_id="machine:M1:production_blocked",
						machine_id="M1",
						agent_name=self.name,
					),
					report_id=f"flow-t{self.calls}",
					time=float(self.calls),
				),
				replace(
					_report(
						"fix_stuck",
						diagnosis_id="sensor:M1:Power:stuck",
						machine_id="M1",
						agent_name=self.name,
					),
					report_id=f"sensor-t{self.calls}",
					time=float(self.calls),
				),
			)

		def execute_action(self, _report, *, require_sensor_operational=False):
			_ = require_sensor_operational
			return "succeeded"

	class _Resolver:
		def resolve(self, conflict):
			return _resolution_decision(
				conflict,
				diagnosis="production_blocked",
				action="wait_for_more_evidence",
				confidence="medium",
				diagnosis_id="machine:M1:production_blocked",
			)

	driver = MonitoringDriver(agents=(cast(MonitoringAgent, _ConflictAgent()),), detector=RuleBasedDetector(), resolver=_Resolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)
	for until in (0.1, 1.1, 2.1):
		env.run(until=until)
		reporter.observation("sensor:M1:Power:sensor_stuck_detected", component="M1")
	env.run(until=3.1)

	conflict_report_ids = {report.report_id for conflict in driver.conflicts for report in conflict.reports}
	ledger_report_ids = {report.report_id for report in driver.report_ledger}
	deduped_report_ids = {report.report_id for report in driver.reports}

	assert len(driver.conflicts) >= 2
	assert conflict_report_ids <= ledger_report_ids
	assert not conflict_report_ids <= deduped_report_ids
	driver.close()


def test_monitoring_driver_deduplicates_active_actions_when_report_metadata_changes() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()
	executed_reports = []

	class _ActionAgent:
		name = "PowerSensor"

		def __init__(self):
			self.calls = 0

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			self.calls += 1
			return (
				_report(
					"fix_stuck",
					diagnosis_id="sensor:M1:Power:stuck",
					machine_id="M1",
					evidence=("sensor:M1:Power:sensor_stuck_detected",),
					agent_name=self.name,
				),
			)

		def execute_action(self, report, *, require_sensor_operational=False):
			_ = require_sensor_operational
			executed_reports.append(report)
			return "succeeded"

	driver = MonitoringDriver(agents=(_agent(_ActionAgent()),), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)
	env.run(until=0.1)
	reporter.observation("sensor:M1:Power:sensor_stuck_detected", component="M1")
	env.run(until=1.1)
	reporter.observation("sensor:M1:Power:sensor_stuck_detected", component="M1")
	env.run(until=2.1)

	assert len(driver.reports) == 1
	assert len(executed_reports) == 1
	assert len(driver.executed_actions) == 1
	driver.close()


def test_monitoring_driver_deduplicates_recurring_action_across_many_cycles() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()
	executed_reports = []

	class _ActionAgent:
		name = "PowerSensor"

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			return (
				_report(
					"fix_stuck",
					diagnosis_id="sensor:M1:Power:stuck",
					machine_id="M1",
					evidence=("sensor:M1:Power:sensor_stuck_detected",),
					agent_name=self.name,
				),
			)

		def execute_action(self, report, *, require_sensor_operational=False):
			_ = require_sensor_operational
			executed_reports.append(report)
			return "succeeded"

	driver = MonitoringDriver(agents=(_agent(_ActionAgent()),), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)
	for until in (0.1, 1.1, 2.1, 3.1, 4.1):
		env.run(until=until)
		reporter.observation("sensor:M1:Power:sensor_stuck_detected", component="M1")
	env.run(until=5.1)

	assert len(executed_reports) == 1
	assert len(driver.executed_actions) == 1
	driver.close()


def test_monitoring_driver_does_not_dedupe_conflicted_unselected_actions() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()
	executed_reports = []

	class _ActionAgent:
		name = "ActuatorSensor"

		def __init__(self):
			self.calls = 0

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			self.calls += 1
			sensor_report = _report(
				"fix_no_signal",
				diagnosis_id="sensor:M1:ActuatorSensor:no_signal",
				machine_id="M1",
				evidence=("sensor:M1:ActuatorSensor:sensor_no_signal_detected",),
				agent_name=self.name,
			)
			if self.calls == 1:
				return (
					sensor_report,
					_report(
						"start_cooling",
						diagnosis_id="temperature:M1:overheating",
						machine_id="M1",
						evidence=("sensor:M1:Temperature:overheating_detected",),
						agent_name=self.name,
					),
				)
			return (sensor_report,)

		def execute_action(self, report, *, require_sensor_operational=False):
			_ = require_sensor_operational
			executed_reports.append(report)
			return "succeeded"

	class _Resolver:
		def resolve(self, conflict):
			return _resolution_decision(
				conflict,
				diagnosis="overheating",
				action="start_cooling",
				diagnosis_id="temperature:M1:overheating",
			)

	driver = MonitoringDriver(agents=(_agent(_ActionAgent()),), detector=RuleBasedDetector(), resolver=_Resolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)
	env.run(until=0.1)
	reporter.observation("sensor:M1:ActuatorSensor:sensor_no_signal_detected", component="M1")
	env.run(until=1.1)
	reporter.observation("sensor:M1:ActuatorSensor:sensor_no_signal_detected", component="M1")
	env.run(until=2.1)

	assert [report.recommended_action for report in executed_reports] == ["start_cooling", "fix_no_signal"]
	assert len(driver.executed_actions) == 2
	driver.close()


def test_slow_llm_blocks_simulation_progress_until_resolved() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()
	started = threading.Event()
	release = threading.Event()
	finished = threading.Event()

	class _BlockingAgent:
		name = "BlockingAgent"

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			started.set()
			release.wait()
			return ()

	driver = MonitoringDriver(agents=(_agent(_BlockingAgent()),), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)
	env.run(until=0.1)
	reporter.observation("test:event", component="M1")

	def run_until_next_poll() -> None:
		env.run(until=2.1)
		finished.set()

	thread = threading.Thread(target=run_until_next_poll)
	try:
		thread.start()
		assert started.wait(timeout=1.0)
		assert not finished.wait(timeout=0.05)
		assert thread.is_alive()
		release.set()
		thread.join(timeout=1.0)
		assert finished.is_set()
		assert env.now == 2.1
	finally:
		release.set()
		driver.close()
		thread.join(timeout=1.0)


def test_monitoring_agents_run_concurrently_before_simulation_resumes() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()
	first_started = threading.Event()
	second_started = threading.Event()
	release = threading.Event()
	finished = threading.Event()

	class _FirstBlockingAgent:
		name = "FirstBlockingAgent"

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			first_started.set()
			release.wait()
			return ()

	class _SecondBlockingAgent:
		name = "SecondBlockingAgent"

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			second_started.set()
			release.wait()
			return ()

	driver = MonitoringDriver(agents=(_agent(_FirstBlockingAgent()), _agent(_SecondBlockingAgent())), detector=RuleBasedDetector(), resolver=_NeverCalledResolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)
	env.run(until=0.1)
	reporter.observation("test:event", component="M1")

	def run_until_next_poll() -> None:
		env.run(until=2.1)
		finished.set()

	thread = threading.Thread(target=run_until_next_poll)
	try:
		thread.start()
		assert first_started.wait(timeout=1.0)
		assert second_started.wait(timeout=1.0)
		assert not finished.wait(timeout=0.05)
		release.set()
		thread.join(timeout=1.0)
		assert finished.is_set()
	finally:
		release.set()
		driver.close()
		thread.join(timeout=1.0)


def test_monitoring_cycle_runs_reports_resolution_and_actions_before_env_returns() -> None:
	reporter.clear()
	env, kpi_tracker, machine, _ = _make_machine()
	action_executed = threading.Event()

	class _ActionAgent:
		name = "ActionAgent"

		def start(self, _env):
			return ()

		def generate_reports(self, _context):
			return (
				_report(
					"wait_for_more_evidence",
					diagnosis_id="machine:M1:production_blocked",
					machine_id="M1",
					agent_name=self.name,
				),
				_report(
					"fix_stuck",
					diagnosis_id="sensor:M1:Power:stuck",
					machine_id="M1",
					agent_name=self.name,
				),
			)

		def execute_action(self, _report, *, require_sensor_operational=False):
			_ = require_sensor_operational
			action_executed.set()
			return "succeeded"

	class _Resolver:
		def resolve(self, conflict):
			return _resolution_decision(
				conflict,
				diagnosis="stuck",
				action="fix_stuck",
				diagnosis_id="sensor:M1:Power:stuck",
			)

	driver = MonitoringDriver(agents=(_agent(_ActionAgent()),), detector=RuleBasedDetector(), resolver=_Resolver(), event_reporter=reporter, interval=1.0)
	driver.attach(env, {"M1": machine}, kpi_tracker)
	env.run(until=0.1)
	reporter.observation("sensor:M1:Power:sensor_stuck_detected", component="M1")
	env.run(until=1.1)

	assert action_executed.is_set()
	assert len(driver.reports) == 2
	assert len(driver.conflicts) == 1
	assert len(driver.resolution_decisions) == 1
	assert driver.resolution_decisions[0].selected_action == "fix_stuck"
	assert len(driver.executed_actions) == 1
	assert driver.executed_actions[0]["report_id"] == "test-fix_stuck"
	assert driver.executed_actions[0]["selected_by_resolver"] is True
	assert driver.executed_actions[0]["execution_attempted"] is True
	assert driver.executed_actions[0]["execution_succeeded"] is True
	assert driver.executed_actions[0]["failure_reason"] is None
	driver.close()


class _NeverCalledResolver:
	def resolve(self, conflict):
		raise AssertionError("resolver should not be reached")


def test_conflicted_reports_are_not_selected_without_resolution() -> None:
	reports = [
		_report("wait_for_more_evidence", diagnosis_id="machine:M1:production_blocked", machine_id="M1"),
		_report("fix_stuck", diagnosis_id="sensor:M1:Power:stuck", machine_id="M1"),
	]
	conflict = RuleBasedDetector().detect(reports, window=EvidenceWindow(start_time=0.0, end_time=5.0))[0]

	selection = reports_selected_for_action(reports, [conflict])
	assert selection.reports == ()
	assert selection.resolver_selected_ids == frozenset()


def test_conflict_resolution_selects_only_matching_report_for_action() -> None:
	reports = [
		_report("wait_for_more_evidence", diagnosis_id="machine:M1:production_blocked", machine_id="M1"),
		_report("fix_stuck", diagnosis_id="sensor:M1:Power:stuck", machine_id="M1", agent_name="PowerSensor"),
	]
	conflict = RuleBasedDetector().detect(reports, window=EvidenceWindow(start_time=0.0, end_time=5.0))[0]

	def resolve(conflict):
		return _resolution_decision(
			conflict,
			diagnosis="stuck",
			action="fix_stuck",
			diagnosis_id="sensor:M1:Power:stuck",
		)

	selection = reports_selected_for_action(reports, [conflict], resolve_conflict=resolve)
	assert selection.reports == (reports[1],)
	assert selection.resolver_selected_ids == {reports[1].report_id}


def test_conflict_resolution_uses_selected_report_index_when_report_fields_are_duplicated() -> None:
	reports = [
		replace(
			_report(
				"fix_bearing_wear",
				diagnosis_id="machine:M1:bearing_wear",
				machine_id="M1",
				evidence=("machine:M1:bearing_wear_detected",),
				agent_name="MachineHealth",
				confidence="medium",
			),
			report_id="duplicate-medium",
		),
		replace(
			_report(
				"fix_bearing_wear",
				diagnosis_id="machine:M1:bearing_wear",
				machine_id="M1",
				evidence=("machine:M1:bearing_wear_detected",),
				agent_name="MachineHealth",
				confidence="high",
			),
			report_id="duplicate-high",
		),
	]
	conflict = Conflict(
		conflict_id="conflict-M1-duplicates",
		machine_id="M1",
		window=EvidenceWindow(start_time=0.0, end_time=5.0),
		conflict_types=("confidence",),
		reports=tuple(reports),
	)

	def resolve(conflict):
		return ResolutionDecision(
			decision_id=f"resolution-{conflict.conflict_id}",
			conflict_id=conflict.conflict_id,
			selected_diagnosis="bearing_wear",
			selected_action="fix_bearing_wear",
			confidence="high",
			supporting_report_ids=tuple(report.report_id for report in conflict.reports),
			explanation="second report has stronger confidence",
			metadata={"selected_report_index": 2},
			selected_diagnosis_id="machine:M1:bearing_wear",
		)

	selection = reports_selected_for_action(reports, [conflict], resolve_conflict=resolve)
	assert selection.reports == (reports[1],)
	assert selection.resolver_selected_ids == {reports[1].report_id}


def test_conflict_resolution_allows_direct_sensor_repair_as_supplemental_action() -> None:
	reports = [
		_report("start_cooling", diagnosis_id="temperature:M1:overheating", machine_id="M1", agent_name="TemperatureSensor"),
		_report("fix_no_signal", diagnosis_id="sensor:M1:ActuatorSensor:no_signal", machine_id="M1", agent_name="ActuatorSensor"),
	]
	conflict = RuleBasedDetector().detect(reports, window=EvidenceWindow(start_time=0.0, end_time=5.0))[0]

	def resolve(conflict):
		return _resolution_decision(
			conflict,
			diagnosis="overheating",
			action="start_cooling",
			diagnosis_id="temperature:M1:overheating",
		)

	selection = reports_selected_for_action(reports, [conflict], resolve_conflict=resolve)
	assert selection.reports == (reports[0], reports[1])
	# The supplemental sensor repair was not chosen by the resolver decision.
	assert selection.resolver_selected_ids == {reports[0].report_id}


def test_conflict_resolution_allows_direct_actuator_repair_as_supplemental_action() -> None:
	reports = [
		_report("wait_for_more_evidence", diagnosis_id="machine:M1:production_blocked", machine_id="M1"),
		_report(
			"fix_stuck",
			diagnosis_id="actuator:M1:stuck",
			machine_id="M1",
			evidence=("sensor:M1:ActuatorSensor:actuator_stuck_detected",),
			agent_name="ActuatorSensor",
		),
	]
	conflict = RuleBasedDetector().detect(reports, window=EvidenceWindow(start_time=0.0, end_time=5.0))[0]

	def resolve(conflict):
		return _resolution_decision(
			conflict,
			diagnosis="production_blocked",
			action="wait_for_more_evidence",
			confidence="medium",
			diagnosis_id="machine:M1:production_blocked",
		)

	selection = reports_selected_for_action(reports, [conflict], resolve_conflict=resolve)
	assert selection.reports == (reports[0], reports[1])
	assert selection.resolver_selected_ids == {reports[0].report_id}


def test_conflict_resolution_selects_passive_wait_for_more_evidence_for_accounting() -> None:
	reports = [
		_report("wait_for_more_evidence", diagnosis_id="sensor:M1:Power:stuck", machine_id="M1", agent_name="PowerSensor"),
		_report("fix_stuck", diagnosis_id="sensor:M1:Power:stuck", machine_id="M1", agent_name="PowerSensor"),
	]
	conflict = RuleBasedDetector().detect(reports, window=EvidenceWindow(start_time=0.0, end_time=5.0))[0]

	def resolve(conflict):
		return _resolution_decision(
			conflict,
			diagnosis="stuck",
			action="wait_for_more_evidence",
			confidence="medium",
			diagnosis_id="sensor:M1:Power:stuck",
		)

	selection = reports_selected_for_action(reports, [conflict], resolve_conflict=resolve)
	assert selection.reports == (reports[0], reports[1])
	assert selection.resolver_selected_ids == {reports[0].report_id}


def test_monitoring_driver_recovers_observed_stuck_actuator() -> None:
	reporter.clear()
	env, kpi_tracker, machine, network = _make_machine()
	machine.actuator.inject_fault("stuck")

	monitoring_client = MockLLMClient(
		[
			"""
			{"reports": [{
				"diagnosis": "stuck",
				"recommended_action": "fix_stuck",
				"confidence": "high",
				"evidence": ["sensor:M1:ActuatorSensor:actuator_stuck_detected"],
				"rationale": "ActuatorSensor observed a stuck actuator.",
				"diagnosis_id": "actuator:M1:stuck"
			}]}
			"""
		]
	)

	class _Resolver:
		def resolve(self, conflict):
			return _resolution_decision(
				conflict,
				diagnosis="stuck",
				action="fix_stuck",
				diagnosis_id="actuator:M1:stuck",
			)

	driver = configure_monitoring(env, {"M1": machine}, kpi_tracker, network, RuleBasedDetector(), _Resolver(), monitoring_client, reporter)
	assert isinstance(driver, MonitoringDriver)

	env.run(until=AGENT_POLL_INTERVAL + ACTUATOR_REPAIR_MAX_TIME + 1)

	assert machine.actuator.fault_type is None
	assert machine.actuator.pending_repair is None
	assert "M1-Actuator" not in kpi_tracker.open_faults
	driver.close()
