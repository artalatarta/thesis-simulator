import json
from types import MappingProxyType
from typing import cast

import pytest
import simpy

from cps.agents.contracts import AGENT_ROLES, AgentRole, MonitoringReport
from cps.agents.monitoring import (
	build_monitoring_context,
	monitoring_agents_for_machines,
	monitoring_context_from_machines,
	run_monitoring_agents,
)
from tests.fakes import MockLLMClient
from cps.core.flow import BeltSegment
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from cps.core.reporting import ReportedEvent, ReportKind

# A generic, schema-valid monitoring report the mock client can replay for any
# scoped event; the parser grounds and canonicalizes it from that event.
_GENERIC_REPORT_JSON = json.dumps(
	{
		"reports": [
			{
				"confidence": "medium",
				"component": "Line",
				"evidence": [],
				"rationale": "scoped analysis",
			}
		]
	}
)


def _event(
	identifier: str,
	kind: ReportKind,
	*,
	cause_id: str | None = None,
	time: float = 10.0,
) -> ReportedEvent:
	return ReportedEvent(
		identifier=identifier,
		kind=kind,
		component="test",
		cause_id=cause_id,
		context=MappingProxyType({"time": time}),
	)


def _context(events: list[ReportedEvent], **kwargs: object):
	return build_monitoring_context(events, **kwargs)  # type: ignore[arg-type]


def _machine(machine_id: str) -> tuple[simpy.Environment, KPITracker, Machine, Network]:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states([machine_id])
	network = Network(env)
	machine = Machine(env, machine_id, [], network, kpi_tracker)
	return env, kpi_tracker, machine, network


def _line_with_belt() -> tuple[simpy.Environment, KPITracker, Machine, Machine, Network]:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	network = Network(env)
	downstream = Machine(env, "M2", [], network, kpi_tracker)
	upstream = Machine(env, "M1", [], network, kpi_tracker)
	upstream.outgoing_belt = BeltSegment(env, upstream, downstream, network)
	return env, kpi_tracker, upstream, downstream, network


def _agent_named(agents, name: str):
	return next(agent for agent in agents if agent.name == name)


def _report_schema_properties(client: MockLLMClient) -> dict:
	response_format = client.response_formats[0]
	assert response_format is not None
	return response_format["json_schema"]["schema"]["properties"]["reports"]["items"]["properties"]  # type: ignore[index]


def _report(
	report_id: str,
	*,
	diagnosis_id: str,
	diagnosis: str,
	action: str,
	machine_id: str | None,
	confidence: str = "medium",
	role: AgentRole = "power",
	evidence: tuple[str, ...] | None = None,
) -> MonitoringReport:
	return MonitoringReport(
		report_id=report_id,
		agent_role=role,
		machine_id=machine_id,
		time=10.0,
		diagnosis=diagnosis,  # type: ignore[arg-type]
		recommended_action=action,  # type: ignore[arg-type]
		confidence=confidence,  # type: ignore[arg-type]
		evidence=evidence if evidence is not None else (diagnosis_id,),
		diagnosis_id=diagnosis_id,
		agent_name="Test",
		agent_kind="llm_agent",
		agent_model="mock",
	)


# --------------------------------------------------------------------------- #
# LLM report generation
# --------------------------------------------------------------------------- #
def test_llm_monitoring_agent_generates_valid_report() -> None:
	events = [_event("sensor:M1:Power:low_battery_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(
		[
			"""
			{"reports": [{
				"diagnosis": "battery_issue",
				"recommended_action": "replace_battery",
				"confidence": "high",
				"evidence": ["sensor:M1:Power:low_battery_detected"],
				"rationale": "Power sensor reports low battery.",
				"diagnosis_id": "battery:M1:low_battery"
			}]}
			""",
		],
		model="mock-monitor",
	)
	reports = _agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events))

	assert len(reports) == 1
	assert reports[0].agent_kind == "llm_agent"
	assert reports[0].agent_model == "mock-monitor"
	assert reports[0].agent_role == "power"
	assert reports[0].diagnosis_id == "battery:M1:low_battery"


def test_llm_monitoring_agent_uses_deterministic_window_report_id() -> None:
	events = [_event("sensor:M1:Power:low_battery_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(
		[
			"""
			{"reports": [{
				"report_id": "model-invented-id",
				"diagnosis": "battery_issue",
				"recommended_action": "replace_battery",
				"confidence": "high",
				"evidence": ["sensor:M1:Power:low_battery_detected"],
				"rationale": "Power sensor reports low battery.",
				"diagnosis_id": "battery:M1:low_battery"
			}]}
			""",
		]
	)

	report = _agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events, window_start=5.0))[0]

	assert report.report_id == "PowerSensor@M1-M1-t5-1"
	assert report.agent_name == "PowerSensor@M1"
	assert report.metadata["persona"] == "PowerSensor@M1 LLM monitoring agent (power)"


def test_llm_monitoring_agent_extracts_identifiers_from_structured_evidence() -> None:
	events = [_event("sensor:M1:Power:sensor_stuck_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(
		[
			"""
			{"reports": [{
				"diagnosis": "sensor_fault",
				"recommended_action": "fix_stuck",
				"confidence": "high",
				"evidence": [{"identifier": "sensor:M1:Power:sensor_stuck_detected"}],
				"rationale": "Power sensor stuck.",
				"diagnosis_id": "sensor:M1:Power:stuck"
			}]}
			""",
		]
	)

	reports = _agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events))

	assert reports[0].evidence == ("sensor:M1:Power:sensor_stuck_detected",)
	assert reports[0].machine_id == "M1"


def test_machine_bound_agent_ignores_evidence_outside_its_scope() -> None:
	events = [_event("sensor:M1:Power:sensor_stuck_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(
		[
			"""
			{"reports": [{
				"diagnosis": "sensor_fault",
				"recommended_action": "fix_stuck",
				"confidence": "high",
				"evidence": ["unparseable-evidence"],
				"rationale": "Power sensor stuck.",
				"diagnosis_id": "sensor:M1:Power:stuck"
			}]}
			""",
		]
	)

	agent = _agent_named(monitoring_agents_for_machines([machine], network, kpi_tracker, client), "PowerSensor")

	report = agent.generate_reports(_context(events))[0]

	assert report.evidence == ("sensor:M1:Power:sensor_stuck_detected",)
	assert report.diagnosis_id == "sensor:M1:Power:stuck"
	assert report.metadata["canonicalization_repairs"] == {
		"component_repaired": True,
		"diagnosis_repaired": True,
		"dropped_evidence": ["unparseable-evidence"],
	}


def test_llm_monitoring_agent_falls_back_for_action_outside_agent_scope() -> None:
	events = [_event("sensor:M1:Power:sensor_stuck_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(
		[
			"""
			{"reports": [{
				"diagnosis": "actuator_fault",
				"recommended_action": "fix_stuck",
				"confidence": "high",
				"evidence": ["sensor:M1:Power:sensor_stuck_detected"],
				"diagnosis_id": "actuator:M1:stuck"
			}]}
			""",
		]
	)

	report = _agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events))[0]

	# The model's diagnosis and diagnosis_id are ignored; the report is
	# canonicalized from scoped evidence.
	assert report.component == "PowerSensor"
	assert report.diagnosis == "stuck"
	assert report.diagnosis_id == "sensor:M1:Power:stuck"
	assert report.recommended_action == "fix_stuck"
	assert len(client.calls) == 1


def test_llm_monitoring_agent_ignores_model_disagreement_with_evidence() -> None:
	events = [_event("sensor:M1:ActuatorSensor:actuator_stuck_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(
		[
			"""
			{"reports": [{
				"diagnosis": "production_flow_issue",
				"recommended_action": "wait_for_more_evidence",
				"confidence": "high",
				"evidence": ["sensor:M1:ActuatorSensor:actuator_stuck_detected"],
				"diagnosis_id": "machine:M1:production_blocked"
			}]}
			""",
		]
	)

	report = _agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"ActuatorSensor",
	).generate_reports(_context(events))[0]

	assert report.component == "Actuator"
	assert report.diagnosis_id == "actuator:M1:stuck"
	assert report.diagnosis == "stuck"
	assert report.recommended_action == "fix_stuck"
	assert report.evidence == ("sensor:M1:ActuatorSensor:actuator_stuck_detected",)


def test_llm_monitoring_agent_repairs_action_incompatible_with_diagnosis_id() -> None:
	events = [_event("sensor:M1:ActuatorSensor:sensor_no_signal_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(
		[
			"""
			{"reports": [{
				"diagnosis": "sensor_fault",
				"recommended_action": "fix_slow_response",
				"confidence": "high",
				"evidence": ["sensor:M1:ActuatorSensor:sensor_no_signal_detected"],
				"rationale": "ActuatorSensor reports no signal.",
				"diagnosis_id": "sensor:M1:ActuatorSensor:no_signal"
			}]}
			""",
		]
	)

	report = _agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"ActuatorSensor",
	).generate_reports(_context(events))[0]

	assert report.diagnosis_id == "sensor:M1:ActuatorSensor:no_signal"
	assert report.recommended_action == "fix_no_signal"
	repairs = cast(dict[str, object], report.metadata["canonicalization_repairs"])
	assert repairs["recommended_action_repaired"] is True


def test_llm_monitoring_agent_raises_on_invalid_output() -> None:
	events = [_event("sensor:M1:Power:low_battery_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(["not json"])

	agent = _agent_named(monitoring_agents_for_machines([machine], network, kpi_tracker, client), "PowerSensor")
	# Failures now surface rather than silently degrading to a deterministic stub.
	with pytest.raises(ValueError):
		agent.generate_reports(_context(events))


def test_llm_monitoring_agent_retries_invalid_output() -> None:
	events = [_event("sensor:M1:Power:low_battery_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(["not json", _GENERIC_REPORT_JSON])

	reports = _agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events))

	assert len(reports) == 1
	assert [call[2] for call in client.calls] == [0.0, 0.2]


def test_run_monitoring_agents_isolates_invalid_agent_output() -> None:
	events = [
		_event("sensor:M1:Power:low_battery_detected", "observation"),
		_event("sensor:M1:Temperature:overheating_detected", "observation"),
	]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(["not json", "still not json", "also not json", _GENERIC_REPORT_JSON])
	agents = monitoring_agents_for_machines([machine], network, kpi_tracker, client)

	reports = run_monitoring_agents(_context(events), agents)

	assert reports
	assert {report.agent_name for report in reports} <= {"PowerSensor@M1", "TemperatureSensor@M1"}
	assert len(client.calls) == 4


def test_llm_monitoring_prompt_uses_concrete_agent_identity() -> None:
	events = [_event("sensor:M1:Power:low_battery_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(['{"reports": []}'])

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events))

	system_prompt, user_prompt, *_ = client.calls[0]
	assert "You are PowerSensor@M1" in system_prompt
	assert json.loads(user_prompt)["agent"] == "PowerSensor@M1"


def test_llm_monitoring_prompt_includes_agent_decision_history_as_background() -> None:
	events = [_event("sensor:M1:Power:low_battery_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(['{"reports": []}'])
	history = {
		"PowerSensor@M1": [
			{
				"time": 5.0,
				"evidence": ["sensor:M1:Power:low_battery_detected"],
				"diagnosis": "battery_issue",
				"diagnosis_id": "battery:M1:low_battery",
				"recommended_action": "replace_battery",
				"confidence": "high",
			}
		],
		"TemperatureSensor@M1": [
			{
				"time": 6.0,
				"evidence": ["sensor:M1:Temperature:overheating_detected"],
				"diagnosis": "temperature_issue",
				"recommended_action": "start_cooling",
				"confidence": "medium",
			}
		],
	}

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events, agent_decision_history=history))

	system_prompt, user_prompt, *_ = client.calls[0]
	payload = json.loads(user_prompt)
	assert system_prompt
	assert payload["supplementary_context"]["decision_history"] == history["PowerSensor@M1"]
	assert "TemperatureSensor@M1" not in user_prompt


def test_llm_monitoring_prompt_contains_only_agent_scoped_evidence() -> None:
	events = [
		_event("sensor:M1:Power:low_battery_detected", "observation", cause_id="battery:M1:low_battery"),
		_event("sensor:M1:Temperature:overheating_detected", "observation"),
		_event("network:packet_loss", "root_fault"),
	]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(['{"reports": []}'])

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events))

	user_prompt = client.calls[0][1]
	payload = json.loads(user_prompt)
	identifiers = [event["identifier"] for event in payload["events"]]
	assert identifiers == ["sensor:M1:Power:low_battery_detected"]
	assert "cause_id" not in payload["events"][0]
	assert "battery:M1:low_battery" not in user_prompt
	assert "network:packet_loss" not in user_prompt
	assert "sensor:M1:Temperature:overheating_detected" not in user_prompt


def test_faulty_power_sensor_prompt_does_not_include_battery_level() -> None:
	events = [_event("sensor:M1:Power:sensor_no_signal_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	machine.battery.level = 10.0
	machine.power_sensor.inject_fault("no_signal")
	client = MockLLMClient(['{"reports": []}'])

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events, machine_status={"M1": {"battery_level": 10.0}}))

	payload = json.loads(client.calls[0][1])
	assert payload["events"][0]["identifier"] == "sensor:M1:Power:sensor_no_signal_detected"
	assert payload["supplementary_context"] == {"machine_id": "M1"}


def test_faulty_temperature_sensor_prompt_does_not_include_temperature_state() -> None:
	events = [_event("sensor:M1:Temperature:sensor_stuck_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	machine.temperature_sensor.inject_fault("stuck")
	client = MockLLMClient(['{"reports": []}'])

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"TemperatureSensor",
	).generate_reports(
		_context(
			events,
			machine_status={
				"M1": {
					"temperature": 95.0,
					"temperature_state": "temperature:M1:overheating",
				}
			},
		)
	)

	payload = json.loads(client.calls[0][1])
	assert payload["events"][0]["identifier"] == "sensor:M1:Temperature:sensor_stuck_detected"
	assert payload["supplementary_context"] == {"machine_id": "M1"}


def test_faulty_actuator_sensor_prompt_does_not_include_actuator_fault_type() -> None:
	events = [_event("sensor:M1:ActuatorSensor:sensor_no_signal_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	machine.actuator.inject_fault("stuck")
	machine.actuator_sensor.inject_fault("no_signal")
	client = MockLLMClient(['{"reports": []}'])

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"ActuatorSensor",
	).generate_reports(_context(events))

	payload = json.loads(client.calls[0][1])
	assert payload["events"][0]["identifier"] == "sensor:M1:ActuatorSensor:sensor_no_signal_detected"
	assert payload["supplementary_context"] == {"machine_id": "M1"}


def test_llm_monitoring_prompt_scopes_actions_to_agent() -> None:
	events = [_event("sensor:M1:Power:low_battery_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(['{"reports": []}'])

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"PowerSensor",
	).generate_reports(_context(events))

	report_properties = _report_schema_properties(client)
	allowed_actions = report_properties["recommended_action"]["enum"]
	assert {"fix_stuck", "replace_battery"} <= set(allowed_actions)
	assert "fix_slow_response" not in allowed_actions
	assert "fix_packet_loss" not in allowed_actions


def test_actuator_monitoring_prompt_includes_only_actuator_agent_actions() -> None:
	events = [_event("sensor:M1:ActuatorSensor:actuator_stuck_detected", "observation")]
	_, kpi_tracker, machine, network = _machine("M1")
	client = MockLLMClient(['{"reports": []}'])

	_agent_named(
		monitoring_agents_for_machines([machine], network, kpi_tracker, client),
		"ActuatorSensor",
	).generate_reports(_context(events))

	report_properties = _report_schema_properties(client)
	allowed_actions = report_properties["recommended_action"]["enum"]
	assert {"fix_stuck", "fix_slow_response"} <= set(allowed_actions)
	assert "reboot_machine_process" not in allowed_actions
	assert "replace_battery" not in allowed_actions
	assert "fix_packet_loss" not in allowed_actions


def test_run_monitoring_agents_collects_every_role() -> None:
	events = [
		_event("sensor:M1:Power:low_battery_detected", "observation"),
		_event("sensor:M1:Temperature:overheating_detected", "observation"),
		_event("sensor:M1:ActuatorSensor:actuator_stuck_detected", "observation"),
		_event("network:network_latency_detected", "observation"),
		_event("belt:M1:M2:handoff_blocked", "derived_issue"),
		_event("machine:M1:production_blocked", "derived_issue"),
	]
	_, kpi_tracker, upstream, _, network = _line_with_belt()
	client = MockLLMClient([_GENERIC_REPORT_JSON])
	reports = run_monitoring_agents(_context(events), monitoring_agents_for_machines([upstream], network, kpi_tracker, client))

	produced_roles = {report.agent_role for report in reports}
	assert produced_roles == set(AGENT_ROLES)
	assert all(isinstance(report, MonitoringReport) for report in reports)


def test_oracle_only_physical_states_are_not_reported() -> None:
	events = [
		_event("temperature:M1:critical_overheating", "physical_state"),
		_event("battery:M1:dead_battery", "physical_state"),
		_event("network:packet_loss", "root_fault"),
	]
	_, kpi_tracker, machine, network = _machine("M1")
	# No observable scoped evidence: agents short-circuit before reaching the model.
	client = MockLLMClient(['{"reports": []}'])
	reports = run_monitoring_agents(_context(events), monitoring_agents_for_machines([machine], network, kpi_tracker, client))

	assert reports == ()
	assert client.calls == []


# --------------------------------------------------------------------------- #
# Context construction
# --------------------------------------------------------------------------- #
def test_build_monitoring_context_defaults_window_end_to_latest_event() -> None:
	events = [
		_event("sensor:M1:Power:low_battery_detected", "observation", time=5.0),
		_event("sensor:M1:Power:dead_battery_detected", "observation", time=18.0),
	]
	context = build_monitoring_context(events, window_start=2.0)

	assert context.window.start_time == 2.0
	assert context.window.end_time == 18.0


def test_monitoring_context_from_machines_snapshots_status_and_belts() -> None:
	events = [_event("machine:M1:production_slowdown", "derived_issue")]
	_, _, upstream, _, _ = _line_with_belt()
	upstream.is_processing = True
	assert upstream.outgoing_belt is not None
	upstream.outgoing_belt.diagnostics.symptom("persistent_queue_pressure").observe(12.0)
	context = monitoring_context_from_machines([upstream], events, window_end=12.0)

	assert context.machine_status["M1"]["is_processing"] is True
	assert context.belt_diagnostics["M1"] == ("belt:M1:M2:persistent_queue_pressure",)


# --------------------------------------------------------------------------- #
# Action execution scoping
# --------------------------------------------------------------------------- #
def test_machine_health_action_executes_on_the_reported_machine_only() -> None:
	from cps.agents.monitoring.actions import execute_report_actions_with_reports
	from cps.agents.monitoring.components import MachineHealthAgent

	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	network = Network(env)
	first = Machine(env, "M1", [], network, kpi_tracker)
	second = Machine(env, "M2", [], network, kpi_tracker)
	second.inject_fault("jammed_workpiece")
	kpi_tracker.track_fault_start(second.id, "Machine")
	client = MockLLMClient([_GENERIC_REPORT_JSON])
	agents = (MachineHealthAgent(first, client), MachineHealthAgent(second, client))
	report = _report(
		"MachineHealth-M2-1",
		diagnosis_id="machine:M2:jammed_workpiece",
		diagnosis="jammed_workpiece",
		action="fix_jammed_workpiece",
		machine_id="M2",
		role="machine_health",
	)
	report = MonitoringReport(**{**report.__dict__, "agent_name": "MachineHealth"})

	results = execute_report_actions_with_reports(agents, [report])

	assert results == [(report, "succeeded")]
	assert second.fault_type is None
	assert second.pending_repair is None
	assert first.fault_type is None


def test_belt_segment_action_executes_on_the_reported_belt_only() -> None:
	from cps.agents.monitoring.actions import execute_report_actions_with_reports
	from cps.agents.monitoring.components import BeltSegmentAgent

	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1", "M2", "M3"])
	network = Network(env)
	machines = [Machine(env, machine_id, [], network, kpi_tracker) for machine_id in ("M1", "M2", "M3")]
	first_belt = BeltSegment(env, machines[0], machines[1], network)
	second_belt = BeltSegment(env, machines[1], machines[2], network)
	second_belt.inject_fault("belt_jam")
	kpi_tracker.track_fault_start("M2->M3", "Belt")
	client = MockLLMClient([_GENERIC_REPORT_JSON])
	agents = (BeltSegmentAgent(first_belt, client), BeltSegmentAgent(second_belt, client))
	report = _report(
		"BeltSegment-M2-1",
		diagnosis_id="belt:M2:M3:belt_jam",
		diagnosis="belt_jam",
		action="fix_belt_jam",
		machine_id="M2",
		role="machine_health",
	)
	report = MonitoringReport(**{**report.__dict__, "agent_name": "BeltSegment"})

	results = execute_report_actions_with_reports(agents, [report])

	assert results == [(report, "succeeded")]
	assert second_belt.fault_type is None
	assert second_belt.pending_repair is None
	assert first_belt.fault_type is None


def test_clear_action_after_fault_already_cleared_is_already_resolved() -> None:
	from cps.agents.monitoring.actions import execute_report_actions_with_reports
	from cps.agents.monitoring.components import MachineHealthAgent

	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	agents = (MachineHealthAgent(machine, MockLLMClient([_GENERIC_REPORT_JSON])),)
	report = _report(
		"MachineHealth-M1-1",
		diagnosis_id="machine:M1:jammed_workpiece",
		diagnosis="jammed_workpiece",
		action="fix_jammed_workpiece",
		machine_id="M1",
		role="machine_health",
	)
	report = MonitoringReport(**{**report.__dict__, "agent_name": "MachineHealth"})

	results = execute_report_actions_with_reports(agents, [report])

	assert results == [(report, "already_resolved")]


def test_machine_fault_symptom_recurs_while_fault_persists() -> None:
	env, _, machine, _ = _machine("M1")
	reporter = machine.event_reporter
	agent_client = MockLLMClient([_GENERIC_REPORT_JSON])
	from cps.agents.monitoring.components import MachineHealthAgent

	agent = MachineHealthAgent(machine, agent_client)
	agent.start(env)
	machine.inject_fault("bearing_wear")
	env.run(until=10)

	recurring = [
		event
		for event in reporter.events
		if event.identifier == "machine:M1:production_slowdown" and event.cause_id == "machine:M1:bearing_wear"
	]
	assert len(recurring) >= 3  # injection-time emission plus periodic re-reports

	machine.clear_fault("bearing_wear")
	emitted_before_clear = len(recurring)
	env.run(until=20)
	recurring_after = [
		event
		for event in reporter.events
		if event.identifier == "machine:M1:production_slowdown" and event.cause_id == "machine:M1:bearing_wear"
	]
	assert len(recurring_after) == emitted_before_clear
