import logging

import pytest
import simpy

from cps.agents.contracts import MonitoringReport
from cps.agents.diagnosis import component_label_for_identifier, diagnosis_label_for_catalog_id
from cps.agents.monitoring.sensors import ActuatorSensorAgent
from cps.components.actuators import ACTUATOR_REPAIR_MIN_TIME
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from tests.fakes import MockLLMClient


def make_machine() -> tuple[simpy.Environment, KPITracker, Machine]:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	return env, kpi_tracker, machine


def make_actuator_agent(machine: Machine) -> ActuatorSensorAgent:
	return ActuatorSensorAgent(machine.actuator_sensor, machine, MockLLMClient(['{"reports": []}']))


def actuator_report(action: str, diagnosis_id: str, evidence: str) -> MonitoringReport:
	return MonitoringReport(
		report_id=f"test-{action}",
		agent_role="actuator",
		machine_id="M1",
		time=10.0,
		diagnosis=diagnosis_label_for_catalog_id(diagnosis_id),
		recommended_action=action,  # type: ignore[arg-type]
		confidence="high",
		evidence=(evidence,),
		diagnosis_id=diagnosis_id,
		component=component_label_for_identifier(diagnosis_id),
		agent_name="ActuatorSensor@M1",
	)


def test_actuator_sensor_reports_stuck_and_slow_response_only_when_working() -> None:
	_, _, machine = make_machine()

	machine.actuator.inject_fault("stuck")
	assert machine.actuator_sensor.observe_actuator_status() == "sensor:M1:ActuatorSensor:actuator_stuck_detected"

	machine.actuator.clear_fault()
	machine.actuator.inject_fault("slow_response")
	assert machine.actuator_sensor.observe_actuator_status() == "sensor:M1:ActuatorSensor:actuator_slow_response_detected"


def test_actuator_sensor_rejects_measurement_faults() -> None:
	_, _, machine = make_machine()

	with pytest.raises(ValueError):
		machine.actuator_sensor.inject_fault("stuck")


def test_fix_stuck_schedules_instant_stuck_repair(monkeypatch: pytest.MonkeyPatch) -> None:
	env, kpi_tracker, machine = make_machine()
	machine.actuator.inject_fault("stuck")
	kpi_tracker.track_fault_start("M1", "Actuator")
	monkeypatch.setattr(machine.actuator, "_sample_repair_time", lambda: ACTUATOR_REPAIR_MIN_TIME)

	outcome = make_actuator_agent(machine).execute_action(
		actuator_report(
			"fix_stuck",
			"actuator:M1:stuck",
			"sensor:M1:ActuatorSensor:actuator_stuck_detected",
		)
	)

	assert outcome == "succeeded"
	assert machine.actuator.fault_type is None
	assert machine.actuator.pending_repair is None
	assert "M1-Actuator" not in kpi_tracker.open_faults


def test_fix_slow_response_schedules_instant_slow_response_repair(monkeypatch: pytest.MonkeyPatch) -> None:
	env, kpi_tracker, machine = make_machine()
	machine.actuator.inject_fault("slow_response")
	kpi_tracker.track_fault_start("M1", "Actuator")
	monkeypatch.setattr(machine.actuator, "_sample_repair_time", lambda: ACTUATOR_REPAIR_MIN_TIME)

	outcome = make_actuator_agent(machine).execute_action(
		actuator_report(
			"fix_slow_response",
			"actuator:M1:slow_response",
			"sensor:M1:ActuatorSensor:actuator_slow_response_detected",
		)
	)

	assert outcome == "succeeded"
	assert machine.actuator.fault_type is None
	assert machine.actuator.pending_repair is None
	assert "M1-Actuator" not in kpi_tracker.open_faults


def test_matching_actuator_repair_clears_fault_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
	env, _, machine = make_machine()
	machine.actuator.inject_fault("stuck")
	monkeypatch.setattr(machine.actuator, "_sample_repair_time", lambda: ACTUATOR_REPAIR_MIN_TIME)
	agent = make_actuator_agent(machine)
	report = actuator_report(
		"fix_stuck",
		"actuator:M1:stuck",
		"sensor:M1:ActuatorSensor:actuator_stuck_detected",
	)

	assert agent.execute_action(report) == "succeeded"
	assert machine.actuator.fault_type is None
	assert machine.actuator.pending_repair is None
	assert agent.execute_action(report) == "failed"


def test_fix_stuck_does_not_repair_slow_response() -> None:
	_, _, machine = make_machine()
	machine.actuator.inject_fault("slow_response")

	outcome = make_actuator_agent(machine).execute_action(
		actuator_report(
			"fix_stuck",
			"actuator:M1:stuck",
			"sensor:M1:ActuatorSensor:actuator_stuck_detected",
		)
	)

	assert outcome == "failed"
	assert machine.actuator.fault_type == "slow_response"
	assert machine.actuator.pending_repair is None


def test_fix_slow_response_does_not_repair_stuck() -> None:
	_, _, machine = make_machine()
	machine.actuator.inject_fault("stuck")

	outcome = make_actuator_agent(machine).execute_action(
		actuator_report(
			"fix_slow_response",
			"actuator:M1:slow_response",
			"sensor:M1:ActuatorSensor:actuator_slow_response_detected",
		)
	)

	assert outcome == "failed"
	assert machine.actuator.fault_type == "stuck"
	assert machine.actuator.pending_repair is None


def test_delayed_stuck_actuator_repair_resumes_production(monkeypatch: pytest.MonkeyPatch) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 1.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	machine.actuator.inject_fault("stuck")
	kpi_tracker.track_fault_start("M1", "Actuator")
	monkeypatch.setattr(machine.actuator, "_sample_repair_time", lambda: ACTUATOR_REPAIR_MIN_TIME)

	env.run(until=0.5)
	assert machine.parts_produced == 0
	assert machine.production_state.production_schedule == [("P1", 1.0)]
	assert not machine.production_process_is_alive()

	assert make_actuator_agent(machine).execute_action(
		actuator_report(
			"fix_stuck",
			"actuator:M1:stuck",
			"sensor:M1:ActuatorSensor:actuator_stuck_detected",
		)
	) == "succeeded"

	assert machine.actuator.fault_type is None
	env.run(until=3)
	assert machine.parts_produced == 1


def test_actuator_repair_requires_matching_fault() -> None:
	_, kpi_tracker, machine = make_machine()
	machine.actuator.inject_fault("stuck")

	assert not machine.actuator.dispatch_repair(
		kpi_tracker,
		fault_type="slow_response",
	)
	assert machine.actuator.fault_type == "stuck"


def test_actuator_repair_rejects_stale_observation_without_signal() -> None:
	_, kpi_tracker, machine = make_machine()
	machine.actuator.inject_fault("stuck")
	machine.actuator_sensor.inject_fault("no_signal")

	assert not machine.actuator.dispatch_repair(
		kpi_tracker,
		fault_type="stuck",
		sensor_fault_type=machine.actuator_sensor.fault_type,
	)
	assert machine.actuator.fault_type == "stuck"


def test_actuator_root_fault_identifier_has_no_actuator_type_segment(caplog) -> None:
	_, _, machine = make_machine()

	with caplog.at_level(logging.ERROR):
		machine.actuator.inject_fault("stuck")

	assert not hasattr(machine.actuator, "actuator_type")
	assert any(record.message == "FAULT INJECTED actuator:M1:stuck" for record in caplog.records)


def test_actuator_fault_emits_root_fault_only(caplog) -> None:
	_, _, machine = make_machine()

	with caplog.at_level(logging.ERROR):
		machine.actuator.inject_fault("slow_response")

	assert any(
		record.event_id == "actuator:M1:slow_response" and record.event_kind == "root_fault" for record in caplog.records
	)
	assert not any(record.event_id == "actuator:M1:slow_response" and record.event_kind == "physical_state" for record in caplog.records)


def test_actuator_faults_emit_derived_machine_issues(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	stuck_machine = Machine(env, "M1", [("P1", 1.0)], Network(env), kpi_tracker)
	stuck_machine.inbound_parts.append("P1")
	stuck_machine.start()
	stuck_machine.actuator.inject_fault("stuck")

	with caplog.at_level(logging.CRITICAL):
		env.run(until=1)

	assert any(record.message == "machine:M1:production_blocked caused by actuator:M1:stuck." for record in caplog.records)

	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M2"])
	slow_machine = Machine(env, "M2", [("P1", 1.0)], Network(env), kpi_tracker)
	slow_machine.inbound_parts.append("P1")
	slow_machine.start()
	slow_machine.actuator.inject_fault("slow_response")

	with caplog.at_level(logging.WARNING):
		env.run(until=1)

	assert any(record.message == "machine:M2:production_slowdown caused by actuator:M2:slow_response." for record in caplog.records)
