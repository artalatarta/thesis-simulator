import logging

import simpy

from cps.agents.contracts import MonitoringReport
from cps.agents.monitoring import PowerSensorAgent
from cps.components.battery import Battery
from cps.core.flow import BeltSegment
from cps.core.node import FinalStorage
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from tests.fakes import MockLLMClient

# These tests drive only the agent's physical observation/recovery processes, so
# the LLM client is required by the constructor but never reached.
_UNUSED_LLM_CLIENT = MockLLMClient(['{"reports": []}'])


def start_power_monitor(env: simpy.Environment, machine: Machine) -> None:
	agent = PowerSensorAgent(machine.power_sensor, machine, _UNUSED_LLM_CLIENT)
	for process in agent.start(env):
		assert process.is_alive


def test_battery_drains_by_processing_state() -> None:
	battery = Battery("M1", level=10.0, drain_rate=1.0)

	battery.drain(is_processing=True)
	assert battery.level == 9.0

	battery.drain(is_processing=False)
	assert battery.level == 8.8


def test_battery_reports_low_and_dead_state_ids() -> None:
	battery = Battery("M1", level=25.0, low_threshold=20.0)
	assert battery.state_id is None

	battery.level = 19.9
	assert battery.is_low
	assert battery.state_id == "battery:M1:low_battery"

	battery.level = 0.0
	assert battery.is_dead
	assert battery.state_id == "battery:M1:dead_battery"


def test_battery_replacement_restores_full_charge() -> None:
	battery = Battery("M1", level=0.0)

	battery.replace()

	assert battery.level == 100.0
	assert not battery.is_dead


def test_battery_replacement_dispatch_completes_immediately() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	battery = Battery("M1", level=0.0)

	assert battery.dispatch_replacement(env, kpi_tracker)
	assert battery.level == 100.0
	assert not battery.pending_replacement


def test_battery_replacement_dispatch_ignores_sampled_duration() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	battery = Battery("M1", level=0.0)

	assert battery.dispatch_replacement(env, kpi_tracker)
	assert battery.level == 100.0
	assert not battery.pending_replacement


def test_machine_stops_processing_when_battery_dies() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 10.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	start_power_monitor(env, machine)
	machine.battery.level = 0.05

	env.run(until=2)

	assert machine.battery.is_dead
	assert not machine.is_processing
	assert machine.parts_produced == 0
	assert machine.production_state.production_schedule == [("P1", 10.0)]


def test_machine_drains_battery_during_actuator_intake() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 10.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	start_power_monitor(env, machine)
	machine.battery.level = 100.0
	machine.battery.drain_rate = 1.0
	machine.actuator.base_action_time = 2.0

	env.run(until=1.5)

	assert machine.is_processing
	assert machine.battery.level < 99.9


def test_dead_battery_emits_production_blocked_issue(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	network = Network(env)
	machine = Machine(env, "M1", [], network, kpi_tracker)
	machine.outgoing_belt = BeltSegment(env, machine, FinalStorage(), network)
	machine.start()
	start_power_monitor(env, machine)
	machine.battery.level = 0.0

	with caplog.at_level(logging.WARNING):
		env.run(until=2)

	assert any(record.message == "machine:M1:production_blocked caused by battery:M1:dead_battery." for record in caplog.records)
	assert any(record.event_id == "battery:M1:dead_battery" and record.event_kind == "physical_state" for record in caplog.records)


def test_dead_battery_is_observed_after_power_sensor_repair(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	start_power_monitor(env, machine)
	machine.battery.level = 0.05
	machine.power_sensor.inject_fault("no_signal")

	with caplog.at_level(logging.WARNING):
		env.run(until=6)
		machine.power_sensor.clear_fault()
		env.run(until=11)

	assert machine.battery.is_dead
	assert any(record.event_id == "sensor:M1:Power:dead_battery_detected" for record in caplog.records)


def test_machine_continues_production_after_battery_replacement() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 1.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	start_power_monitor(env, machine)
	machine.battery.level = 0.05

	env.run(until=2)
	assert machine.battery.is_dead
	assert machine.production_state.production_schedule == [("P1", 1.0)]

	machine.battery.dispatch_replacement(
		env,
		kpi_tracker,
		after_replace=machine.resume_production_if_ready,
	)
	env.run(until=5)

	assert not machine.battery.is_dead
	assert machine.parts_produced == 1
	assert machine.production_state.production_schedule == []


def test_power_sensor_agent_dispatches_battery_replacement(monkeypatch) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	agent = PowerSensorAgent(machine.power_sensor, machine, _UNUSED_LLM_CLIENT)
	machine.battery.level = 0.0
	report = MonitoringReport(
		report_id="test-battery-replacement",
		agent_role="power",
		machine_id="M1",
		time=0.0,
		diagnosis="dead_battery",
		recommended_action="replace_battery",
		confidence="high",
		evidence=("sensor:M1:Power:dead_battery_detected",),
		diagnosis_id="battery:M1:dead_battery",
		agent_name="PowerSensor",
	)

	assert agent.execute_action(report) == "succeeded"
	assert machine.battery.level == 100.0
	assert not machine.battery.pending_replacement


def test_power_sensor_agent_rejects_replacement_without_battery_diagnosis() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	agent = PowerSensorAgent(machine.power_sensor, machine, _UNUSED_LLM_CLIENT)
	machine.battery.level = 0.0
	report = MonitoringReport(
		report_id="test-battery-replacement",
		agent_role="power",
		machine_id="M1",
		time=0.0,
		diagnosis="stuck",
		recommended_action="replace_battery",
		confidence="high",
		evidence=("sensor:M1:Power:dead_battery_detected",),
		diagnosis_id="sensor:M1:Power:stuck",
		agent_name="PowerSensor",
	)

	assert agent.execute_action(report) == "failed"
	assert not machine.battery.pending_replacement


def test_low_battery_observation_requires_working_power_sensor() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.battery.level = machine.battery.low_threshold - 1

	assert machine.power_sensor.observe_low_battery() == "sensor:M1:Power:low_battery_detected"

	machine.power_sensor.inject_fault("stuck")

	assert machine.power_sensor.observe_low_battery() is None


def test_dead_battery_observation_requires_working_power_sensor() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.battery.level = 0.0

	assert machine.power_sensor.observe_low_battery() == "sensor:M1:Power:dead_battery_detected"

	machine.power_sensor.inject_fault("stuck")

	assert machine.power_sensor.observe_low_battery() is None


def test_low_battery_warning_rearms_after_replacement(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	start_power_monitor(env, machine)
	machine.battery.level = machine.battery.low_threshold + 0.1
	machine.battery.drain_rate = 10.0

	with caplog.at_level(logging.WARNING):
		env.run(until=2)
		machine.battery.replace()
		machine.resume_production_if_ready()
		env.run(until=83)

	low_battery_logs = [record for record in caplog.records if "battery:M1:low_battery" in record.message]
	assert len(low_battery_logs) == 2
