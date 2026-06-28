import logging

import simpy

from cps.agents.contracts import MonitoringReport
from cps.agents.monitoring import TemperatureSensorAgent
from tests.fakes import MockLLMClient
from cps.components.temperature import Temperature
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from cps.types import CoolingState

# These tests drive only the agent's physical observation/recovery processes, so
# the LLM client is required by the constructor but never reached.
_UNUSED_LLM_CLIENT = MockLLMClient(['{"reports": []}'])


def start_temperature_monitor(env: simpy.Environment, machine: Machine) -> TemperatureSensorAgent:
	agent = TemperatureSensorAgent(machine.temperature_sensor, machine, _UNUSED_LLM_CLIENT)
	for process in agent.start(env):
		assert process.is_alive
	return agent


def test_temperature_reports_overheating_and_critical_state_ids() -> None:
	temperature = Temperature("M1", value=89.0, warning_threshold=90.0, critical_threshold=100.0)
	assert temperature.state_id is None

	temperature.value = 95.0
	assert temperature.is_overheating
	assert temperature.state_id == "temperature:M1:overheating"

	temperature.value = 100.0
	assert temperature.is_critical
	assert temperature.state_id == "temperature:M1:critical_overheating"


def test_light_cooling_default_rate_is_lower_than_idle_cooling_rate() -> None:
	temperature = Temperature("M1")

	assert temperature.light_cooling_rate < temperature.idle_cooling_rate


def test_light_cooling_applies_only_while_processing_and_idle_uses_idle_rate() -> None:
	temperature = Temperature("M1", value=95.0)
	assert temperature.start_light_cooling() is True

	temperature.update(is_processing=True)
	assert temperature.value == 95.0 - temperature.light_cooling_rate

	value_before_idle_tick = temperature.value
	temperature.update(is_processing=False)
	assert temperature.value == value_before_idle_tick - temperature.idle_cooling_rate


def test_light_cooling_dispatch_is_idempotent_while_cooling_is_active() -> None:
	temperature = Temperature("M1", value=95.0)

	assert temperature.start_light_cooling() is True
	assert temperature.start_light_cooling() is True
	assert temperature.cooling_state is CoolingState.LIGHT


def test_light_cooling_dispatch_succeeds_when_intense_cooling_already_handles_temperature() -> None:
	temperature = Temperature("M1", value=105.0)

	assert temperature.start_intense_cooling() is True
	assert temperature.start_light_cooling() is True
	assert temperature.cooling_state is CoolingState.INTENSE


def test_faulty_temperature_sensor_suppresses_light_cooling() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	start_temperature_monitor(env, machine)
	machine.temperature.value = machine.temperature.warning_threshold + 1
	machine.temperature.idle_cooling_rate = 0.0
	machine.temperature_sensor.inject_fault("stuck")

	env.run(until=5.1)

	assert machine.temperature.cooling_state is CoolingState.NONE


def test_direct_light_cooling_dispatch_requires_working_temperature_sensor() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	start_temperature_monitor(env, machine)
	machine.temperature.value = machine.temperature.warning_threshold + 1
	machine.temperature_sensor.inject_fault("no_signal")
	agent = TemperatureSensorAgent(machine.temperature_sensor, machine, _UNUSED_LLM_CLIENT)
	report = MonitoringReport(
		report_id="temperature-test",
		agent_role="temperature",
		machine_id="M1",
		time=0.0,
		diagnosis="overheating",
		recommended_action="start_cooling",
		confidence="high",
		evidence=("sensor:M1:Temperature:overheating_detected",),
		diagnosis_id="temperature:M1:overheating",
		agent_name="TemperatureSensor",
	)

	assert agent.execute_action(report, require_sensor_operational=True) == "failed"
	assert machine.temperature.cooling_state is CoolingState.NONE


def test_machine_periodic_sensor_observation_only_emits_overheating_observation(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	start_temperature_monitor(env, machine)
	machine.temperature.value = machine.temperature.warning_threshold + 1
	machine.temperature.idle_cooling_rate = 0.0

	with caplog.at_level(logging.WARNING):
		env.run(until=5.1)

	assert machine.temperature.cooling_state is CoolingState.NONE
	assert any(record.event_id == "sensor:M1:Temperature:overheating_detected" for record in caplog.records)


def test_faulty_temperature_sensor_prevents_periodic_light_cooling() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.temperature.value = machine.temperature.warning_threshold + 1
	machine.temperature.idle_cooling_rate = 0.0
	machine.temperature_sensor.inject_fault("stuck")

	env.run(until=5.1)

	assert machine.temperature.cooling_state is CoolingState.NONE


def test_light_cooling_slows_production_and_reduces_temperature() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 4.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	agent = start_temperature_monitor(env, machine)
	machine.temperature.value = machine.temperature.warning_threshold + 1
	machine.temperature.heating_rate = 0.0
	machine.temperature.light_cooling_rate = 1.0
	report = MonitoringReport(
		report_id="temperature-test",
		agent_role="temperature",
		machine_id="M1",
		time=0.0,
		diagnosis="overheating",
		recommended_action="start_cooling",
		confidence="high",
		evidence=("sensor:M1:Temperature:overheating_detected",),
		diagnosis_id="temperature:M1:overheating",
		agent_name="TemperatureSensor",
	)
	assert agent.execute_action(report) == "succeeded"

	env.run(until=5.5)

	assert machine.parts_produced == 0
	assert machine.temperature.value < machine.temperature.warning_threshold + 1

	env.run(until=7)

	assert machine.parts_produced == 1


def test_critical_temperature_observation_does_not_start_deterministic_cooling() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 10.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	start_temperature_monitor(env, machine)
	machine.temperature.value = machine.temperature.critical_threshold + 1
	machine.temperature.intense_cooling_rate = 20.0

	env.run(until=2)

	assert not machine.thermal_blocked
	assert machine.temperature.cooling_state is CoolingState.NONE
	assert machine.parts_produced == 0


def test_intense_cooling_agent_action_blocks_and_recovers_production() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 10.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	agent = start_temperature_monitor(env, machine)
	machine.temperature.value = machine.temperature.critical_threshold + 1
	machine.temperature.intense_cooling_rate = 20.0
	report = MonitoringReport(
		report_id="temperature-test",
		agent_role="temperature",
		machine_id="M1",
		time=0.0,
		diagnosis="critical_overheating",
		recommended_action="start_intense_cooling",
		confidence="high",
		evidence=("sensor:M1:Temperature:critical_overheating_detected",),
		diagnosis_id="temperature:M1:critical_overheating",
		agent_name="TemperatureSensor",
	)

	assert agent.execute_action(report) == "succeeded"
	assert machine.thermal_blocked
	assert machine.temperature.cooling_state is CoolingState.INTENSE
	assert machine.parts_produced == 0
	assert machine.production_state.production_schedule == [("P1", 10.0)]

	env.run(until=5)

	assert not machine.thermal_blocked
	assert machine.parts_produced == 0

	env.run(until=20)

	assert machine.parts_produced == 1


def test_shutdown_threshold_latches_until_safe() -> None:
	temperature = Temperature("M1", value=105.0)
	assert not temperature.is_shutdown
	assert not temperature.is_thermal_blocked

	# Heating across the shutdown threshold latches the cutoff.
	temperature.update(is_processing=True)  # 105 -> 107
	temperature.update(is_processing=True)  # 107 -> 109
	temperature.update(is_processing=True)  # 109 -> 111
	assert temperature.value >= temperature.shutdown_threshold
	assert temperature.is_shutdown
	assert temperature.is_thermal_blocked
	assert temperature.state_id == "temperature:M1:critical_overheating"

	# Idle cooling between safe and shutdown keeps the cutoff latched.
	temperature.value = 80.0
	temperature.update(is_processing=False)
	assert temperature.is_shutdown
	assert temperature.is_thermal_blocked

	# Reaching the safe threshold releases it.
	temperature.value = temperature.safe_threshold
	temperature.update(is_processing=False)
	assert not temperature.is_shutdown
	assert not temperature.is_thermal_blocked


def test_machine_auto_shuts_down_and_recovers_at_shutdown_threshold() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [("P1", 100.0)], Network(env), kpi_tracker)
	machine.inbound_parts.append("P1")
	machine.start()
	machine.temperature.value = machine.temperature.shutdown_threshold - 1
	# No idle cooling, so once the cutoff trips the machine stays stopped.
	machine.temperature.idle_cooling_rate = 0.0

	# Heating while processing crosses 110C and trips the cutoff mid-part.
	env.run(until=8)
	assert machine.thermal_blocked
	assert machine.temperature.is_shutdown
	assert machine.parts_produced == 0

	# Cooling back to the safe threshold releases the cutoff automatically.
	machine.temperature.idle_cooling_rate = 50.0
	env.run(until=10)
	assert not machine.thermal_blocked
	assert not machine.temperature.is_shutdown


def test_light_cooling_on_no_longer_overheating_machine_is_already_resolved() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	start_temperature_monitor(env, machine)
	machine.temperature.value = machine.temperature.warning_threshold - 1
	agent = TemperatureSensorAgent(machine.temperature_sensor, machine, _UNUSED_LLM_CLIENT)
	report = MonitoringReport(
		report_id="temperature-test",
		agent_role="temperature",
		machine_id="M1",
		time=0.0,
		diagnosis="overheating",
		recommended_action="start_cooling",
		confidence="high",
		evidence=("sensor:M1:Temperature:overheating_detected",),
		diagnosis_id="temperature:M1:overheating",
		agent_name="TemperatureSensor",
	)

	assert agent.execute_action(report) == "already_resolved"
	assert machine.temperature.cooling_state is CoolingState.NONE


def test_light_cooling_rejects_non_temperature_diagnosis() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	machine.temperature.value = machine.temperature.warning_threshold + 1
	agent = TemperatureSensorAgent(machine.temperature_sensor, machine, _UNUSED_LLM_CLIENT)
	report = MonitoringReport(
		report_id="temperature-test",
		agent_role="temperature",
		machine_id="M1",
		time=0.0,
		diagnosis="stuck",
		recommended_action="start_cooling",
		confidence="high",
		evidence=("sensor:M1:Temperature:overheating_detected",),
		diagnosis_id="sensor:M1:Temperature:stuck",
		agent_name="TemperatureSensor",
	)

	assert agent.execute_action(report) == "failed"
	assert machine.temperature.cooling_state is CoolingState.NONE


def test_intense_cooling_rejects_non_critical_temperature_diagnosis() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	machine = Machine(env, "M1", [], Network(env), kpi_tracker)
	machine.start()
	machine.temperature.value = machine.temperature.critical_threshold + 1
	agent = TemperatureSensorAgent(machine.temperature_sensor, machine, _UNUSED_LLM_CLIENT)
	report = MonitoringReport(
		report_id="temperature-test",
		agent_role="temperature",
		machine_id="M1",
		time=0.0,
		diagnosis="overheating",
		recommended_action="start_intense_cooling",
		confidence="high",
		evidence=("sensor:M1:Temperature:overheating_detected",),
		diagnosis_id="temperature:M1:overheating",
		agent_name="TemperatureSensor",
	)

	assert agent.execute_action(report) == "failed"
	assert machine.temperature.cooling_state is CoolingState.NONE
