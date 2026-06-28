"""Machine-bound sensor monitoring agents."""

import logging
from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

import simpy

from cps.agents.contracts import MonitoringReport
from cps.agents.fault_catalog import (
	actuator_diagnosis_id_template,
	actuator_sensor_diagnosis_id_template,
	battery_diagnosis_id_template,
	measurement_sensor_diagnosis_id_template,
	temperature_diagnosis_id_template,
)
from cps.agents.identifiers import machine_id_from_identifier, parse_identifier
from cps.agents.monitoring.base import MonitoringAgent
from cps.agents.monitoring.context import MonitoringContext
from cps.agents.monitoring.state_observers import BatteryStateObserver, TemperatureStateObserver
from cps.agents.resolution import LLMClient
from cps.components.sensors import ActuatorSensor, PowerSensor, TemperatureSensor
from cps.config import OBSERVATION_MONITOR_INTERVAL
from cps.core.node.machine import Machine
from cps.core.reporting import ReportedEvent, machine_issue_id
from cps.types import ActionOutcome, ActuatorFaultType, CoolingState, ProcessGenerator


def _is_sensor_event_for(event: ReportedEvent, sensor_type: str) -> bool:
	identifier = parse_identifier(event.identifier)
	return identifier.kind == "sensor" and identifier.sensor_type == sensor_type


SensorT = TypeVar("SensorT", bound=PowerSensor | TemperatureSensor | ActuatorSensor)


class MachineBoundSensorAgent(MonitoringAgent, Generic[SensorT]):
	sensor_type: ClassVar[str]
	# ``sensor`` is annotated with each subclass's concrete sensor type so the
	# generated UML draws an aggregation edge to the specific sensor it monitors.
	sensor: SensorT

	def __init__(self, sensor: SensorT, machine: Machine, llm_client: LLMClient) -> None:
		super().__init__(llm_client)
		self.sensor = sensor
		self.machine = machine

	@property
	def identity_name(self) -> str:
		return f"{self.name}@{self.machine.id}"

	def owns_event(self, event: ReportedEvent) -> bool:
		if machine_id_from_identifier(event.identifier) != self.sensor.machine_id:
			return False
		return _is_sensor_event_for(event, self.sensor_type)

	def _monitors(self) -> tuple[Callable[[], ProcessGenerator], ...]:
		"""Monitoring loops started for this agent."""
		return (self.observe_sensor,)

	def _domain_observations(self) -> tuple[Callable[[], str | None], ...]:
		"""Sensor observations beyond observe_fault that are reported each cycle."""
		return ()

	def _machine_context(self) -> dict[str, object]:
		return {"machine_id": self.machine.id}

	def _report_machine_id(self, identifier: str) -> str:
		return self.machine.id

	def start(self, env: simpy.Environment) -> tuple[simpy.Process, ...]:
		return tuple(env.process(monitor()) for monitor in self._monitors())

	def observe_sensor(self) -> ProcessGenerator:
		while True:
			yield self.machine.env.timeout(OBSERVATION_MONITOR_INTERVAL)
			for observe in (self.sensor.observe_fault, *self._domain_observations()):
				observation_id = observe()
				if observation_id is not None:
					self.machine.event_reporter.observation(observation_id, component=self.machine.id)

	def execute_action(self, report: MonitoringReport, *, require_sensor_operational: bool = False) -> ActionOutcome | None:
		if report.machine_id != self.machine.id:
			return None
		return super().execute_action(report, require_sensor_operational=require_sensor_operational)

	def _fix_sensor_fault(self, report: MonitoringReport, require_sensor_operational: bool, fault_type: str) -> bool | None:
		_ = require_sensor_operational
		parsed = parse_identifier(report.diagnosis_id or "")
		if parsed.kind != "sensor" or parsed.sensor_type != self.sensor_type:
			return None
		if parsed.observation != fault_type:
			return False
		if self.sensor.fault_type != fault_type:
			return False
		return self.sensor.dispatch_repair(self.machine.kpi_tracker, fault_type=fault_type)  # type: ignore[arg-type]

	def _fix_stuck(self, report: MonitoringReport, require_sensor_operational: bool) -> bool | None:
		return self._fix_sensor_fault(report, require_sensor_operational, "stuck")

	def _fix_no_signal(self, report: MonitoringReport, require_sensor_operational: bool) -> bool | None:
		return self._fix_sensor_fault(report, require_sensor_operational, "no_signal")

	def _action_handlers(self) -> dict[str, Callable[[MonitoringReport, bool], ActionOutcome | bool | None]]:
		return {"fix_stuck": self._fix_stuck, "fix_no_signal": self._fix_no_signal}


class PowerSensorAgent(MachineBoundSensorAgent[PowerSensor]):
	"""Reports Power sensor faults and the battery state it observes."""

	role = "power"
	name = "PowerSensor"
	sensor_type = "Power"
	sensor: PowerSensor

	def __init__(self, sensor: PowerSensor, machine: Machine, llm_client: LLMClient) -> None:
		super().__init__(sensor, machine, llm_client)
		self.state_observer = BatteryStateObserver(machine)

	def _monitors(self) -> tuple[Callable[[], ProcessGenerator], ...]:
		return (self.state_observer.monitor, self.observe_sensor)

	def _domain_observations(self) -> tuple[Callable[[], str | None], ...]:
		return (self.sensor.observe_low_battery,)

	system_prompt_focus = "Focus on Power sensor faults and battery state evidence for your assigned machine."
	system_prompt_diagnosis_ids = (measurement_sensor_diagnosis_id_template("Power"), battery_diagnosis_id_template())
	system_prompt_action_guidance = (
		"A Power sensor with no active fault reports the battery charge it reads; a low or depleted "
		"battery is a physical condition that worsens until the battery is serviced and can block "
		"production if it is left unaddressed."
	)

	def _llm_supplementary_context(self, context: MonitoringContext) -> dict[str, object]:
		if self.sensor.fault_type is not None:
			return self._machine_context()
		status = context.machine_status.get(self.machine.id)
		return self._machine_context() | {"battery_level": status["battery_level"] if status else self.machine.battery.level}

	def _replace_battery(self, report: MonitoringReport, require_sensor_operational: bool) -> bool:
		_ = require_sensor_operational
		parsed = parse_identifier(report.diagnosis_id or "")
		if parsed.kind != "battery" or parsed.parts[1] != self.machine.id:
			return False
		if parsed.state_or_issue not in {"low_battery", "dead_battery"}:
			return False
		if not (self.machine.battery.is_low or self.machine.battery.is_dead):
			return False
		return self.machine.battery.dispatch_replacement(
			self.machine.env,
			self.machine.kpi_tracker,
			after_replace=self.machine.resume_production_if_ready,
		)

	def _action_handlers(self) -> dict[str, Callable[[MonitoringReport, bool], ActionOutcome | bool | None]]:
		return {**super()._action_handlers(), "replace_battery": self._replace_battery}


class TemperatureSensorAgent(MachineBoundSensorAgent[TemperatureSensor]):
	"""Reports Temperature sensor faults and the thermal state it observes."""

	role = "temperature"
	name = "TemperatureSensor"
	sensor_type = "Temperature"
	sensor: TemperatureSensor

	def __init__(self, sensor: TemperatureSensor, machine: Machine, llm_client: LLMClient) -> None:
		super().__init__(sensor, machine, llm_client)
		self.state_observer = TemperatureStateObserver(machine)

	def _monitors(self) -> tuple[Callable[[], ProcessGenerator], ...]:
		return (self.state_observer.monitor, self.observe_sensor)

	def _domain_observations(self) -> tuple[Callable[[], str | None], ...]:
		return (self.sensor.observe_overheating,)

	system_prompt_focus = "Focus on Temperature sensor faults and overheating evidence for your assigned machine."
	system_prompt_diagnosis_ids = (measurement_sensor_diagnosis_id_template("Temperature"), temperature_diagnosis_id_template())
	system_prompt_action_guidance = (
		"A Temperature sensor with no active fault reports the thermal state it reads; overheating is a "
		"physical condition that escalates to a critical state and can block production if it is not "
		"relieved in time."
	)

	def _llm_supplementary_context(self, context: MonitoringContext) -> dict[str, object]:
		if self.sensor.fault_type is not None:
			return self._machine_context()
		status = context.machine_status.get(self.machine.id)
		return self._machine_context() | {
			"temperature": status["temperature"] if status else self.machine.temperature.value,
			"temperature_state": status["temperature_state"] if status else self.machine.temperature.state_id,
		}

	def _report_diagnosis_matches_temperature_state(self, report: MonitoringReport, state: str) -> bool:
		parsed = parse_identifier(report.diagnosis_id or "")
		return parsed.kind == "temperature" and parsed.parts[1] == self.machine.id and parsed.state_or_issue == state

	def _start_cooling(self, report: MonitoringReport, require_sensor_operational: bool) -> ActionOutcome:
		if require_sensor_operational and self.sensor.fault_type is not None:
			return "failed"
		if not self._report_diagnosis_matches_temperature_state(report, "overheating"):
			return "failed"
		if not self.machine.temperature.is_overheating:
			return "already_resolved"
		if not self.machine.temperature.start_light_cooling():
			# The overheating that justified the report ended during the polling window.
			return "already_resolved"
		logging.info(f"AGENT ACTION: Starting light cooling for {self.machine.id}", extra={"component": self.machine.id})
		self.machine.event_reporter.derived_issue(
			machine_issue_id(self.machine.id, "production_slowdown"),
			component=self.machine.id,
			cause_id=self.machine.temperature.state_id,
		)
		return "succeeded"

	def _start_intense_cooling(self, report: MonitoringReport, require_sensor_operational: bool) -> ActionOutcome:
		if require_sensor_operational and self.sensor.fault_type is not None:
			return "failed"
		if not self._report_diagnosis_matches_temperature_state(report, "critical_overheating"):
			return "failed"
		temperature = self.machine.temperature
		was_intense = temperature.cooling_state is CoolingState.INTENSE
		if not temperature.is_critical and not was_intense:
			return "already_resolved"
		if was_intense:
			logging.info(f"AGENT ACTION: Starting intense cooling for {self.machine.id}", extra={"component": self.machine.id})
			return "succeeded"
		temperature.start_intense_cooling()
		self.machine.block_production(temperature.state_id)
		self.machine.env.process(self.state_observer.complete_intense_cooling())
		logging.info(f"AGENT ACTION: Starting intense cooling for {self.machine.id}", extra={"component": self.machine.id})
		return "succeeded"

	def _action_handlers(self) -> dict[str, Callable[[MonitoringReport, bool], ActionOutcome | bool | None]]:
		return {
			**super()._action_handlers(),
			"start_cooling": self._start_cooling,
			"start_intense_cooling": self._start_intense_cooling,
		}


class ActuatorSensorAgent(MachineBoundSensorAgent[ActuatorSensor]):
	"""Reports actuator execution status and ActuatorSensor signal loss."""

	role = "actuator"
	name = "ActuatorSensor"
	sensor_type = "ActuatorSensor"
	sensor: ActuatorSensor

	def _domain_observations(self) -> tuple[Callable[[], str | None], ...]:
		return (self.sensor.observe_actuator_status,)

	system_prompt_focus = "Focus on actuator execution observations and ActuatorSensor signal health for your assigned machine."
	system_prompt_diagnosis_ids = (actuator_sensor_diagnosis_id_template(), actuator_diagnosis_id_template())
	system_prompt_action_guidance = (
		"A working ActuatorSensor reports the actuator's mechanical execution condition; a stuck or slow "
		"actuator is a fault of the actuator itself that persists until the actuator is serviced and can "
		"block production if it is left unaddressed."
	)

	def _llm_supplementary_context(self, context: MonitoringContext) -> dict[str, object]:
		if self.sensor.fault_type is not None:
			return self._machine_context()
		return self._machine_context() | {
			"actuator_fault_type": self.machine.actuator.fault_type,
		}

	def _fix_actuator_fault(self, report: MonitoringReport, require_sensor_operational: bool, fault_type: ActuatorFaultType) -> bool | None:
		parsed = parse_identifier(report.diagnosis_id or "")
		if parsed.kind != "actuator":
			return None
		if parsed.state_or_issue != fault_type:
			return False
		repair_dispatched = self.machine.actuator.dispatch_repair(
			self.machine.kpi_tracker,
			fault_type=fault_type,
			sensor_fault_type=self.sensor.fault_type,
			require_sensor_operational=require_sensor_operational,
			after_stuck_cleared=self.machine.resume_production_if_ready,
		)
		return repair_dispatched

	def _fix_stuck(self, report: MonitoringReport, require_sensor_operational: bool) -> bool | None:
		return self._fix_actuator_fault(report, require_sensor_operational, "stuck")

	def _fix_slow_response(self, report: MonitoringReport, require_sensor_operational: bool) -> bool | None:
		return self._fix_actuator_fault(report, require_sensor_operational, "slow_response")

	def _action_handlers(self) -> dict[str, Callable[[MonitoringReport, bool], ActionOutcome | bool | None]]:
		return {
			**super()._action_handlers(),
			"fix_stuck": self._fix_stuck,
			"fix_slow_response": self._fix_slow_response,
		}
