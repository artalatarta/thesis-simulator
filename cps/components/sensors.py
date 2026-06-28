import logging
import math
import random
from collections import deque
from collections.abc import Callable

import simpy

from cps.components.actuators import Actuator
from cps.components.battery import Battery
from cps.components.temperature import Temperature
from cps.core.kpi import KPITracker
from cps.core.reporting import EventReporter, sensor_event_id
from cps.types import SensorFaultType

SENSOR_FAULT_TYPES: tuple[SensorFaultType, ...] = ("stuck", "no_signal")
SENSOR_STUCK_FAULT_RANGE = 5.0
SENSOR_REPAIR_MIN_TIME = 4.0
SENSOR_REPAIR_MAX_TIME = 6.0


class Sensor:
	def __init__(self, env: simpy.Environment, machine_id: str, sensor_type: str, event_reporter: EventReporter | None = None) -> None:
		self.env = env
		self.machine_id = machine_id
		self.sensor_type = sensor_type
		self.event_reporter = event_reporter or EventReporter()
		self.fault_type: SensorFaultType | None = None
		self.pending_repair: SensorFaultType | None = None

	def inject_fault(self, fault_type: SensorFaultType) -> None:
		if fault_type != "no_signal":
			raise ValueError(f"{self.sensor_type} supports only no_signal sensor faults")
		self.fault_type = fault_type
		fault_id = sensor_event_id(self.machine_id, self.sensor_type, self.fault_type)
		self.event_reporter.root_fault(fault_id, message=f"FAULT INJECTED on {self.machine_id}-{self.sensor_type}: Type: {self.fault_type}")

	def observe_fault(self) -> str | None:
		if self.fault_type != "no_signal":
			return None
		return sensor_event_id(self.machine_id, self.sensor_type, "sensor_no_signal_detected")

	def clear_fault(self) -> None:
		logging.info(f"Corrective action: Clearing fault on {self.machine_id}-{self.sensor_type}", extra={"component": "System"})
		if self.fault_type is not None:
			self.event_reporter.fault_resolved(sensor_event_id(self.machine_id, self.sensor_type, self.fault_type), component=self.machine_id)
		self.fault_type = None

	def dispatch_repair(self, kpi_tracker: KPITracker, *, fault_type: SensorFaultType) -> bool:
		if self.fault_type != fault_type:
			return False
		logging.info(f"AGENT ACTION: Dispatching {self.sensor_type} repair for {self.machine_id} ({fault_type})", extra={"component": self.machine_id})
		logging.info(f"Corrective action: Sensor {self.sensor_type} for {self.machine_id} repaired {fault_type}.", extra={"component": "System"})
		self.clear_fault()
		kpi_tracker.track_fault_end(self.machine_id, self.sensor_type)
		return True

	def _sample_repair_time(self) -> float:
		if SENSOR_REPAIR_MIN_TIME > SENSOR_REPAIR_MAX_TIME:
			raise ValueError("SENSOR_REPAIR_MIN_TIME must be less than or equal to SENSOR_REPAIR_MAX_TIME")
		return random.uniform(SENSOR_REPAIR_MIN_TIME, SENSOR_REPAIR_MAX_TIME)

	def inject_random_fault(self) -> tuple[str, str]:
		"""Inject a randomly chosen supported fault; return the KPI tracking key."""
		self.inject_fault("no_signal")
		return self.machine_id, self.sensor_type


class MeasurementSensor(Sensor):
	HISTORY_LIMIT = 8
	MIN_HISTORY_FOR_TREND = 4
	NOISE_TOLERANCE = 0.05
	MIN_TRUE_CHANGE_FOR_STUCK = 1.0

	def __init__(self, env: simpy.Environment, machine_id: str, sensor_type: str, true_value_func: Callable[[], float], event_reporter: EventReporter | None = None) -> None:
		super().__init__(env, machine_id, sensor_type, event_reporter)
		self.true_value_func = true_value_func
		self.fault_param = 0.0
		self.reading_history: deque[tuple[float, float, float]] = deque(maxlen=self.HISTORY_LIMIT)

	def read_value(self) -> float:
		true_value = self.true_value_func()
		measured_value = true_value
		if self.fault_type == "stuck":
			measured_value = self.fault_param
		elif self.fault_type == "no_signal":
			measured_value = math.nan
		self._record_reading(measured_value, true_value)
		return measured_value

	def inject_fault(self, fault_type: SensorFaultType) -> None:
		if fault_type not in SENSOR_FAULT_TYPES:
			raise ValueError(f"{self.sensor_type} supports only stuck and no_signal sensor faults")
		self.fault_type = fault_type
		self.reading_history.clear()
		true_value = self.true_value_func()
		if self.fault_type == "stuck":
			self.fault_param = true_value + (random.random() - 0.5) * SENSOR_STUCK_FAULT_RANGE
		else:
			self.fault_param = 0.0
		fault_id = sensor_event_id(self.machine_id, self.sensor_type, self.fault_type)
		self.event_reporter.root_fault(fault_id, message=f"FAULT INJECTED on {self.machine_id}-{self.sensor_type}: Type: {self.fault_type}")

	def inject_random_fault(self) -> tuple[str, str]:
		self.inject_fault(random.choice(SENSOR_FAULT_TYPES))
		return self.machine_id, self.sensor_type

	def clear_fault(self) -> None:
		super().clear_fault()
		self.fault_param = 0.0
		self.reading_history.clear()

	def observe_fault(self) -> str | None:
		measured_value = self.read_value()
		if math.isnan(measured_value):
			return sensor_event_id(self.machine_id, self.sensor_type, "sensor_no_signal_detected")
		detected_fault = self._detect_fault_from_history()
		if detected_fault is None:
			return None
		return sensor_event_id(self.machine_id, self.sensor_type, f"sensor_{detected_fault}_detected")

	def _record_reading(self, measured_value: float, true_value: float) -> None:
		self.reading_history.append((self.env.now, measured_value, true_value))

	def _detect_fault_from_history(self) -> SensorFaultType | None:
		valid_readings = [(time, measured, true) for time, measured, true in self.reading_history if not math.isnan(measured) and not math.isnan(true)]
		if len(valid_readings) < self.MIN_HISTORY_FOR_TREND:
			return None

		measurements = [measured for _, measured, _ in valid_readings]
		true_values = [true for _, _, true in valid_readings]
		measurement_range = max(measurements) - min(measurements)
		true_range = max(true_values) - min(true_values)

		if measurement_range <= self.NOISE_TOLERANCE and true_range >= self.MIN_TRUE_CHANGE_FOR_STUCK:
			return "stuck"
		return None


class PowerSensor(MeasurementSensor):
	def __init__(self, env: simpy.Environment, machine_id: str, battery: Battery, event_reporter: EventReporter | None = None) -> None:
		super().__init__(env, machine_id, "Power", lambda: battery.level, event_reporter)
		self.battery = battery

	def observe_low_battery(self) -> str | None:
		if self.fault_type is not None:
			return None

		if self.battery.is_dead:
			return sensor_event_id(self.machine_id, "Power", "dead_battery_detected")

		if not self.battery.is_low:
			return None

		return sensor_event_id(self.machine_id, "Power", "low_battery_detected")


class TemperatureSensor(MeasurementSensor):
	def __init__(self, env: simpy.Environment, machine_id: str, temperature: Temperature, event_reporter: EventReporter | None = None) -> None:
		super().__init__(env, machine_id, "Temperature", lambda: temperature.value, event_reporter)
		self.temperature = temperature

	def observe_overheating(self) -> str | None:
		if self.fault_type is not None:
			return None

		if self.temperature.is_critical:
			return sensor_event_id(self.machine_id, "Temperature", "critical_overheating_detected")

		if not self.temperature.is_overheating:
			return None

		return sensor_event_id(self.machine_id, "Temperature", "overheating_detected")


class ActuatorSensor(Sensor):
	def __init__(self, env: simpy.Environment, machine_id: str, actuator: Actuator, event_reporter: EventReporter | None = None) -> None:
		super().__init__(env, machine_id, "ActuatorSensor", event_reporter)
		self.actuator = actuator

	def observe_actuator_status(self) -> str | None:
		if self.fault_type is not None:
			return None

		if self.actuator.fault_type == "stuck":
			return sensor_event_id(self.machine_id, "ActuatorSensor", "actuator_stuck_detected")

		if self.actuator.fault_type == "slow_response":
			return sensor_event_id(self.machine_id, "ActuatorSensor", "actuator_slow_response_detected")
		return None
