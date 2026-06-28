import simpy

from cps.components.actuators import Actuator
from cps.components.battery import Battery
from cps.components.sensors import ActuatorSensor, MeasurementSensor, PowerSensor, TemperatureSensor
from cps.components.temperature import Temperature


def observed_after_history(sensor: MeasurementSensor) -> str | None:
	observation_id = None
	for _ in range(sensor.MIN_HISTORY_FOR_TREND):
		observation_id = sensor.observe_fault()
	return observation_id


def test_no_signal_fault_is_observable_immediately() -> None:
	env = simpy.Environment()
	sensor = MeasurementSensor(env, "M1", "Temperature", lambda: 50.0)
	sensor.inject_fault("no_signal")

	assert sensor.observe_fault() == "sensor:M1:Temperature:sensor_no_signal_detected"


def test_stuck_fault_is_observable_when_true_value_changes() -> None:
	env = simpy.Environment()
	true_value = 50.0

	def read_true_value() -> float:
		nonlocal true_value
		true_value += 1.0
		return true_value

	sensor = MeasurementSensor(env, "M1", "Power", read_true_value)
	sensor.inject_fault("stuck")

	assert observed_after_history(sensor) == "sensor:M1:Power:sensor_stuck_detected"


def test_faulty_power_sensor_suppresses_battery_state_observations() -> None:
	env = simpy.Environment()
	battery = Battery("M1", level=10.0)
	sensor = PowerSensor(env, "M1", battery)

	assert sensor.observe_low_battery() == "sensor:M1:Power:low_battery_detected"

	sensor.inject_fault("no_signal")

	assert sensor.observe_low_battery() is None


def test_faulty_temperature_sensor_suppresses_overheating_observations() -> None:
	env = simpy.Environment()
	temperature = Temperature("M1", value=95.0)
	sensor = TemperatureSensor(env, "M1", temperature)

	assert sensor.observe_overheating() == "sensor:M1:Temperature:overheating_detected"

	sensor.inject_fault("stuck")

	assert sensor.observe_overheating() is None


def test_temperature_sensor_distinguishes_critical_overheating() -> None:
	env = simpy.Environment()
	temperature = Temperature("M1", value=101.0)
	sensor = TemperatureSensor(env, "M1", temperature)

	assert sensor.observe_overheating() == "sensor:M1:Temperature:critical_overheating_detected"


def test_faulty_temperature_sensor_suppresses_critical_overheating_observations() -> None:
	env = simpy.Environment()
	temperature = Temperature("M1", value=101.0)
	sensor = TemperatureSensor(env, "M1", temperature)
	sensor.inject_fault("stuck")

	assert sensor.observe_overheating() is None


def test_faulty_actuator_sensor_suppresses_actuator_state_observations() -> None:
	env = simpy.Environment()
	actuator = Actuator(env, "M1")
	sensor = ActuatorSensor(env, "M1", actuator)
	actuator.inject_fault("stuck")

	assert sensor.observe_actuator_status() == "sensor:M1:ActuatorSensor:actuator_stuck_detected"

	sensor.inject_fault("no_signal")

	assert sensor.observe_actuator_status() is None
