"""Construction of the physical components attached to a machine."""

from dataclasses import dataclass

import simpy

from cps.components.actuators import Actuator
from cps.components.battery import Battery
from cps.components.sensors import ActuatorSensor, PowerSensor, TemperatureSensor
from cps.components.temperature import Temperature
from cps.core.reporting import EventReporter


@dataclass(frozen=True)
class MachineComponents:
	temperature: Temperature
	temperature_sensor: TemperatureSensor
	battery: Battery
	power_sensor: PowerSensor
	actuator: Actuator
	actuator_sensor: ActuatorSensor

	def faultable(self) -> tuple[TemperatureSensor, PowerSensor, ActuatorSensor, Actuator]:
		return self.temperature_sensor, self.power_sensor, self.actuator_sensor, self.actuator


def build_machine_components(env: simpy.Environment, machine_id: str, event_reporter: EventReporter) -> MachineComponents:
	temperature = Temperature(machine_id)
	battery = Battery(machine_id)
	actuator = Actuator(env, machine_id, event_reporter)
	return MachineComponents(
		temperature=temperature,
		temperature_sensor=TemperatureSensor(env, machine_id, temperature, event_reporter),
		battery=battery,
		power_sensor=PowerSensor(env, machine_id, battery, event_reporter),
		actuator=actuator,
		actuator_sensor=ActuatorSensor(env, machine_id, actuator, event_reporter),
	)
