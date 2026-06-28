"""Physical-state observation loops used by machine-bound monitoring agents."""

import logging
from typing import TYPE_CHECKING

from cps.config import OBSERVATION_MONITOR_INTERVAL
from cps.core.node.machine import report_machine_fault_issue
from cps.core.reporting import machine_issue_id
from cps.types import CoolingState, ProcessGenerator

if TYPE_CHECKING:
	from cps.core.node.machine import Machine


class BatteryStateObserver:
	def __init__(self, machine: "Machine") -> None:
		self.machine = machine

	def monitor(self) -> ProcessGenerator:
		last_reported_state: str | None = None
		while True:
			battery = self.machine.battery
			state = "dead" if battery.is_dead else "low" if battery.is_low else None
			if state is not None and state != last_reported_state:
				self._report_state(dead=state == "dead")
			last_reported_state = state
			yield self.machine.env.timeout(1)

	def _report_state(self, *, dead: bool) -> None:
		battery = self.machine.battery
		state_id = battery.state_id
		if state_id is None:
			return
		self.machine.event_reporter.physical_state(
			state_id,
			component=self.machine.id,
			message=f"{state_id}: Level: {battery.level:.2f}%",
			context={"level": battery.level},
		)
		if dead:
			self.machine.block_production(state_id)


class MachineFaultSymptomObserver:
	"""Re-emit a persistent machine fault each observation interval.

	The injection-time derived issue fires once, giving monitoring a single
	polling window to act on the cause before it disappears from view. While the
	root fault persists it stays physically observable, so each interval re-emits
	both the production-flow symptom (production_blocked/slowdown) and a
	fault-naming ``*_detected`` observation. The detection mirrors the sensor and
	network monitors: it names the fault directly so the machine-health agent can
	cite it and dispatch the matching repair, rather than only seeing a generic
	flow symptom it is told not to act on.
	"""

	def __init__(self, machine: "Machine") -> None:
		self.machine = machine

	def monitor(self) -> ProcessGenerator:
		while True:
			yield self.machine.env.timeout(OBSERVATION_MONITOR_INTERVAL)
			fault_type = self.machine.fault_type
			if fault_type is None:
				continue
			report_machine_fault_issue(self.machine.event_reporter, self.machine.id, fault_type, self.machine.fault_param)
			self.machine.event_reporter.observation(machine_issue_id(self.machine.id, f"{fault_type}_detected"), component=self.machine.id)


class TemperatureStateObserver:
	def __init__(self, machine: "Machine") -> None:
		self.machine = machine

	def monitor(self) -> ProcessGenerator:
		last_state_id: str | None = None
		while True:
			temperature = self.machine.temperature
			state_id = temperature.state_id
			if state_id and state_id != last_state_id:
				self._report_state(state_id)
			last_state_id = state_id
			if temperature.cooling_state is CoolingState.LIGHT and temperature.is_safe:
				self._complete_light_cooling()
			yield self.machine.env.timeout(1)

	def _report_state(self, state_id: str) -> None:
		temperature = self.machine.temperature
		self.machine.event_reporter.physical_state(
			state_id,
			component=self.machine.id,
			message=f"{state_id}: Temperature: {temperature.value:.2f}C",
			context={"temperature": temperature.value},
		)

	def _complete_light_cooling(self) -> None:
		logging.info(f"{self.machine.id}: Light cooling complete.", extra={"component": self.machine.id})
		self.machine.temperature.stop_cooling()

	def complete_intense_cooling(self) -> ProcessGenerator:
		temperature = self.machine.temperature
		while not temperature.is_safe:
			yield self.machine.env.timeout(1)
		logging.info(f"{self.machine.id}: Intense cooling complete.", extra={"component": self.machine.id})
		temperature.stop_cooling()
		self.machine.resume_production_if_ready()
