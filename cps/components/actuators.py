import logging
import random
from collections.abc import Callable

import simpy

from cps.core.kpi import KPITracker
from cps.core.reporting import EventReporter, actuator_state_id, machine_issue_id
from cps.types import ActuatorFaultType, ProcessGenerator

ACTUATOR_FAULT_TYPES: tuple[ActuatorFaultType, ...] = ("slow_response", "stuck")
ACTUATOR_REPAIR_MIN_TIME = 4.0
ACTUATOR_REPAIR_MAX_TIME = 6.0
# Slow-response actuator fault adds a delay factor sampled from this range.
ACTUATOR_SLOW_RESPONSE_MIN_FACTOR = 0.5
ACTUATOR_SLOW_RESPONSE_MAX_FACTOR = 2.0


class Actuator:
	def __init__(self, env: simpy.Environment, machine_id: str, event_reporter: EventReporter | None = None) -> None:
		self.env = env
		self.machine_id = machine_id
		self.event_reporter = event_reporter or EventReporter()
		self.fault_type: ActuatorFaultType | None = None
		self.fault_param = 0.0
		self.base_action_time = 0.1
		self.pending_repair: ActuatorFaultType | None = None

	def perform_action(self, action: str) -> ProcessGenerator:
		if self.fault_type == "stuck":
			state_id = actuator_state_id(self.machine_id, "stuck")
			self.event_reporter.derived_issue(
				machine_issue_id(self.machine_id, "production_blocked"),
				component="Actuator",
				cause_id=state_id,
			)
			logging.critical(f"CRITICAL FAULT {self.machine_id}: Actuator is STUCK trying to '{action}'.", extra={"component": "Actuator"})
			yield self.env.timeout(0)
			return False
		action_time = self.base_action_time
		if self.fault_type == "slow_response":
			action_time += self.fault_param
			state_id = actuator_state_id(self.machine_id, "slow_response")
			self.event_reporter.derived_issue(
				machine_issue_id(self.machine_id, "production_slowdown"),
				component="Actuator",
				cause_id=state_id,
			)
			logging.warning(f"{self.machine_id}: Actuator is slow. Action '{action}' taking {action_time:.2f}s.", extra={"component": "Actuator"})
		else:
			logging.info(f"{self.machine_id}: Actuator performing action: '{action}'.", extra={"component": "Actuator"})
		yield self.env.timeout(action_time)
		return True

	def inject_fault(self, fault_type: ActuatorFaultType) -> None:
		self.fault_type = fault_type
		if self.fault_type == "slow_response":
			self.fault_param = random.uniform(ACTUATOR_SLOW_RESPONSE_MIN_FACTOR, ACTUATOR_SLOW_RESPONSE_MAX_FACTOR)
		fault_id = actuator_state_id(self.machine_id, self.fault_type)
		self.event_reporter.root_fault(fault_id, message=f"FAULT INJECTED {fault_id}")
		if self.fault_type == "stuck":
			# A stuck actuator blocks intake immediately, independent of detection or
			# of a part being mid-transfer (faults.md: "regardless of detection, the
			# machine cannot accept input -> machine:<id>:production_blocked"). Emit the
			# derived issue at injection so the symptom surfaces even when perform_action
			# is not currently being driven; per-tick duplicates are collapsed downstream.
			self.event_reporter.derived_issue(
				machine_issue_id(self.machine_id, "production_blocked"),
				component="Actuator",
				cause_id=actuator_state_id(self.machine_id, "stuck"),
			)

	def inject_random_fault(self) -> tuple[str, str]:
		self.inject_fault(random.choice(ACTUATOR_FAULT_TYPES))
		return self.machine_id, "Actuator"

	def clear_fault(self) -> None:
		logging.info(f"Corrective action: Clearing fault on actuator:{self.machine_id}", extra={"component": "System"})
		if self.fault_type is not None:
			self.event_reporter.fault_resolved(actuator_state_id(self.machine_id, self.fault_type), component=self.machine_id)
		self.fault_type = None
		self.fault_param = 0.0

	def dispatch_repair(
		self,
		kpi_tracker: KPITracker,
		*,
		fault_type: ActuatorFaultType,
		sensor_fault_type: str | None = None,
		require_sensor_operational: bool = True,
		after_stuck_cleared: Callable[[], object] | None = None,
	) -> bool:
		if require_sensor_operational and sensor_fault_type is not None:
			return False
		if self.fault_type != fault_type:
			return False
		logging.info(f"AGENT ACTION: Dispatching actuator repair for {self.machine_id} ({fault_type})", extra={"component": self.machine_id})
		logging.info(f"Corrective action: Actuator for {self.machine_id} repaired {fault_type}.", extra={"component": "System"})
		self.clear_fault()
		kpi_tracker.track_fault_end(self.machine_id, "Actuator")
		if fault_type == "stuck" and after_stuck_cleared is not None:
			after_stuck_cleared()
		return True

	def _sample_repair_time(self) -> float:
		if ACTUATOR_REPAIR_MIN_TIME > ACTUATOR_REPAIR_MAX_TIME:
			raise ValueError("ACTUATOR_REPAIR_MIN_TIME must be less than or equal to ACTUATOR_REPAIR_MAX_TIME")
		return random.uniform(ACTUATOR_REPAIR_MIN_TIME, ACTUATOR_REPAIR_MAX_TIME)
