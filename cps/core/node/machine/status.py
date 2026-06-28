"""Flow and capacity status derived from machine state."""

from typing import TYPE_CHECKING

from cps.core.node import DownstreamStatus
from cps.core.flow.reasons import (
	ACTUATOR_SLOW_RESPONSE,
	ACTUATOR_STUCK,
	DEAD_BATTERY,
	INPUT_CAPACITY,
	OUTPUT_CAPACITY,
	THERMAL_BLOCKED,
)
from cps.core.reporting import actuator_state_id, machine_issue_id

if TYPE_CHECKING:
	from cps.core.node.machine import Machine


class MachineFlowStatus:
	def __init__(self, machine: "Machine") -> None:
		self.machine = machine

	def rejection_reason(self) -> str | None:
		machine = self.machine
		if machine.battery.is_dead:
			return DEAD_BATTERY
		if machine.thermal_blocked:
			return THERMAL_BLOCKED
		if machine.actuator.fault_type == "stuck":
			return ACTUATOR_STUCK
		if machine.fault_type == "jammed_workpiece":
			return "machine_fault_blocked"
		if machine.production_state.handoff_product_id() is not None:
			return OUTPUT_CAPACITY
		if len(machine.inbound_parts) >= machine.input_capacity:
			return INPUT_CAPACITY
		if not machine.production_state.production_schedule:
			return INPUT_CAPACITY
		return None

	def blocking_state_cause(self, reason: str) -> str | None:
		machine = self.machine
		causes: dict[str, str | None] = {
			DEAD_BATTERY: machine.battery.state_id,
			THERMAL_BLOCKED: machine.temperature.state_id,
			ACTUATOR_STUCK: actuator_state_id(machine.id, "stuck"),
			ACTUATOR_SLOW_RESPONSE: actuator_state_id(machine.id, "slow_response"),
			"machine_fault_blocked": f"machine:{machine.id}:jammed_workpiece",
			INPUT_CAPACITY: None,
		}
		return causes.get(reason)

	def capacity_pressure_cause(self) -> str | None:
		if self.machine.fault_type == "bearing_wear":
			return "machine_fault_slowdown"
		return ACTUATOR_SLOW_RESPONSE if self.machine.actuator.fault_type == "slow_response" else None

	def capacity_pressure(self) -> float:
		machine = self.machine
		if self.rejection_reason() is not None:
			return 1.0
		pressures: list[float] = []
		if machine.actuator.fault_type == "slow_response":
			pressures.append(min(machine.actuator.fault_param / (machine.actuator.base_action_time + machine.actuator.fault_param), 1.0))
		if machine.fault_type == "bearing_wear":
			pressures.append(min((machine.fault_param - 1.0) / machine.fault_param, 1.0))
		if machine.outgoing_belt is not None:
			if not machine.outgoing_belt.has_capacity():
				pressures.append(1.0)
			if machine.outgoing_belt.capacity > 0:
				pressures.append(min(len(machine.outgoing_belt.queue) / machine.outgoing_belt.capacity, 1.0))
		if machine.input_capacity <= 0:
			return 1.0
		pressures.append(min(len(machine.inbound_parts) / machine.input_capacity, 1.0))
		return max(pressures, default=0.0)

	def downstream_status(self) -> DownstreamStatus:
		machine = self.machine
		reason = self.rejection_reason()
		state_cause_id = self.blocking_state_cause(reason) if reason is not None else None
		pressure_cause = self.capacity_pressure_cause()
		if state_cause_id is None and pressure_cause == ACTUATOR_SLOW_RESPONSE:
			state_cause_id = actuator_state_id(machine.id, "slow_response")
		if state_cause_id is None and pressure_cause == "machine_fault_slowdown":
			state_cause_id = f"machine:{machine.id}:bearing_wear"
		return DownstreamStatus(
			node_id=machine.id,
			can_accept=reason is None,
			rejection_reason=reason,
			capacity_pressure=self.capacity_pressure(),
			capacity_pressure_cause=pressure_cause,
			production_blocked_issue_id=machine_issue_id(machine.id, "production_blocked"),
			production_slowdown_issue_id=machine_issue_id(machine.id, "production_slowdown"),
			blocking_state_cause_id=state_cause_id,
		)
