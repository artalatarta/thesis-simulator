"""Production-loop orchestration for :class:`cps.core.node.machine.Machine`."""

import logging
from typing import TYPE_CHECKING

import simpy

from cps.core.node import FinalStorage
from cps.core.processes import process_is_alive
from cps.core.production_state import WorkLocation
from cps.core.reporting import belt_issue_id, machine_issue_id
from cps.types import ProcessGenerator

if TYPE_CHECKING:
	from cps.core.node.machine import Machine


class MachineProductionRunner:
	def __init__(self, machine: "Machine") -> None:
		self.machine = machine

	def run(self) -> ProcessGenerator:
		machine = self.machine
		try:
			while not machine.battery.is_dead and not machine.thermal_blocked and machine.actuator.fault_type != "stuck" and machine.fault_type != "jammed_workpiece":
				handoff_product_id = machine.production_state.handoff_product_id()
				if handoff_product_id is not None:
					handoff_completed = yield self.start_child_process(self.deliver_completed_output(handoff_product_id))
					if not handoff_completed:
						break
					continue
				work_item = yield from self.next_work_item()
				if work_item is None:
					break
				machine.production_state.start_work(work_item)
				production_completed = yield self.start_child_process(self.produce_part(work_item.product_id, work_item.process_time))
				if not production_completed:
					break
				machine.production_state.clear_active_work()
		except simpy.Interrupt:
			self.handle_interrupt()

	def start_child_process(self, generator: ProcessGenerator) -> simpy.Process:
		machine = self.machine
		process = machine.env.process(generator)
		machine._active_child_process = process
		process.callbacks.append(lambda _: self.clear_child_process(process))
		return process

	def clear_child_process(self, process: simpy.Process) -> None:
		machine = self.machine
		if machine._active_child_process is process:
			machine._active_child_process = None
		if not process.ok:
			process.defused = True

	def next_work_item(self) -> ProcessGenerator:
		machine = self.machine
		if machine.production_state.active_work is not None and machine.production_state.active_work.location is WorkLocation.WORK_IN_PROGRESS:
			return machine.production_state.next_work()
		if not machine.production_state.production_schedule:
			return None
		while not machine.inbound_parts and not machine.battery.is_dead and not machine.thermal_blocked and machine.actuator.fault_type != "stuck" and machine.fault_type != "jammed_workpiece":
			yield machine._input_available
		if machine.battery.is_dead or machine.thermal_blocked or machine.actuator.fault_type == "stuck" or machine.fault_type == "jammed_workpiece":
			return None
		return machine.production_state.next_work()

	def produce_part(self, product_id: str, process_time: float) -> ProcessGenerator:
		machine = self.machine
		if machine.battery.is_dead:
			machine.production_state.restore_work_item()
			logging.critical(f"{machine.id}: Cannot start {product_id}; {machine.battery.state_id}.", extra={"component": machine.id})
			return False
		if machine.thermal_blocked:
			machine.production_state.restore_work_item()
			logging.critical(f"{machine.id}: Cannot start {product_id}; {machine.temperature.state_id}.", extra={"component": machine.id})
			return False
		if machine.fault_type == "jammed_workpiece":
			machine.production_state.restore_work_item()
			logging.critical(f"{machine.id}: Cannot start {product_id}; machine production is blocked.", extra={"component": machine.id})
			return False
		if machine.actuator.fault_type == "stuck":
			machine.production_state.restore_work_item()
			logging.critical(f"{machine.id}: Cannot start {product_id}; actuator is stuck.", extra={"component": machine.id})
			return False
		self.start_part(product_id)
		action_completed = yield machine.env.process(machine.actuator.perform_action(f"Feed part for {product_id}"))
		if not action_completed:
			machine.production_state.restore_work_item()
			self.stop_active_processing()
			return False
		machine.production_state.mark_active_input_as_work_in_progress()
		effective_process_time = process_time * machine.temperature.process_time_factor * machine.fault_param
		yield machine.env.timeout(effective_process_time)
		self.complete_processing(product_id)
		yield machine.env.process(self.deliver_completed_output(product_id))
		return True

	def deliver_completed_output(self, product_id: str) -> ProcessGenerator:
		machine = self.machine
		belt = machine.outgoing_belt
		if belt is not None:
			machine.production_state.prepare_handoff(product_id)
			# The belt's delivered-parts ledger absorbs deliveries that complete
			# after this waiting process was interrupted, so a restarted run finds
			# the part at the loop head instead of handing it off a second time.
			while not belt.consume_delivered_part(product_id):
				if belt.has_queued_part(product_id):
					delivered = yield machine.env.process(belt.deliver_queued_part(product_id))
				else:
					delivered = yield machine.env.process(belt.handoff(product_id))
				if delivered:
					continue
				machine.event_reporter.derived_issue(
					machine_issue_id(machine.id, "production_blocked"),
					component=machine.id,
					cause_id=belt_issue_id(machine.id, belt.to_node.id, "handoff_blocked"),
				)
				yield machine.env.timeout(1)
		if belt is None or isinstance(belt.to_node, FinalStorage):
			machine.kpi_tracker.track_production()
		machine.parts_produced += 1
		machine.production_state.clear_active_work()
		return True

	def start_part(self, product_id: str) -> None:
		machine = self.machine
		machine.is_processing = True
		machine.kpi_tracker.track_machine_state_change(machine.id, True)
		logging.info(f"{machine.id}: Starting production of {product_id}.", extra={"component": machine.id})

	def complete_processing(self, product_id: str) -> None:
		machine = self.machine
		logging.info(f"{machine.id}: Finished production of {product_id}.", extra={"component": machine.id})
		machine.production_state.prepare_handoff(product_id)
		machine.is_processing = False
		machine.kpi_tracker.track_machine_state_change(machine.id, False)

	def handle_interrupt(self) -> None:
		machine = self.machine
		logging.warning(f"INTERRUPTED: Production process for {machine.id} was interrupted by agent.", extra={"component": machine.id})
		if process_is_alive(machine._active_child_process):
			machine._active_child_process.interrupt()
		active_output_is_complete = machine.production_state.active_output_is_complete(machine.outgoing_belt)
		if machine.production_state.active_work is not None and machine.production_state.active_work.schedule_entry is not None and not active_output_is_complete:
			logging.info(f"Restoring product {machine.production_state.active_work.product_id} to schedule for {machine.id}.", extra={"component": machine.id})
			machine.production_state.restore_work_item()
		elif active_output_is_complete and machine.production_state.active_work is not None:
			machine.production_state.prepare_handoff()
		self.stop_active_processing()

	def stop_active_processing(self) -> None:
		machine = self.machine
		if machine.is_processing:
			machine.kpi_tracker.track_machine_state_change(machine.id, False)
		machine.is_processing = False
