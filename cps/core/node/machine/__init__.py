import logging
import random
from collections.abc import Sequence
from typing import Literal

import simpy

from cps.components.actuators import Actuator
from cps.components.sensors import ActuatorSensor, PowerSensor, TemperatureSensor
from cps.core.flow import BeltSegment
from cps.core.node import DownstreamStatus, Node
from cps.core.node.machine.components import build_machine_components
from cps.core.kpi import KPITracker
from cps.core.node.machine.production import MachineProductionRunner
from cps.core.node.machine.status import MachineFlowStatus
from cps.core.network import Network
from cps.core.processes import process_is_alive
from cps.core.production_state import ActiveWorkItem, PendingWorkItem as PendingWorkItem, ProductionState, WorkLocation
from cps.core.reporting import EventReporter, machine_issue_id
from cps.types import ProcessGenerator, ScheduleEntry

MachineFaultType = Literal["bearing_wear", "jammed_workpiece"]
MACHINE_FAULT_TYPES: tuple[MachineFaultType, ...] = ("bearing_wear", "jammed_workpiece")
MACHINE_SLOWDOWN_MIN_FACTOR = 1.5
MACHINE_SLOWDOWN_MAX_FACTOR = 2.5
MACHINE_REPAIR_MIN_TIME = 4.0
MACHINE_REPAIR_MAX_TIME = 6.0

__all__ = [
	"ActiveWorkItem",
	"MACHINE_FAULT_TYPES",
	"MACHINE_SLOWDOWN_MAX_FACTOR",
	"MACHINE_SLOWDOWN_MIN_FACTOR",
	"Machine",
	"MachineFaultType",
	"PendingWorkItem",
	"WorkLocation",
	"report_machine_fault_issue",
]


def report_machine_fault_issue(event_reporter: EventReporter, machine_id: str, fault_type: MachineFaultType, fault_param: float) -> None:
	"""Emit the derived production issue for an active machine fault."""
	issue = "production_slowdown" if fault_type == "bearing_wear" else "production_blocked"
	event_reporter.derived_issue(
		machine_issue_id(machine_id, issue),
		component=machine_id,
		cause_id=f"machine:{machine_id}:{fault_type}",
		context={"slowdown_factor": fault_param} if fault_type == "bearing_wear" else {},
	)


class Machine(Node):
	def __init__(
		self,
		env: simpy.Environment,
		machine_id: str,
		production_schedule: Sequence[ScheduleEntry],
		network: Network,
		kpi_tracker: KPITracker,
		input_capacity: int = 1,
	) -> None:
		self.env = env
		self.id = machine_id
		self.inbound_parts: list[str] = []
		self.input_capacity = input_capacity
		self.production_state = ProductionState(production_schedule, self.inbound_parts)
		self._input_available = env.event()
		self.network = network
		self.event_reporter = network.event_reporter
		self.incoming_belt: BeltSegment | None = None
		self.outgoing_belt: BeltSegment | None = None

		self.is_processing = False
		self.parts_produced = 0
		self.fault_type: MachineFaultType | None = None
		self.fault_param = 1.0
		self.pending_repair: MachineFaultType | None = None
		self._active_child_process: simpy.Process | None = None
		self._idle_restart_requested = False

		components = build_machine_components(env, machine_id, self.event_reporter)
		self.temperature = components.temperature
		self.temperature_sensor = components.temperature_sensor
		self.battery = components.battery
		self.power_sensor = components.power_sensor
		self.actuator = components.actuator
		self.actuator_sensor = components.actuator_sensor
		self._faultable_components = components.faultable()

		self.kpi_tracker = kpi_tracker
		self.flow_status = MachineFlowStatus(self)
		self.production_runner = MachineProductionRunner(self)
		self.production_process: simpy.Process | None = None
		self.physics_process: simpy.Process | None = None

	def faultable_components(self) -> tuple[TemperatureSensor, PowerSensor, ActuatorSensor, Actuator, "Machine"]:
		return (*self._faultable_components, self)

	def inject_fault(self, fault_type: MachineFaultType) -> None:
		self.fault_type = fault_type
		self.fault_param = random.uniform(MACHINE_SLOWDOWN_MIN_FACTOR, MACHINE_SLOWDOWN_MAX_FACTOR) if fault_type == "bearing_wear" else 1.0
		fault_id = f"machine:{self.id}:{fault_type}"
		self.event_reporter.root_fault(fault_id, message=f"FAULT INJECTED {fault_id}")
		report_machine_fault_issue(self.event_reporter, self.id, fault_type, self.fault_param)
		if fault_type == "jammed_workpiece":
			self.block_production(fault_id)

	def inject_random_fault(self) -> tuple[str, str]:
		self.inject_fault(random.choice(MACHINE_FAULT_TYPES))
		return self.id, "Machine"

	def clear_fault(self, fault_type: str) -> bool:
		if self.fault_type != fault_type:
			return False
		logging.info(f"Corrective action: Clearing fault on machine:{self.id}", extra={"component": "System"})
		self.event_reporter.fault_resolved(f"machine:{self.id}:{self.fault_type}", component=self.id)
		self.fault_type = None
		self.fault_param = 1.0
		self.kpi_tracker.track_fault_end(self.id, "Machine")
		self.resume_production_if_ready()
		return True

	def dispatch_repair(self, fault_type: MachineFaultType) -> bool:
		if self.fault_type != fault_type:
			return False
		logging.info(f"AGENT ACTION: Dispatching machine repair for {self.id} ({fault_type})", extra={"component": self.id})
		self.clear_fault(fault_type)
		return True

	def _sample_repair_time(self) -> float:
		if MACHINE_REPAIR_MIN_TIME > MACHINE_REPAIR_MAX_TIME:
			raise ValueError("MACHINE_REPAIR_MIN_TIME must be less than or equal to MACHINE_REPAIR_MAX_TIME")
		return random.uniform(MACHINE_REPAIR_MIN_TIME, MACHINE_REPAIR_MAX_TIME)

	def start(self) -> None:
		if not self.production_process_is_alive():
			self._spawn_production()
		if not process_is_alive(self.physics_process):
			self.physics_process = self.env.process(self._run_physics())

	def _spawn_production(self) -> None:
		self.production_process = self.env.process(self._run_production())

	def _run_physics(self) -> ProcessGenerator:
		"""Advance this machine's own physical state once per tick.

		Battery depletion and thermal evolution are intrinsic to the machine, not
		to whether a monitoring agent is watching it. Driving them here keeps the
		physics independently of the monitoring/recovery layer.
		"""
		while True:
			if not self.battery.is_dead:
				self.battery.drain(self.is_processing)
			was_shutdown = self.temperature.is_shutdown
			self.temperature.update(is_processing=self.is_processing)
			if self.temperature.is_shutdown and not was_shutdown:
				# Hard thermal cutoff: stop any in-flight processing immediately.
				self.block_production(self.temperature.state_id)
			elif was_shutdown and not self.temperature.is_shutdown:
				# Cooled back to the safe threshold; the cutoff releases on its own.
				self.resume_production_if_ready()
			yield self.env.timeout(1)

	@property
	def thermal_blocked(self) -> bool:
		return self.temperature.is_thermal_blocked

	@property
	def production_is_complete(self) -> bool:
		if self.production_state.production_schedule or self.inbound_parts:
			return False
		if self.production_state.active_work is not None or self.is_processing:
			return False
		return self.outgoing_belt is None or not self.outgoing_belt.queue

	@property
	def recovery_is_complete(self) -> bool:
		"""True when no fault is open and no repair is in flight on this machine,
		its belts, or the shared network. Judged from component state directly so
		run completion does not depend on the KPI tracker."""
		if any(component.fault_type is not None for component in self.faultable_components()):
			return False
		if any(belt is not None and belt.fault_type is not None for belt in (self.incoming_belt, self.outgoing_belt)):
			return False
		if any(belt is not None and belt.pending_repair is not None for belt in (self.incoming_belt, self.outgoing_belt)):
			return False
		return (
			self.network.fault_type is None
			and not self.network.pending_repairs
			and not self.battery.pending_replacement
			and self.actuator.pending_repair is None
			and self.power_sensor.pending_repair is None
			and self.temperature_sensor.pending_repair is None
			and self.actuator_sensor.pending_repair is None
			and self.pending_repair is None
		)

	def can_accept_part(self) -> bool:
		return self.rejection_reason() is None

	def block_production(self, cause_id: str | None) -> None:
		"""Report production blocked by an internal fault and interrupt the production process."""
		self.event_reporter.derived_issue(
			machine_issue_id(self.id, "production_blocked"),
			component=self.id,
			cause_id=cause_id,
		)
		if process_is_alive(self.production_process):
			self.production_process.interrupt()

	def blocking_state_cause(self, reason: str) -> str | None:
		return self.flow_status.blocking_state_cause(reason)

	def capacity_pressure_cause(self) -> str | None:
		return self.flow_status.capacity_pressure_cause()

	def capacity_pressure(self) -> float:
		return self.flow_status.capacity_pressure()

	def downstream_status(self) -> DownstreamStatus:
		return self.flow_status.downstream_status()

	def rejection_reason(self) -> str | None:
		return self.flow_status.rejection_reason()

	def receive_part(self, product_id: str) -> bool:
		if not self.can_accept_part():
			return False
		self.inbound_parts.append(product_id)
		if not self._input_available.triggered:
			self._input_available.succeed()
		self._input_available = self.env.event()
		self.resume_production_if_ready()
		logging.info(f"{self.id}: Received {product_id} from incoming belt.", extra={"component": self.id})
		return True

	def can_resume_production(self) -> bool:
		return (
			not self.battery.is_dead
			and not self.thermal_blocked
			and self.actuator.fault_type != "stuck"
			and self.fault_type != "jammed_workpiece"
			and self.production_state.has_pending_work()
		)

	def resume_production_if_ready(self) -> bool:
		if self.production_process_is_alive():
			return False
		self._idle_restart_requested = False
		if not self.can_resume_production():
			return False
		self._spawn_production()
		return True

	def recover_stalled_production(self) -> None:
		"""Single recovery entrypoint for the per-tick simulation loop.

		Wakes any blocked input wait, then either restarts a stalled idle process
		or resumes production if the machine is ready.
		"""
		if not self.inbound_parts and not self.production_state.has_pending_work():
			# Idle: nothing to wake or restart. Drop a stale idle-restart request
			# once the interrupted process dies so it cannot block a later restart.
			if not self.production_process_is_alive():
				self._idle_restart_requested = False
			return
		self._wake_pending_work()
		if self._restart_if_idle_process_is_stalled():
			return
		self.resume_production_if_ready()

	def _wake_pending_work(self) -> None:
		if not self.inbound_parts:
			return
		if self._input_available.triggered:
			return
		self._input_available.succeed()
		self._input_available = self.env.event()

	def _restart_if_idle_process_is_stalled(self) -> bool:
		process = self.production_process
		if self._idle_restart_requested or not process_is_alive(process):
			return False
		if self.is_processing or self.production_state.active_work is not None:
			return False
		if not self.can_resume_production():
			return False
		self._idle_restart_requested = True
		process.interrupt()
		return True

	def _run_production(self) -> ProcessGenerator:
		return (yield from self.production_runner.run())

	def production_process_is_alive(self) -> bool:
		return process_is_alive(self.production_process)
