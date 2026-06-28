import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import simpy

from cps.config import MEAN_TIME_BETWEEN_FAULTS
from cps.core.flow import BeltSegment
from cps.core.kpi import KPITracker
from cps.core.network import Network
from cps.core.node import FinalStorage, RawMaterialSource
from cps.core.node.machine import Machine
from cps.core.reporting import EventReporter
from cps.simulation.factory_line_config import FactoryLineConfig
from cps.types import ProcessGenerator


class FaultableComponent(Protocol):
	@property
	def fault_type(self) -> object | None: ...

	def inject_random_fault(self) -> tuple[str, str]: ...


@dataclass(frozen=True)
class FactoryLine:
	machines: dict[str, Machine]
	faultable_components: list[FaultableComponent]
	network: Network
	raw_material_source: RawMaterialSource
	final_storage: FinalStorage
	belt_segments: list[BeltSegment]
	event_reporter: EventReporter


class FactoryLineBuilder:
	def __init__(self, env: simpy.Environment, kpi_tracker: KPITracker, config: FactoryLineConfig, event_reporter: EventReporter) -> None:
		self.env = env
		self.kpi_tracker = kpi_tracker
		self.config = config
		self.event_reporter = event_reporter
		self.network = Network(env, event_reporter)
		self.machines: dict[str, Machine] = {}
		self.faultable_components: list[FaultableComponent] = [self.network]
		self.raw_material_source = RawMaterialSource(config.source_id)
		self.final_storage = FinalStorage(config.storage_id, env=env)
		self.belt_segments: list[BeltSegment] = []

	def build(self) -> FactoryLine:
		self._create_machines()
		self.kpi_tracker.initialize_machine_states(self.machines.keys())
		self._link_machines()
		first_machine = next(iter(self.machines.values()), None)
		if first_machine is not None:
			product_ids = [product_id for product_id, _ in first_machine.production_state.production_schedule]
			self.raw_material_source.start_feeding(self.env, first_machine, product_ids)
		for machine in self.machines.values():
			machine.start()
		return FactoryLine(
			self.machines,
			self.faultable_components,
			self.network,
			self.raw_material_source,
			self.final_storage,
			self.belt_segments,
			self.event_reporter,
		)

	def _create_machines(self) -> None:
		for station in self.config.stations:
			machine_id = station[0]
			machine = Machine(self.env, machine_id, self.config.schedule_for(station), self.network, self.kpi_tracker)
			self.machines[machine_id] = machine
			self.faultable_components.extend(machine.faultable_components())
			logging.info(f"Created machine: {machine_id}", extra={"component": "Setup"})

	def _link_machines(self) -> None:
		machines = list(self.machines.values())
		if not machines:
			return

		# The raw-material source feeds the first machine directly, without a belt.
		# An infeed conveyor here would be an infinite boundary feed (a blank/coil
		# stack into the press) that never starves and feeds the bottleneck station,
		# so it would only ever emit backpressure confounded with the machine being
		# busy. The outfeed belt to FinalStorage is kept because its sink always
		# accepts, so a jam or slippage there backs up the line and stays observable.
		logging.info(f"Linked {self.raw_material_source.id} -> {machines[0].id} (direct feed, no belt)", extra={"component": "Setup"})

		for machine, next_machine in zip(machines, machines[1:]):
			next_machine.incoming_belt = BeltSegment(self.env, machine, next_machine, self.network)
			machine.outgoing_belt = next_machine.incoming_belt
			self.belt_segments.append(next_machine.incoming_belt)
			self.faultable_components.append(next_machine.incoming_belt)
			logging.info(f"Linked {machine.id} -> {next_machine.id}", extra={"component": "Setup"})

		last_machine = machines[-1]
		last_machine.outgoing_belt = BeltSegment(self.env, last_machine, self.final_storage, self.network)
		self.belt_segments.append(last_machine.outgoing_belt)
		self.faultable_components.append(last_machine.outgoing_belt)
		logging.info(f"Linked {last_machine.id} -> {self.final_storage.id}", extra={"component": "Setup"})


def fault_injector(
	env: simpy.Environment,
	all_components: list[FaultableComponent],
	kpi_tracker: KPITracker,
	should_inject: Callable[[], bool],
) -> ProcessGenerator:
	while should_inject():
		yield env.timeout(random.expovariate(1.0 / MEAN_TIME_BETWEEN_FAULTS))
		# Let production and handoff events scheduled for this same simulation
		# timestamp settle before injecting another fault.
		yield env.timeout(0)
		if not should_inject():
			return
		# Only target components that are currently fault-free. Re-injecting on an
		# already-faulted component would overwrite its open-fault start time in the
		# KPI tracker (biasing MTTR low) and emit a duplicate root_fault event.
		available_components = list(filter(lambda component: component.fault_type is None, all_components))

		if not available_components:
			continue

		target_component = random.choice(available_components)
		owner, component = target_component.inject_random_fault()
		kpi_tracker.track_fault_start(owner, component)


def setup_simulation(
	env: simpy.Environment,
	kpi_tracker: KPITracker,
	factory_config: FactoryLineConfig,
	event_reporter: EventReporter,
) -> FactoryLine:
	event_reporter.clear()
	return FactoryLineBuilder(env, kpi_tracker, factory_config, event_reporter).build()
