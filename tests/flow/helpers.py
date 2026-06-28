import logging

import pytest
import simpy

from cps.core.flow import BeltSegment
from cps.core.node import FinalStorage, Node
from cps.core.kpi import KPITracker
from cps.core.node.machine import ActiveWorkItem, Machine, WorkLocation
from cps.core.network import Network
from cps.simulation.setup import setup_simulation
from cps.types import CoolingState

__all__ = [
	"ActiveWorkItem",
	"BeltSegment",
	"CoolingState",
	"FinalStorage",
	"KPITracker",
	"Machine",
	"Network",
	"Node",
	"WorkLocation",
	"active_product",
	"link_output",
	"make_environment",
	"make_machine",
	"logging",
	"prime_input",
	"pytest",
	"setup_simulation",
	"simpy",
	"start_machines",
]


def make_environment() -> tuple[simpy.Environment, KPITracker, Network]:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	return env, kpi_tracker, network


def make_machine(
	env: simpy.Environment,
	machine_id: str = "M1",
	schedule: list[tuple[str, float]] | None = None,
	network: Network | None = None,
	kpi_tracker: KPITracker | None = None,
	input_capacity: int = 1,
) -> Machine:
	resolved_network = network or Network(env)
	resolved_kpi_tracker = kpi_tracker or KPITracker(env)
	return Machine(env, machine_id, schedule or [], resolved_network, resolved_kpi_tracker, input_capacity=input_capacity)


def start_machines(*machines: Machine) -> None:
	for machine in machines:
		machine.start()


def link_output(env: simpy.Environment, machine: Machine, target: Node, network: Network, capacity: int = 1) -> BeltSegment:
	machine.outgoing_belt = BeltSegment(env, machine, target, network, capacity=capacity)
	return machine.outgoing_belt


def active_product(machine: Machine, location: WorkLocation) -> str | None:
	if machine.production_state.active_work is None or machine.production_state.active_work.location is not location:
		return None
	return machine.production_state.active_work.product_id


def prime_input(machine: Machine, *product_ids: str) -> None:
	machine.inbound_parts.extend(product_ids)
