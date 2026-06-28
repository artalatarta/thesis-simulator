from collections.abc import Callable, Iterable, Mapping
from typing import Protocol

import simpy

from cps.core.node.machine import Machine


class SimulationDashboard(Protocol):
	"""The slice of the UI dashboard the realtime loop needs."""

	@property
	def machines(self) -> Mapping[str, Machine]: ...

	def display(self) -> None: ...


def run_is_complete(machines: Iterable[Machine]) -> bool:
	return all(machine.production_is_complete for machine in machines)


def recovery_is_complete(machines: Iterable[Machine]) -> bool:
	return all(machine.recovery_is_complete for machine in machines)


def resume_idle_work(machines: Iterable[Machine]) -> None:
	for machine in machines:
		machine.recover_stalled_production()


def run_simulation_loop(
	env: simpy.Environment,
	dashboard: SimulationDashboard,
	*,
	step_duration: float = 1.0,
	is_complete: Callable[[], bool] | None = None,
	after_step: Callable[[], None] | None = None,
) -> None:
	"""Advance the SimPy environment while keeping the terminal dashboard responsive.

	``is_complete`` replaces the default stop condition (production complete)
	so a caller can keep the run alive until, e.g., every open fault is repaired.
	"""
	complete = is_complete if is_complete is not None else (lambda: run_is_complete(dashboard.machines.values()))
	while not complete():
		resume_idle_work(dashboard.machines.values())
		env.run(until=env.now + step_duration)
		if after_step is not None:
			after_step()
		dashboard.display()
