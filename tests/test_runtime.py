import simpy
from typing import Any, cast

from cps.core.flow import BeltSegment
from cps.core.node import FinalStorage
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from cps.simulation.runtime import resume_idle_work, run_is_complete, run_simulation_loop


def test_production_is_incomplete_while_schedule_has_work() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [("P1", 1.0)], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.inbound_parts.append("P1")
	machine.start()

	assert not machine.production_is_complete


def test_production_is_complete_after_schedule_and_handoff_drain() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [("P1", 1.0)], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.inbound_parts.append("P1")
	machine.start()

	env.run(until=2)

	assert machine.production_is_complete
	assert run_is_complete([machine])


def test_production_is_incomplete_while_belt_queue_has_work() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [], network, kpi_tracker)
	machine.outgoing_belt = BeltSegment(env, machine, FinalStorage(), network)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.start()
	assert machine.outgoing_belt is not None
	machine.outgoing_belt.queue.append("P1")

	assert not machine.production_is_complete


def test_run_is_complete_while_battery_replacement_is_pending() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.start()

	machine.battery.pending_replacement = True

	assert machine.production_is_complete
	assert not machine.recovery_is_complete
	assert run_is_complete([machine])


def test_run_is_complete_while_actuator_repair_is_pending() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.start()

	machine.actuator.pending_repair = "stuck"

	assert machine.production_is_complete
	assert not machine.recovery_is_complete
	assert run_is_complete([machine])


def test_run_is_complete_while_network_repair_is_pending() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.start()

	network.pending_repairs.add("packet_loss")

	assert machine.production_is_complete
	assert not machine.recovery_is_complete
	assert run_is_complete([machine])


def test_run_is_complete_while_fault_is_open() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.start()

	machine.inject_fault("bearing_wear")

	assert machine.production_is_complete
	assert not machine.recovery_is_complete
	assert run_is_complete([machine])


def test_resume_idle_work_restarts_ready_machine_with_pending_schedule() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	machine = Machine(env, "M1", [], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.start()

	env.run(until=0.1)
	assert machine.production_process is not None
	assert not machine.production_process.is_alive
	machine.production_state.production_schedule = [("P1", 1.0)]
	machine.inbound_parts.append("P1")

	resume_idle_work([machine])
	env.run(until=2)

	assert machine.parts_produced == 1
	assert machine.production_is_complete


def test_simulation_loop_runs_without_sleeping() -> None:
	env = simpy.Environment()
	display_calls = []

	class Dashboard:
		def __init__(self) -> None:
			self.machines = {"M1": _CompletesAfterOneStepMachine(env)}

		def display(self) -> None:
			display_calls.append(env.now)

	run_simulation_loop(env, cast(Any, Dashboard()))

	assert display_calls == [1.0]


class _CompletesAfterOneStepMachine:
	def __init__(self, env: simpy.Environment) -> None:
		self.env = env
		self.inbound_parts = []
		self.is_processing = False
		self.outgoing_belt = None
		self.production_state = _EmptiesAfterOneStepProductionState(env)
		self.battery = _NoPendingBattery()
		self.actuator = _NoPendingActuator()
		self.network = Network(env)

	def recover_stalled_production(self) -> None:
		pass

	@property
	def production_is_complete(self) -> bool:
		return not self.production_state.production_schedule

	@property
	def recovery_is_complete(self) -> bool:
		return True


class _EmptiesAfterOneStepProductionState:
	active_work = None

	def __init__(self, env: simpy.Environment) -> None:
		self.env = env

	@property
	def production_schedule(self) -> list[tuple[str, float]]:
		if self.env.now >= 1.0:
			return []
		return [("P1", 1.0)]


class _NoPendingBattery:
	pending_replacement = False


class _NoPendingActuator:
	pending_repair = None


def test_simulation_loop_keeps_stepping_until_run_is_complete() -> None:
	env = simpy.Environment()
	display_calls = []

	class Dashboard:
		def __init__(self) -> None:
			self.machines = {"M1": _CompletesAfterOneStepMachine(env)}

		def display(self) -> None:
			display_calls.append(env.now)

	# Production completes after one step, but an open fault keeps the run alive
	# for two more steps before the stop predicate releases it.
	open_faults = ["fault"]

	def is_complete() -> bool:
		if env.now >= 3.0:
			open_faults.clear()
		return not open_faults

	run_simulation_loop(env, cast(Any, Dashboard()), is_complete=is_complete)

	assert display_calls == [1.0, 2.0, 3.0]
