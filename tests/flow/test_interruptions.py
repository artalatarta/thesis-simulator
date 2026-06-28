from tests.flow.helpers import ActiveWorkItem, CoolingState, FinalStorage, WorkLocation, active_product, link_output, make_environment, make_machine, prime_input, start_machines


def test_downstream_interrupt_before_intake_does_not_duplicate_input_queue() -> None:
	env, kpi_tracker, network = make_environment()
	downstream = make_machine(env, "M2", [("cycle-1", 5.0)], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M2"])
	start_machines(downstream)

	assert downstream.receive_part("P1")
	env.run(until=0.05)

	assert downstream.inbound_parts == ["P1"]

	assert downstream.production_process is not None
	downstream.production_process.interrupt()
	env.run(until=0.1)

	assert downstream.inbound_parts == ["P1"]
	assert active_product(downstream, WorkLocation.WORK_IN_PROGRESS) is None
	assert downstream.production_state.production_schedule == [("cycle-1", 5.0)]


def test_downstream_interrupt_after_intake_keeps_internal_work_in_progress_for_restart() -> None:
	env, kpi_tracker, network = make_environment()
	storage = FinalStorage()
	downstream = make_machine(env, "M2", [("cycle-1", 5.0)], network, kpi_tracker)
	link_output(env, downstream, storage, network)
	kpi_tracker.initialize_machine_states(["M2"])
	start_machines(downstream)

	assert downstream.receive_part("P1")
	env.run(until=0.2)

	assert downstream.inbound_parts == []
	assert downstream.is_processing
	assert active_product(downstream, WorkLocation.WORK_IN_PROGRESS) == "P1"

	assert downstream.production_process is not None
	downstream.production_process.interrupt()
	env.run(until=0.3)

	assert downstream.inbound_parts == []
	assert active_product(downstream, WorkLocation.WORK_IN_PROGRESS) == "P1"
	assert downstream.production_state.production_schedule == [("cycle-1", 5.0)]

	assert downstream.resume_production_if_ready()
	env.run(until=6)

	assert downstream.parts_produced == 1
	assert active_product(downstream, WorkLocation.WORK_IN_PROGRESS) is None
	assert storage.stored_parts == ["P1"]


def test_stuck_actuator_feed_restores_schedule_entry() -> None:
	env, kpi_tracker, network = make_environment()
	machine = make_machine(env, "M1", [("P1", 1.0)], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.actuator.inject_fault("stuck")
	prime_input(machine, "P1")
	start_machines(machine)

	env.run(until=0.2)
	machine.actuator.clear_fault()
	env.run(until=0.3)

	assert machine.production_state.production_schedule == [("P1", 1.0)]
	assert machine.parts_produced == 0
	assert not machine.is_processing
	assert machine.production_state.active_work is None


def test_stuck_actuator_feed_restores_duplicate_schedule_entry() -> None:
	env, kpi_tracker, network = make_environment()
	machine = make_machine(env, "M1", [("P1", 1.0), ("P1", 1.0)], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M1"])
	machine.actuator.inject_fault("stuck")
	prime_input(machine, "P1")
	start_machines(machine)

	env.run(until=0.2)
	machine.actuator.clear_fault()
	env.run(until=0.3)

	assert machine.production_state.production_schedule == [("P1", 1.0), ("P1", 1.0)]
	assert machine.parts_produced == 0


def test_interrupt_during_blocked_handoff_keeps_completed_output_on_belt() -> None:
	env, kpi_tracker, network = make_environment()
	downstream = make_machine(env, "M2", [("cycle-1", 0.1)], network, kpi_tracker, input_capacity=1)
	upstream = make_machine(env, "M1", [("P1", 0.1)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	assert upstream.outgoing_belt is not None
	downstream.temperature.cooling_state = CoolingState.INTENSE
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)

	env.run(until=0.3)
	assert list(upstream.outgoing_belt.queue) == ["P1"]
	assert active_product(upstream, WorkLocation.HANDOFF) == "P1"

	assert upstream.production_process is not None
	upstream.production_process.interrupt()
	env.run(until=0.4)

	assert list(upstream.outgoing_belt.queue) == ["P1"]
	assert upstream.production_state.production_schedule == []
	assert upstream.parts_produced == 0
	assert active_product(upstream, WorkLocation.HANDOFF) == "P1"

	downstream.temperature.cooling_state = CoolingState.NONE
	upstream.resume_production_if_ready()
	env.run(until=1)

	assert list(upstream.outgoing_belt.queue) == []
	assert active_product(upstream, WorkLocation.HANDOFF) is None
	assert upstream.parts_produced == 1
	assert downstream.inbound_parts == []
	assert downstream.parts_produced == 1


def test_interrupt_after_handoff_does_not_restore_completed_output() -> None:
	env, kpi_tracker, network = make_environment()
	downstream = make_machine(env, "M2", [("cycle-1", 1.0)], network, kpi_tracker)
	upstream = make_machine(env, "M1", [], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	assert upstream.outgoing_belt is not None

	upstream.production_state.active_work = ActiveWorkItem("P1", ("P1", 1.0), WorkLocation.WORK_IN_PROGRESS)
	upstream.outgoing_belt.queue.append("P1")

	upstream.production_runner.handle_interrupt()

	assert upstream.production_state.production_schedule == []
	assert list(upstream.outgoing_belt.queue) == ["P1"]
	assert active_product(upstream, WorkLocation.HANDOFF) == "P1"
