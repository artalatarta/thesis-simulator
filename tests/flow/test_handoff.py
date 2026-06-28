import logging

import simpy

from tests.flow.helpers import BeltSegment, CoolingState, FinalStorage, KPITracker, Machine, Network, Node, WorkLocation, active_product, link_output, prime_input, start_machines

def test_belt_segment_handoffs_store_parts_in_final_storage() -> None:
	env = simpy.Environment()
	network = Network(env)
	source = Node("M1")
	storage = FinalStorage()
	belt = BeltSegment(env, source, storage, network)

	env.process(belt.handoff("P1"))
	env.run()

	assert list(belt.queue) == []
	assert storage.stored_parts == ["P1"]



def test_machine_hands_completed_output_to_final_storage() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	storage = FinalStorage()
	machine = Machine(env, "M1", [("P1", 1.0)], network, kpi_tracker)
	link_output(env, machine, storage, network)
	kpi_tracker.initialize_machine_states(["M1"])
	prime_input(machine, "P1")
	start_machines(machine)

	env.run(until=2)

	assert machine.parts_produced == 1
	assert storage.stored_parts == ["P1"]



def test_downstream_machine_processes_handed_off_parts_only() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	storage = FinalStorage()
	downstream = Machine(env, "M2", [("downstream-cycle-1", 1.0)], network, kpi_tracker)
	link_output(env, downstream, storage, network)
	kpi_tracker.initialize_machine_states(["M2"])
	start_machines(downstream)

	env.run(until=2)

	assert downstream.parts_produced == 0
	assert downstream.production_state.production_schedule == [("downstream-cycle-1", 1.0)]
	assert storage.stored_parts == []

	assert downstream.receive_part("P1")
	env.run(until=4)

	assert downstream.parts_produced == 1
	assert downstream.inbound_parts == []
	assert active_product(downstream, WorkLocation.WORK_IN_PROGRESS) is None
	assert downstream.production_state.production_schedule == []
	assert storage.stored_parts == ["P1"]



def test_downstream_machine_rejects_input_without_schedule_capacity() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	downstream = Machine(env, "M2", [], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M2"])

	assert not downstream.receive_part("P1")
	assert downstream.rejection_reason() == "input_capacity"
	assert downstream.inbound_parts == []



def test_downstream_input_stays_occupied_until_actuator_feeds_part() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	downstream = Machine(env, "M2", [("cycle-1", 1.0), ("cycle-2", 1.0)], network, kpi_tracker)
	kpi_tracker.initialize_machine_states(["M2"])
	start_machines(downstream)

	assert downstream.receive_part("P1")
	env.run(until=0.05)

	assert downstream.inbound_parts == ["P1"]
	assert not downstream.receive_part("P2")
	assert downstream.rejection_reason() == "input_capacity"

	env.run(until=0.2)

	assert downstream.inbound_parts == []
	assert active_product(downstream, WorkLocation.WORK_IN_PROGRESS) == "P1"
	assert downstream.receive_part("P2")



def test_machine_handoff_to_blocked_downstream_reports_belt_issue(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	downstream = Machine(env, "M2", [], network, kpi_tracker)
	upstream = Machine(env, "M1", [("P1", 1.0)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	downstream.temperature.cooling_state = CoolingState.INTENSE
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)

	with caplog.at_level(logging.WARNING):
		env.run(until=2)

	assert upstream.outgoing_belt is not None
	assert any(record.event_id == "belt:M1:M2:handoff_blocked" for record in caplog.records)
	assert any(record.event_id == "machine:M1:production_blocked" for record in caplog.records)
	assert upstream.parts_produced == 0
	assert not upstream.is_processing
	assert active_product(upstream, WorkLocation.HANDOFF) == "P1"
	assert list(upstream.outgoing_belt.queue) == ["P1"]
	assert downstream.inbound_parts == []



def test_stuck_downstream_actuator_blocks_incoming_handoff(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	downstream = Machine(env, "M2", [], network, kpi_tracker)
	upstream = Machine(env, "M1", [("P1", 1.0)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	downstream.actuator.inject_fault("stuck")
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)

	with caplog.at_level(logging.WARNING):
		env.run(until=2)

	assert upstream.parts_produced == 0
	assert not upstream.is_processing
	assert active_product(upstream, WorkLocation.HANDOFF) == "P1"
	assert downstream.inbound_parts == []
	assert any(record.event_id == "belt:M1:M2:handoff_blocked" for record in caplog.records)



def test_machine_input_capacity_blocks_additional_handoffs(caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	downstream = Machine(env, "M2", [("cycle-1", 1.0)], network, kpi_tracker, input_capacity=1)
	upstream = Machine(env, "M1", [("P1", 0.1), ("P2", 0.1)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1", "P2")
	start_machines(upstream)

	with caplog.at_level(logging.WARNING):
		env.run(until=3)

	assert downstream.inbound_parts == []
	assert downstream.parts_produced == 1
	assert upstream.parts_produced == 1
	assert not upstream.is_processing
	assert upstream.outgoing_belt is not None
	assert list(upstream.outgoing_belt.queue) == ["P2"]
	assert any(record.event_reason == "input_capacity" for record in caplog.records if record.event_id == "belt:M1:M2:handoff_blocked")



def test_full_belt_blocks_additional_handoff_without_overfilling() -> None:
	env = simpy.Environment()
	network = Network(env)
	source = Node("M1")
	downstream = Node("M2")
	belt = BeltSegment(env, source, downstream, network, capacity=1)
	downstream.can_accept_part = lambda: False
	downstream.rejection_reason = lambda: "input_capacity"
	results: list[bool] = []

	def run_handoff(product_id: str):
		result = yield env.process(belt.handoff(product_id))
		results.append(result)

	env.process(run_handoff("P1"))
	env.run(until=0.2)
	env.process(run_handoff("P2"))
	env.run(until=0.3)

	assert list(belt.queue) == ["P1"]
	assert results == [False]



def test_node_blockage_does_not_emit_machine_issue(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	downstream = Node("M2")
	belt = BeltSegment(env, source, downstream, network)
	downstream.can_accept_part = lambda: False
	downstream.rejection_reason = lambda: "input_capacity"

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.run(until=0.2)

	assert any(record.event_id == "belt:M1:M2:handoff_blocked" for record in caplog.records)
	assert not any(record.event_id == "machine:M1:production_blocked" for record in caplog.records)



def test_direct_terminal_machine_can_be_linked_to_final_storage() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)

	machine = Machine(env, "M1", [], network, kpi_tracker)
	link_output(env, machine, FinalStorage(), network)

	assert machine.outgoing_belt is not None
	assert isinstance(machine.outgoing_belt.to_node, FinalStorage)
