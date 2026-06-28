import logging

import simpy

from tests.flow.helpers import BeltSegment, CoolingState, FinalStorage, KPITracker, Machine, Network, Node, link_output, prime_input, start_machines


def test_partial_downstream_capacity_slows_machine_handoff_throughput(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	storage = FinalStorage()
	downstream = Machine(
		env,
		"M2",
		[("cycle-1", 10.0), ("cycle-2", 10.0)],
		network,
		kpi_tracker,
		input_capacity=2,
	)
	link_output(env, downstream, storage, network)
	start_machines(downstream)
	assert downstream.production_process is not None
	downstream.production_process.interrupt()
	env.run(until=0.01)
	upstream = Machine(env, "M1", [("P1", 0.1)], network, kpi_tracker)
	upstream_belt = link_output(env, upstream, downstream, network)
	downstream.inbound_parts.append("Queued-P0")
	assert downstream.capacity_pressure() == 0.5
	upstream_belt.congestion_delay_per_part = 1.0
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1")
	start_machines(upstream)
	caplog.clear()

	with caplog.at_level(logging.WARNING):
		env.run(until=0.4)

	assert upstream.parts_produced == 0
	assert any(record.event_reason == "downstream_capacity_pressure" for record in caplog.records)
	assert not any(record.event_id == "machine:M1:production_slowdown" for record in caplog.records)

	env.run(until=1)

	assert upstream.parts_produced == 1


def test_slow_downstream_actuator_reports_incoming_belt_transfer_rate_degraded(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	storage = FinalStorage()
	downstream = Machine(env, "M2", [("cycle-1", 10.0)], network, kpi_tracker)
	link_output(env, downstream, storage, network)
	downstream.actuator.inject_fault("slow_response")
	upstream = Machine(env, "M1", [("P1", 0.1)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	assert upstream.outgoing_belt is not None
	upstream.outgoing_belt.congestion_delay_per_part = 1.0
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)
	caplog.clear()

	with caplog.at_level(logging.WARNING):
		env.run(until=0.3)

	throughput_records = [record for record in caplog.records if record.event_id == "belt:M1:M2:transfer_rate_degraded"]
	assert len(throughput_records) == 1
	assert throughput_records[0].cause_id == "machine:M2:production_slowdown"
	assert throughput_records[0].event_reason == "downstream_actuator_slow_response"
	assert any(record.event_id == "machine:M2:production_slowdown" and record.cause_id == "actuator:M2:slow_response" for record in caplog.records)


def test_network_latency_reports_belt_transfer_rate_degraded(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env, handoff_timeout=1.0)
	network.inject_fault("latency")
	latencies = iter([0.05, 0.5])
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: next(latencies))
	source = Node("M1")
	storage = FinalStorage()
	belt = BeltSegment(env, source, storage, network)

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.run()

	throughput_records = [record for record in caplog.records if record.event_id == "belt:M1:FinalStorage:transfer_rate_degraded"]
	assert len(throughput_records) == 1
	assert throughput_records[0].cause_id == "network:network_latency_detected"
	assert throughput_records[0].event_reason == "network_latency"


def test_hard_downstream_blockage_does_not_emit_partial_congestion(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	downstream = Machine(env, "M2", [("cycle-1", 10.0)], network, kpi_tracker, input_capacity=1)
	upstream = Machine(env, "M1", [("P1", 0.1)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	assert upstream.outgoing_belt is not None
	downstream.temperature.cooling_state = CoolingState.INTENSE
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)

	with caplog.at_level(logging.WARNING):
		env.run(until=0.3)

	assert any(record.event_id == "belt:M1:M2:handoff_blocked" for record in caplog.records)
	assert not any(record.event_id == "belt:M1:M2:transfer_rate_degraded" and record.event_reason == "downstream_capacity_pressure" for record in caplog.records)


def test_partial_downstream_output_pressure_slows_upstream_handoff(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	storage = FinalStorage()
	downstream = Machine(
		env,
		"M2",
		[("cycle-1", 10.0)],
		network,
		kpi_tracker,
		input_capacity=2,
	)
	link_output(env, downstream, storage, network, capacity=2)
	upstream = Machine(env, "M1", [("P1", 0.1)], network, kpi_tracker)
	upstream_belt = link_output(env, upstream, downstream, network)
	assert downstream.outgoing_belt is not None
	downstream.outgoing_belt.queue.append("Queued-P0")
	upstream_belt.congestion_delay_per_part = 1.0
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)

	with caplog.at_level(logging.WARNING):
		env.run(until=0.4)

	assert downstream.can_accept_part()
	assert downstream.capacity_pressure() == 0.5
	assert upstream.parts_produced == 0
	assert any(record.event_reason == "downstream_capacity_pressure" for record in caplog.records)

	env.run(until=1)

	assert upstream.parts_produced == 1


def test_saturated_downstream_output_counts_as_capacity_pressure() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	blocked_storage = Node("BlockedStorage")
	blocked_storage.can_accept_part = lambda: False
	blocked_storage.rejection_reason = lambda: "input_capacity"
	downstream = Machine(
		env,
		"M2",
		[("cycle-1", 0.1)],
		network,
		kpi_tracker,
		input_capacity=2,
	)
	link_output(env, downstream, blocked_storage, network)
	kpi_tracker.initialize_machine_states(["M2"])
	start_machines(downstream)

	assert downstream.receive_part("P1")
	env.run(until=1)

	assert downstream.outgoing_belt is not None
	assert list(downstream.outgoing_belt.queue) == ["P1"]
	assert downstream.capacity_pressure() == 1.0


def test_fifo_waiting_does_not_emit_false_handoff_blockage(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	storage = FinalStorage()
	belt = BeltSegment(env, source, storage, network, capacity=2, congestion_delay_per_part=1.0)

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.process(belt.handoff("P2"))
		env.run(until=1)

	assert storage.stored_parts == ["P1", "P2"]
	assert not any(record.event_id == "belt:M1:FinalStorage:handoff_blocked" for record in caplog.records)
