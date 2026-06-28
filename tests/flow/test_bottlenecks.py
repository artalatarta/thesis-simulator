import logging

import simpy

from tests.flow.helpers import BeltSegment, CoolingState, FinalStorage, KPITracker, Machine, Network, Node, link_output, prime_input, start_machines


def test_persistent_partial_congestion_emits_bottleneck_and_machine_slowdown(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	storage = FinalStorage()
	downstream = Machine(
		env,
		"M2",
		[("cycle-1", 10.0), ("cycle-2", 10.0), ("cycle-3", 10.0)],
		network,
		kpi_tracker,
		input_capacity=2,
	)
	link_output(env, downstream, storage, network)
	upstream = Machine(env, "M1", [("P1", 0.1), ("P2", 0.1)], network, kpi_tracker)
	upstream_belt = link_output(env, upstream, downstream, network)
	downstream.inbound_parts.append("Queued-P0")
	upstream_belt.congestion_delay_per_part = 3.0
	upstream_belt.bottleneck_detection_delay = 0.5
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1", "P2")
	start_machines(downstream, upstream)
	caplog.clear()

	with caplog.at_level(logging.WARNING):
		env.run(until=3)

	assert any(record.event_id == "belt:M1:M2:persistent_queue_pressure" for record in caplog.records)
	assert any(record.event_id == "machine:M1:production_slowdown" for record in caplog.records)


def test_long_single_congestion_delay_emits_persistent_diagnosis(monkeypatch, caplog) -> None:
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
	upstream_belt.congestion_delay_per_part = 3.0
	upstream_belt.bottleneck_detection_delay = 0.5
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1")
	start_machines(upstream)
	caplog.clear()

	with caplog.at_level(logging.WARNING):
		env.run(until=2)

	assert any(record.event_id == "belt:M1:M2:persistent_queue_pressure" for record in caplog.records)
	assert any(record.event_id == "machine:M1:production_slowdown" for record in caplog.records)


def test_recovered_congestion_does_not_make_later_transient_look_persistent(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	storage = FinalStorage()
	belt = BeltSegment(env, source, storage, network, capacity=2, congestion_delay_per_part=0.2, bottleneck_detection_delay=1.0)

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.process(belt.handoff("P2"))
		env.run(until=2)
		env.process(belt.handoff("P3"))
		env.process(belt.handoff("P4"))
		env.run(until=2.3)

	assert storage.stored_parts == ["P1", "P2", "P3", "P4"]
	assert not any(record.event_id == "belt:M1:FinalStorage:persistent_queue_pressure" for record in caplog.records)


def test_failed_congested_handoff_clears_persistence_timer(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	network.inject_fault("packet_loss")
	random_values = iter([0.0, 0.75, 0.75, 0.75])
	monkeypatch.setattr("cps.core.network.random.random", lambda: next(random_values))
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	storage = FinalStorage()
	belt = BeltSegment(env, source, storage, network, capacity=2, congestion_delay_per_part=0.2, bottleneck_detection_delay=1.0)

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.process(belt.handoff("P2"))
		env.run(until=2)
		env.process(belt.handoff("P3"))
		env.process(belt.handoff("P4"))
		env.run(until=2.3)

	assert storage.stored_parts == ["P2", "P3", "P4"]
	assert not any(record.event_id == "belt:M1:FinalStorage:persistent_queue_pressure" for record in caplog.records)


def test_repeated_network_handoff_failures_emit_bottleneck(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	network.inject_fault("packet_loss")
	monkeypatch.setattr("cps.core.network.random.random", lambda: 0.0)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	storage = FinalStorage()
	belt = BeltSegment(
		env,
		source,
		storage,
		network,
		repeated_handoff_failure_threshold=3,
		bottleneck_detection_delay=10.0,
	)

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.run(until=0.1)
		env.process(belt.handoff("P2"))
		env.run(until=0.2)
		env.process(belt.handoff("P3"))
		env.run(until=0.3)

	bottleneck_records = [record for record in caplog.records if record.event_id == "belt:M1:FinalStorage:persistent_queue_pressure"]
	assert len(bottleneck_records) == 1
	assert bottleneck_records[0].event_symptom == "belt:M1:FinalStorage:handoff_blocked"
	assert bottleneck_records[0].event_symptom_occurrences == 3


def test_bottleneck_can_be_reported_again_after_recovery(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	downstream = Node("M2")
	accepting = False

	def can_accept_part() -> bool:
		return accepting

	downstream.can_accept_part = can_accept_part
	downstream.rejection_reason = lambda: "input_capacity"
	belt = BeltSegment(env, source, downstream, network, blocked_retry_interval=0.5, bottleneck_detection_delay=1.0)

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.run(until=1.6)
		accepting = True
		env.run(until=2.1)
		accepting = False
		env.process(belt.handoff("P2"))
		env.run(until=3.7)

	bottleneck_records = [record for record in caplog.records if record.event_id == "belt:M1:M2:persistent_queue_pressure"]
	assert len(bottleneck_records) == 2


def test_belt_active_diagnostics_clear_after_recovery(monkeypatch) -> None:
	env = simpy.Environment()
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	downstream = Node("M2")
	accepting = False

	def can_accept_part() -> bool:
		return accepting

	downstream.can_accept_part = can_accept_part
	downstream.rejection_reason = lambda: "input_capacity"
	belt = BeltSegment(env, source, downstream, network, blocked_retry_interval=0.5, bottleneck_detection_delay=1.0)

	env.process(belt.handoff("P1"))
	env.run(until=1.6)
	assert "belt:M1:M2:persistent_queue_pressure" in belt.active_diagnostic_ids()

	accepting = True
	env.run(until=2.1)

	assert belt.active_diagnostic_ids() == []


def test_persistent_downstream_blockage_emits_bottleneck_diagnosis(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	downstream = Node("M2")
	belt = BeltSegment(env, source, downstream, network, blocked_retry_interval=1.0, bottleneck_detection_delay=2.0)
	downstream.can_accept_part = lambda: False
	downstream.rejection_reason = lambda: "input_capacity"

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.run(until=3.1)

	bottleneck_records = [record for record in caplog.records if record.event_id == "belt:M1:M2:persistent_queue_pressure"]
	assert len(bottleneck_records) == 1
	assert bottleneck_records[0].cause_id == "belt:M1:M2:handoff_blocked"


def test_final_storage_capacity_diagnostic_points_to_belt_not_storage(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	source = Node("M1")
	storage = FinalStorage()
	belt = BeltSegment(env, source, storage, network, capacity=0)

	with caplog.at_level(logging.WARNING):
		env.process(belt.handoff("P1"))
		env.run()

	block_records = [record for record in caplog.records if record.event_id == "belt:M1:FinalStorage:handoff_blocked"]
	assert len(block_records) == 1
	assert block_records[0].event_capacity_cause == "belt_capacity"
	assert not hasattr(block_records[0], "cause_id")
	assert not any(record.event_id == "machine:FinalStorage:production_blocked" for record in caplog.records)


def test_downstream_machine_blockage_bottleneck_traces_to_machine_issue(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	downstream = Machine(env, "M2", [("cycle-1", 10.0)], network, kpi_tracker, input_capacity=1)
	upstream = Machine(env, "M1", [("P1", 0.1)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	assert upstream.outgoing_belt is not None
	upstream.outgoing_belt.blocked_retry_interval = 1.0
	upstream.outgoing_belt.bottleneck_detection_delay = 2.0
	downstream.temperature.cooling_state = CoolingState.INTENSE
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)

	with caplog.at_level(logging.WARNING):
		env.run(until=3.5)

	bottleneck_records = [record for record in caplog.records if record.event_id == "belt:M1:M2:persistent_queue_pressure"]
	assert len(bottleneck_records) == 1
	assert bottleneck_records[0].cause_id == "machine:M2:production_blocked"


def test_downstream_stuck_actuator_blockage_emits_machine_issue(monkeypatch, caplog) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: 0.0)
	downstream = Machine(env, "M2", [("cycle-1", 10.0)], network, kpi_tracker, input_capacity=1)
	upstream = Machine(env, "M1", [("P1", 0.1)], network, kpi_tracker)
	link_output(env, upstream, downstream, network)
	assert upstream.outgoing_belt is not None
	downstream.actuator.inject_fault("stuck")
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	prime_input(upstream, "P1")
	start_machines(downstream, upstream)
	caplog.clear()

	with caplog.at_level(logging.WARNING):
		env.run(until=0.3)

	assert any(record.event_id == "machine:M2:production_blocked" and record.cause_id == "actuator:M2:stuck" for record in caplog.records)
	assert any(record.event_id == "belt:M1:M2:handoff_blocked" and record.cause_id == "machine:M2:production_blocked" for record in caplog.records)
