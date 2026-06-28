import simpy

from cps.core.flow import BeltSegment
from cps.core.node import FinalStorage
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network


def test_throughput_counts_only_final_storage_output() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	final_storage = FinalStorage()
	downstream = Machine(env, "M2", [("cycle-1", 1.0)], network, kpi_tracker)
	downstream.outgoing_belt = BeltSegment(env, downstream, final_storage, network)
	upstream = Machine(env, "M1", [("P1", 1.0)], network, kpi_tracker)
	upstream.outgoing_belt = BeltSegment(env, upstream, downstream, network)
	upstream.inbound_parts.append("P1")
	kpi_tracker.initialize_machine_states(["M1", "M2"])
	downstream.start()
	upstream.start()

	env.run(until=5)

	assert upstream.parts_produced == 1
	assert downstream.parts_produced == 1
	assert final_storage.stored_parts == ["P1"]
	assert kpi_tracker.throughput == 1


def test_generate_report_does_not_mutate_processing_time(capsys) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	kpi_tracker.track_machine_state_change("M1", True)
	env.run(until=5)

	before = dict(kpi_tracker.machine_states["M1"])
	kpi_tracker.generate_report()
	kpi_tracker.generate_report()
	after = dict(kpi_tracker.machine_states["M1"])

	assert after == before
	capsys.readouterr()


def test_generate_report_includes_monitoring_conflict_metrics(capsys) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.track_conflict_detected()
	kpi_tracker.track_resolver_attempt()
	kpi_tracker.track_resolver_success()
	kpi_tracker.track_resolver_failure()
	kpi_tracker.track_agent_action("succeeded")
	kpi_tracker.track_agent_action("failed")
	kpi_tracker.track_agent_action("already_resolved")

	kpi_tracker.generate_report()

	output = capsys.readouterr().out
	assert "Conflicts Detected: 1" in output
	assert "Resolver Attempts: 1" in output
	assert "Resolver Successes: 1" in output
	assert "Resolver Failures: 1" in output
	assert "Agent Actions Attempted: 3" in output
	assert "Agent Actions Succeeded: 1" in output
	assert "Agent Actions Already Resolved: 1" in output
	assert "Agent Actions Failed: 1" in output


def test_track_agent_action_counts_already_resolved_separately() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.track_agent_action("succeeded")
	kpi_tracker.track_agent_action("already_resolved")
	kpi_tracker.track_agent_action("failed")

	assert kpi_tracker.agent_actions_attempted == 3
	assert kpi_tracker.agent_actions_succeeded == 1
	assert kpi_tracker.agent_actions_already_resolved == 1
	assert kpi_tracker.agent_actions_failed == 1
