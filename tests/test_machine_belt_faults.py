import simpy

from cps.agents.monitoring.state_observers import MachineFaultSymptomObserver
from cps.config import OBSERVATION_MONITOR_INTERVAL
from cps.core.flow import BeltSegment
from cps.core.node import FinalStorage
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from cps.core.reporting import EventReporter

reporter = EventReporter()


def _machine(machine_id: str = "M1") -> tuple[simpy.Environment, KPITracker, Machine, Network]:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states([machine_id])
	network = Network(env, reporter)
	machine = Machine(env, machine_id, [("part-001", 1.0)], network, kpi_tracker)
	machine.inbound_parts.append("part-001")
	return env, kpi_tracker, machine, network


def test_machine_fault_injection_reports_and_blocks_production() -> None:
	reporter.clear()
	_, kpi_tracker, machine, _ = _machine()
	machine.start()

	machine.inject_fault("jammed_workpiece")
	kpi_tracker.track_fault_start(machine.id, "Machine")

	assert machine.fault_type == "jammed_workpiece"
	assert not machine.can_resume_production()
	assert "M1-Machine" in kpi_tracker.open_faults
	assert any(event.kind == "root_fault" and event.identifier == "machine:M1:jammed_workpiece" for event in reporter.events)
	assert any(event.kind == "derived_issue" and event.identifier == "machine:M1:production_blocked" for event in reporter.events)

	assert not machine.clear_fault("bearing_wear")
	assert machine.clear_fault("jammed_workpiece")
	assert machine.fault_type is None
	assert "M1-Machine" not in kpi_tracker.open_faults


def test_machine_slowdown_fault_increases_process_time() -> None:
	reporter.clear()
	env, _, machine, _ = _machine()
	machine.inject_fault("bearing_wear")
	machine.fault_param = 2.0
	machine.start()

	env.run(until=1.5)

	assert machine.parts_produced == 0


def test_belt_fault_injection_reports_and_blocks_handoff() -> None:
	reporter.clear()
	env, kpi_tracker, machine, network = _machine()
	storage = FinalStorage()
	belt = BeltSegment(env, machine, storage, network)

	belt.inject_fault("belt_jam")
	kpi_tracker.track_fault_start("M1->FinalStorage", "Belt")
	delivered = env.process(belt.handoff("part-001"))
	env.run(until=0.1)

	assert delivered.value is False
	assert belt.fault_type == "belt_jam"
	assert "M1->FinalStorage-Belt" in kpi_tracker.open_faults
	assert any(event.kind == "root_fault" and event.identifier == "belt:M1:FinalStorage:belt_jam" for event in reporter.events)

	assert not belt.clear_fault(kpi_tracker, "belt_slippage")
	assert belt.clear_fault(kpi_tracker, "belt_jam")
	assert belt.fault_type is None
	assert "M1->FinalStorage-Belt" not in kpi_tracker.open_faults


def test_machine_fault_observer_emits_fault_naming_detection() -> None:
	"""While a machine fault persists, the observer emits a ``*_detected`` observation
	naming the fault so the machine-health agent can cite it, alongside the
	production-flow symptom."""
	reporter.clear()
	env, _, machine, _ = _machine()
	machine.inject_fault("jammed_workpiece")

	env.process(MachineFaultSymptomObserver(machine).monitor())
	env.run(until=OBSERVATION_MONITOR_INTERVAL + 0.1)

	assert any(
		event.kind == "observation" and event.identifier == "machine:M1:jammed_workpiece_detected" for event in reporter.events
	)


def test_faulted_belt_surfaces_fault_naming_detection_in_active_diagnostics() -> None:
	"""A faulted belt surfaces a ``*_detected`` diagnostic naming the fault so the
	belt agent can cite it; a healthy belt surfaces none."""
	reporter.clear()
	env, _, machine, network = _machine()
	storage = FinalStorage()

	jammed_belt = BeltSegment(env, machine, storage, network)
	assert jammed_belt.active_diagnostic_ids() == []
	jammed_belt.inject_fault("belt_jam")
	assert jammed_belt.active_diagnostic_ids() == ["belt:M1:FinalStorage:belt_jam_detected"]

	slipping_belt = BeltSegment(env, machine, storage, network)
	slipping_belt.inject_fault("belt_slippage")
	assert slipping_belt.active_diagnostic_ids() == ["belt:M1:FinalStorage:belt_slippage_detected"]
