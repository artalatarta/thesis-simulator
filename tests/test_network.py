import logging

import simpy

from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import NETWORK_REPAIR_TIME, Network
from cps.core.reporting import EventReporter
from cps.config import DEFAULT_FACTORY_CONFIG
from cps.simulation.setup import fault_injector, setup_simulation
from cps.types import ProcessGenerator


def test_network_fault_lifecycle_and_observations(caplog) -> None:
	env = simpy.Environment()
	network = Network(env)

	with caplog.at_level(logging.ERROR):
		network.inject_fault("latency")

	assert network.fault_type == "latency"
	assert network.observe_fault() == "network:network_latency_detected"
	assert any(record.message == "FAULT INJECTED network:latency" for record in caplog.records)

	network.clear_fault()

	assert network.fault_type is None
	assert network.observe_fault() is None


def test_network_monitor_emits_fault_observations(caplog) -> None:
	env = simpy.Environment()
	network = Network(env)
	network.start_observation_monitor()
	network.inject_fault("packet_loss")

	with caplog.at_level(logging.WARNING):
		env.run(until=5.1)

	assert any(record.event_id == "network:network_packet_loss_detected" for record in caplog.records)


def test_network_packet_loss_fault_blocks_handoff(monkeypatch) -> None:
	env = simpy.Environment()
	network = Network(env)
	network.inject_fault("packet_loss")
	monkeypatch.setattr("cps.core.network.random.random", lambda: 0.0)

	result = None

	def run_handoff():
		nonlocal result
		result = yield env.process(network.coordinate_handoff("M1", "M2", "P1"))

	env.process(run_handoff())
	env.run()

	assert result is not None
	assert not result.success
	assert result.reason == "packet_loss"


def test_network_latency_fault_reports_delayed_success(monkeypatch) -> None:
	env = simpy.Environment()
	network = Network(env, handoff_timeout=1.0)
	network.inject_fault("latency")
	latencies = iter([0.05, 0.5])
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: next(latencies))

	result = None

	def run_handoff():
		nonlocal result
		result = yield env.process(network.coordinate_handoff("M1", "M2", "P1"))

	env.process(run_handoff())
	env.run()

	assert result is not None
	assert result.success
	assert result.delay > 0
	assert result.reason == "network_latency"


def test_network_packet_loss_retry_succeeds(monkeypatch) -> None:
	env = simpy.Environment()
	network = Network(env)
	network.inject_fault("packet_loss")
	monkeypatch.setattr("cps.core.network.random.random", lambda: 0.75)

	result = None

	def run_handoff():
		nonlocal result
		result = yield env.process(network.coordinate_handoff("M1", "M2", "P1"))

	env.process(run_handoff())
	env.run()

	assert result is not None
	assert result.success
	assert result.reason == "packet_loss_retry"


def test_network_latency_can_block_handoff(monkeypatch) -> None:
	env = simpy.Environment()
	network = Network(env, handoff_timeout=0.25)
	network.inject_fault("latency")
	latencies = iter([0.05, 0.5])
	monkeypatch.setattr("cps.core.network.random.normalvariate", lambda *_: next(latencies))

	result = None

	def run_handoff():
		nonlocal result
		result = yield env.process(network.coordinate_handoff("M1", "M2", "P1"))

	env.process(run_handoff())
	env.run()

	assert result is not None
	assert not result.success
	assert result.reason == "network_latency"
	assert result.delay == 0.55


def test_network_latency_fault_can_inject_exact_delay() -> None:
	env = simpy.Environment()
	network = Network(env, handoff_timeout=2.0, latency_base_mean=0.05, latency_base_stddev=0.0)
	network.inject_fault("latency", latency_delay=0.75)
	result = None

	def run_handoff():
		nonlocal result
		result = yield env.process(network.coordinate_handoff("M1", "M2", "P1"))

	env.process(run_handoff())
	env.run()

	assert result is not None
	assert result.success
	assert result.delay == 0.8


def test_network_packet_loss_fault_accepts_100_percent() -> None:
	env = simpy.Environment()
	network = Network(env)
	network.inject_fault("packet_loss", packet_loss_percent=100.0)
	result = None

	def run_handoff():
		nonlocal result
		result = yield env.process(network.coordinate_handoff("M1", "M2", "P1"))

	env.process(run_handoff())
	env.run()

	assert result is not None
	assert not result.success
	assert result.reason == "packet_loss"


	env = simpy.Environment()
	network = Network(env)
	network.inject_fault("packet_loss", packet_loss_percent=100.0)

	event = network.event_reporter.events[-1]
	assert event.identifier == "network:packet_loss"
	assert event.context["packet_loss_percent"] == 100.0
	assert event.context["packet_loss_probability"] == 1.0


def test_setup_registers_shared_network_as_faultable_component() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)

	all_components = setup_simulation(env, kpi_tracker, DEFAULT_FACTORY_CONFIG, EventReporter()).faultable_components

	assert any(isinstance(component, Network) for component in all_components)


def test_fault_injector_tracks_network_fault_in_kpis(monkeypatch) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	monkeypatch.setattr("cps.simulation.setup.random.expovariate", lambda _: 1.0)
	monkeypatch.setattr("cps.simulation.setup.random.choice", lambda choices: choices[0])

	env.process(fault_injector(env, [network], kpi_tracker, lambda: True))
	env.run(until=1.1)

	assert network.fault_type == "latency"
	assert "network-latency" in kpi_tracker.open_faults


def test_fault_injector_stops_before_injecting_when_lifecycle_ends(monkeypatch) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	should_inject = True
	monkeypatch.setattr("cps.simulation.setup.random.expovariate", lambda _: 1.0)
	monkeypatch.setattr("cps.simulation.setup.random.choice", lambda choices: choices[0])

	def stop_injection() -> ProcessGenerator:
		nonlocal should_inject
		yield env.timeout(0.5)
		should_inject = False

	env.process(stop_injection())
	env.process(fault_injector(env, [network], kpi_tracker, lambda: should_inject))
	env.run(until=1.1)

	assert network.fault_type is None
	assert kpi_tracker.open_faults == {}


def test_fault_injector_observes_production_finished_at_injection_time(monkeypatch) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	should_inject = True
	monkeypatch.setattr("cps.simulation.setup.random.expovariate", lambda _: 1.0)
	monkeypatch.setattr("cps.simulation.setup.random.choice", lambda choices: choices[0])

	def finish_production() -> ProcessGenerator:
		nonlocal should_inject
		yield env.timeout(1.0)
		should_inject = False

	env.process(fault_injector(env, [network], kpi_tracker, lambda: should_inject))
	env.process(finish_production())
	env.run(until=1.1)

	assert network.fault_type is None
	assert kpi_tracker.open_faults == {}


def test_network_repairs_shared_fault() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.initialize_machine_states(["M1"])
	network = Network(env)
	Machine(env, "M1", [], network, kpi_tracker)
	network.inject_fault("packet_loss")
	kpi_tracker.track_fault_start("network", "packet_loss")
	assert network.dispatch_repair(kpi_tracker)

	env.run(until=NETWORK_REPAIR_TIME + 1)

	assert network.fault_type is None
	assert "network-packet_loss" not in kpi_tracker.open_faults


def test_network_repair_dispatch_completes_immediately() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	network.inject_fault("packet_loss")

	assert network.dispatch_repair(kpi_tracker)
	assert network.fault_type is None
	assert network.pending_repairs == set()
	assert not network.dispatch_repair(kpi_tracker)


def test_network_repairs_shared_fault_without_machines() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	network = Network(env)
	network.inject_fault("packet_loss")
	kpi_tracker.track_fault_start("network", "packet_loss")
	assert network.dispatch_repair(kpi_tracker)

	env.run(until=NETWORK_REPAIR_TIME + 1)

	assert network.fault_type is None
	assert "network-packet_loss" not in kpi_tracker.open_faults
