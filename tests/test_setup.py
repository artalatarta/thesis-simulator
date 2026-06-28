import simpy

from cps.agents.monitoring import monitoring_agents_for_machines
from cps.agents.monitoring.components import BeltSegmentAgent
from cps.core.node import FinalStorage, RawMaterialSource
from cps.core.kpi import KPITracker
from cps.core.reporting import EventReporter
from cps.simulation.factory_line_config import FactoryLineConfig
from cps.simulation.setup import fault_injector, setup_simulation
from tests.fakes import MockLLMClient

reporter = EventReporter()


def test_setup_builds_ordered_machine_line() -> None:
	env = simpy.Environment()
	config = FactoryLineConfig(
		product="part",
		quantity=1,
		source_id="Input",
		storage_id="Output",
		stations=(
			("M1", 1.0),
			("M2", 2.0),
		),
	)

	simulation = setup_simulation(env, KPITracker(env), config, reporter)
	machines = simulation.machines

	first = machines["M1"]
	second = machines["M2"]
	# The source feeds the first machine directly, so it has no infeed belt.
	assert first.incoming_belt is None
	assert isinstance(simulation.raw_material_source, RawMaterialSource)
	assert simulation.raw_material_source.id == "Input"
	assert second.outgoing_belt is not None
	assert first.outgoing_belt is second.incoming_belt
	assert isinstance(second.outgoing_belt.to_node, FinalStorage)
	assert second.outgoing_belt.to_node is simulation.final_storage
	assert second.outgoing_belt.to_node.id == "Output"
	assert simulation.belt_segments == [first.outgoing_belt, second.outgoing_belt]


def test_setup_wires_one_explicit_reporter_through_the_factory_line() -> None:
	env = simpy.Environment()
	event_reporter = EventReporter()
	config = FactoryLineConfig(product="part", quantity=1, stations=(("M1", 1.0),))

	simulation = setup_simulation(env, KPITracker(env), config, event_reporter)
	machine = simulation.machines["M1"]
	belt = simulation.belt_segments[0]

	assert simulation.event_reporter is event_reporter
	assert simulation.network.event_reporter is event_reporter
	assert machine.event_reporter is event_reporter
	assert machine.power_sensor.event_reporter is event_reporter
	assert machine.temperature_sensor.event_reporter is event_reporter
	assert machine.actuator.event_reporter is event_reporter
	assert machine.actuator_sensor.event_reporter is event_reporter
	assert belt.event_reporter is event_reporter


def test_setup_generates_product_schedules_from_line_config() -> None:
	env = simpy.Environment()
	config = FactoryLineConfig(
		product="part",
		quantity=2,
		stations=(("Cut Station", 1.5),),
	)

	simulation = setup_simulation(env, KPITracker(env), config, reporter)
	machines = simulation.machines

	assert machines["Cut Station"].production_state.production_schedule == [("part-001", 1.5), ("part-002", 1.5)]
	assert machines["Cut Station"].inbound_parts == []
	# The source feeds the first machine directly, with no infeed belt.
	assert machines["Cut Station"].incoming_belt is None

	env.run(until=4.0)

	assert simulation.final_storage.stored_parts == ["part-001", "part-002"]


def test_setup_accepts_empty_config() -> None:
	env = simpy.Environment()
	config = FactoryLineConfig(product="part", quantity=1, stations=())

	simulation = setup_simulation(env, KPITracker(env), config, reporter)

	assert simulation.machines == {}
	assert len(simulation.faultable_components) == 1
	assert simulation.belt_segments == []


def test_setup_registers_each_machines_faultable_components() -> None:
	env = simpy.Environment()
	config = FactoryLineConfig(product="part", quantity=1, stations=(("M1", 1.0),))

	simulation = setup_simulation(env, KPITracker(env), config, reporter)

	machine_components = list(simulation.machines["M1"].faultable_components())
	assert all(component in simulation.faultable_components for component in machine_components)
	assert simulation.machines["M1"] in simulation.faultable_components


def test_setup_registers_machine_belts_as_faultable_and_monitored() -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	config = FactoryLineConfig(product="part", quantity=1, stations=(("M1", 1.0), ("M2", 1.0)))

	simulation = setup_simulation(env, kpi_tracker, config, reporter)

	# The first machine is fed directly by the source, so it has no infeed belt;
	# every belt in the line is therefore some machine's outgoing belt.
	assert simulation.machines["M1"].incoming_belt is None
	monitored_belts = [
		simulation.machines["M1"].outgoing_belt,
		simulation.machines["M2"].outgoing_belt,
	]
	assert all(belt in simulation.faultable_components for belt in monitored_belts)

	agents = monitoring_agents_for_machines(simulation.machines.values(), simulation.network, kpi_tracker, MockLLMClient(['{"reports": []}']))
	belt_agents = [agent for agent in agents if isinstance(agent, BeltSegmentAgent)]
	agent_belts = [agent.belt for agent in belt_agents]
	assert len(agent_belts) == len(monitored_belts)
	assert all(belt in agent_belts for belt in monitored_belts)


def test_setup_clears_previous_reporter_events() -> None:
	reporter.observation("sensor:old:Power:low_battery_detected", component="old")
	env = simpy.Environment()
	config = FactoryLineConfig(product="part", quantity=1, stations=(("M1", 1.0),))

	setup_simulation(env, KPITracker(env), config, reporter)

	assert all(event.identifier != "sensor:old:Power:low_battery_detected" for event in reporter.events)


def test_fault_injector_does_not_cap_active_faults(monkeypatch) -> None:
	env = simpy.Environment()
	kpi_tracker = KPITracker(env)
	kpi_tracker.open_faults["existing-1"] = 0.0
	kpi_tracker.open_faults["existing-2"] = 0.0
	injected = []

	class Component:
		fault_type = None

		def inject_random_fault(self) -> tuple[str, str]:
			injected.append(True)
			return "M1", "Power"

	monkeypatch.setattr("cps.simulation.setup.random.expovariate", lambda _rate: 1.0)
	env.process(fault_injector(env, [Component()], kpi_tracker, lambda: env.now < 2.0))

	env.run(until=2.1)

	assert injected == [True]
