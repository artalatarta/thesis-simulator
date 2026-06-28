"""Factory helpers for constructing monitoring agents."""

from collections.abc import Iterable

from cps.agents.monitoring.base import MonitoringAgent
from cps.agents.monitoring.components import BeltSegmentAgent, MachineHealthAgent, NetworkAgent
from cps.agents.monitoring.sensors import ActuatorSensorAgent, PowerSensorAgent, TemperatureSensorAgent
from cps.agents.resolution import LLMClient
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network


def monitoring_agents_for_machines(
	machines: Iterable[Machine],
	network: Network | None,
	kpi_tracker: KPITracker,
	llm_client: LLMClient,
) -> tuple[MonitoringAgent, ...]:
	agents: list[MonitoringAgent] = []
	machine_list = tuple(machines)
	for machine in machine_list:
		agents.extend(
			(
				PowerSensorAgent(machine.power_sensor, machine, llm_client),
				TemperatureSensorAgent(machine.temperature_sensor, machine, llm_client),
				ActuatorSensorAgent(machine.actuator_sensor, machine, llm_client),
				MachineHealthAgent(machine, llm_client),
			)
		)
		if machine.outgoing_belt is not None:
			agents.append(BeltSegmentAgent(machine.outgoing_belt, llm_client))
	if network is not None:
		agents.append(NetworkAgent(network, kpi_tracker, llm_client))
	return tuple(agents)
