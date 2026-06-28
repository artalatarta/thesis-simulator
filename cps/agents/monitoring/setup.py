from typing import TYPE_CHECKING

import simpy

from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from cps.agents.contracts import DetectsConflicts, ResolvesConflicts
from cps.agents.resolution import LLMClient
from cps.core.reporting import EventReporter

if TYPE_CHECKING:
	from cps.agents.monitoring.driver import MonitoringDriver


def configure_monitoring(
	env: simpy.Environment,
	machines: dict[str, Machine],
	kpi_tracker: KPITracker,
	network: Network | None,
	detector: DetectsConflicts,
	resolver: ResolvesConflicts,
	llm_client: LLMClient,
	event_reporter: EventReporter,
) -> "MonitoringDriver":
	"""Attach the monitoring driver to the simulation."""
	from cps.agents.monitoring.driver import MonitoringDriver
	from cps.agents.monitoring import monitoring_agents_for_machines

	agents = monitoring_agents_for_machines(machines.values(), network, kpi_tracker, llm_client)
	driver = MonitoringDriver(agents=agents, detector=detector, resolver=resolver, event_reporter=event_reporter)
	driver.attach(env, machines, kpi_tracker)
	return driver
