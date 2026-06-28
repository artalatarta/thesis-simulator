"""Base abstractions for role-scoped monitoring agents."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import simpy

from cps.agents.contracts import AgentRole, MonitoringReport
from cps.agents.identifiers import machine_id_from_identifier
from cps.agents.monitoring.context import MonitoringContext
from cps.agents.monitoring.generators import LLMMonitoringReportGenerator
from cps.agents.report_selection import PASSIVE_ACTIONS
from cps.agents.resolution import LLMClient
from cps.core.reporting import ReportedEvent
from cps.types import ActionOutcome

ActionHandler = Callable[[MonitoringReport, bool], ActionOutcome | bool | None]


class MonitoringAgent(ABC):
	"""Base class for role-scoped LLM monitoring agents.

	Each agent prompts its injected :class:`LLMClient` with only its scoped
	evidence and parses strict JSON into :class:`MonitoringReport` objects.
	"""

	role: ClassVar[AgentRole]
	name: ClassVar[str]
	model: ClassVar[str] = "unknown-model"
	system_prompt_focus: ClassVar[str] = "Analyze only evidence in your assigned monitoring scope."
	system_prompt_action_guidance: ClassVar[str] = ""
	# Fault-catalog patterns used to derive diagnosis_id from the model's cited
	# evidence; the prompt lists them as system-owned rather than model output.
	system_prompt_diagnosis_ids: ClassVar[tuple[str, ...]] = ()
	def __init__(self, llm_client: LLMClient) -> None:
		self.llm_client = llm_client
		self.report_generator = LLMMonitoringReportGenerator(llm_client)

	@property
	def identity_name(self) -> str:
		"""Concrete agent identity used in prompts, reports, and action routing."""
		return self.name

	def execute_action(self, report: MonitoringReport, *, require_sensor_operational: bool = False) -> ActionOutcome | None:
		handler = self._action_handlers().get(report.recommended_action)
		if handler is None:
			return None
		result = handler(report, require_sensor_operational)
		if isinstance(result, bool):
			return "succeeded" if result else "failed"
		return result

	def start(self, env: simpy.Environment) -> tuple[simpy.Process, ...]:
		_ = env
		return ()

	@abstractmethod
	def owns_event(self, event: ReportedEvent) -> bool:
		"""Whether this agent is responsible for reporting ``event``."""

	def generate_reports(self, context: MonitoringContext) -> tuple[MonitoringReport, ...]:
		return self.report_generator.generate(self, context)

	def _llm_supplementary_context(self, context: MonitoringContext) -> dict[str, object]:
		_ = context
		return {}

	def _llm_should_use_supplementary_without_events(self, context: MonitoringContext) -> bool:
		_ = context
		return False

	def _report_id(self, machine_id: str | None, window_start: float, seq: int) -> str:
		return f"{self.identity_name}-{machine_id or 'line'}-t{window_start:g}-{seq}"

	def _report_machine_id(self, identifier: str) -> str | None:
		return machine_id_from_identifier(identifier)

	def _persona(self) -> str:
		return f"{self.identity_name} LLM monitoring agent ({self.role})"

	def _llm_allowed_actions(self) -> tuple[str, ...]:
		return tuple(dict.fromkeys((*self._action_handlers(), *PASSIVE_ACTIONS)))

	def _action_handlers(self) -> Mapping[str, ActionHandler]:
		return {}


def run_monitoring_agents(
	context: MonitoringContext,
	agents: Iterable[MonitoringAgent],
) -> tuple[MonitoringReport, ...]:
	"""Run every agent concurrently and isolate transient model failures by role."""
	agent_tuple = tuple(agents)
	if not agent_tuple:
		return ()
	reports: list[MonitoringReport] = []
	with ThreadPoolExecutor(max_workers=len(agent_tuple), thread_name_prefix="monitoring-agent") as executor:
		for agent_reports in executor.map(lambda agent: _generate_reports_for_agent(agent, context), agent_tuple):
			reports.extend(agent_reports)
	return tuple(reports)


def _generate_reports_for_agent(agent: MonitoringAgent, context: MonitoringContext) -> tuple[MonitoringReport, ...]:
	try:
		return agent.generate_reports(context)
	except Exception:
		logging.exception(
			"Monitoring agent %s failed; skipping its reports for window %.2f-%.2f.",
			agent.name,
			context.window.start_time,
			context.window.end_time,
			extra={"component": "MonitoringAgents"},
		)
		return ()
