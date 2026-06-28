"""The LLM-backed conflict resolver.

:class:`ConflictResolver` turns a detected
:class:`~cps.agents.contracts.Conflict` between LLM monitoring-agent reports
into a single :class:`~cps.agents.contracts.ResolutionDecision` with one model
call per conflict (plus bounded retries).

The resolver reaches a model only through the small :class:`LLMClient`
protocol. The production client (:class:`OpenRouterClient`) wraps the OpenAI
SDK pointed at OpenRouter.

Every decision records a trace -- model name, temperature, attempts, parse
failures, latency, and token counts -- into
:attr:`ResolutionDecision.metadata`, so experiment output is auditable and
unit tests never depend on live API calls.
"""

import logging

from cps.agents.contracts import (
	Conflict,
	ResolutionDecision,
)
from cps.agents.llm.client import (
	DEFAULT_OPENROUTER_BASE_URL,
	DEFAULT_OPENROUTER_MODEL,
	LLMClient,
	LLMCompletion,
	OpenRouterClient,
	complete_with_retries,
	openrouter_client_from_env,
)
from cps.agents.resolution.parser import ParsedDecision, parse_decision
from cps.agents.resolution.prompts import format_conflict, response_format, system_prompt

__all__ = [
	"ConflictResolver",
	"DEFAULT_OPENROUTER_BASE_URL",
	"DEFAULT_OPENROUTER_MODEL",
	"LLMClient",
	"LLMCompletion",
	"OpenRouterClient",
	"openrouter_client_from_env",
]

DEFAULT_RESOLVER_RETRY_TEMPERATURE_STEP = 0.2
logger = logging.getLogger(__name__)


def _decision_from_parsed(
	conflict: Conflict,
	parsed: ParsedDecision,
	metadata: dict[str, object],
) -> ResolutionDecision:
	metadata = {**metadata, "selected_report_index": parsed.selected_report_index}
	return ResolutionDecision(
		decision_id=f"resolution-{conflict.conflict_id}",
		conflict_id=conflict.conflict_id,
		selected_diagnosis=parsed.selected_diagnosis,
		selected_action=parsed.selected_action,
		confidence=parsed.confidence,
		supporting_report_ids=tuple(report.report_id for report in conflict.reports),
		explanation=parsed.explanation,
		metadata=metadata,
		selected_diagnosis_id=parsed.selected_diagnosis_id,
	)


def _monitoring_fallback_decision(
	conflict: Conflict,
	*,
	metadata: dict[str, object],
	explanation: str,
) -> ResolutionDecision:
	"""Safe fallback when the model cannot produce a valid resolver decision."""
	report = conflict.reports[0]
	return ResolutionDecision(
		decision_id=f"resolution-{conflict.conflict_id}",
		conflict_id=conflict.conflict_id,
		selected_diagnosis=report.diagnosis,
		selected_action="wait_for_more_evidence",
		confidence="low",
		supporting_report_ids=tuple(report.report_id for report in conflict.reports),
		explanation=explanation,
		metadata=metadata,
		selected_diagnosis_id=report.diagnosis_id,
	)


class ConflictResolver:
	"""Resolve each conflict with a single model call (with bounded retries)."""

	def __init__(
		self,
		client: LLMClient,
		*,
		temperature: float = 0.0,
		max_retries: int = 2,
		retry_temperature_step: float = DEFAULT_RESOLVER_RETRY_TEMPERATURE_STEP,
	) -> None:
		self._client = client
		self._temperature = temperature
		self._max_retries = max(max_retries, 0)
		self._retry_temperature_step = retry_temperature_step

	def resolve(self, conflict: Conflict) -> ResolutionDecision:
		system = system_prompt()
		user = format_conflict(conflict)
		decision_schema = response_format()
		logger.debug(
			"RESOLVER_PROMPT conflict_id=%s reports=%s system=%s user=%s",
			conflict.conflict_id,
			[report.report_id for report in conflict.reports],
			system,
			user,
			extra={"component": "MonitoringAgents"},
		)

		# Retry attempts follow the configured temperature schedule; with the
		# defaults the resolver tries 0.0, 0.2, and 0.4 before falling back.
		def log_completion(attempts: int, attempt_temperature: float, completion: LLMCompletion) -> None:
			logger.debug(
				"RESOLVER_COMPLETION conflict_id=%s attempt=%d temperature=%.2f model=%s prompt_tokens=%s completion_tokens=%s latency_ms=%.2f text=%s",
				conflict.conflict_id,
				attempts,
				attempt_temperature,
				completion.model,
				completion.prompt_tokens,
				completion.completion_tokens,
				completion.latency_ms,
				completion.text,
				extra={"component": "MonitoringAgents"},
			)

		retry_result = complete_with_retries(
			self._client,
			system,
			user,
			temperature=self._temperature,
			max_retries=self._max_retries,
			retry_temperature_step=self._retry_temperature_step,
			response_format=decision_schema,
			parse=lambda text: parse_decision(text, conflict_reports=conflict.reports),
			on_completion=log_completion,
		)
		parsed = retry_result.parsed
		metadata = retry_result.metadata()
		if parsed is None:
			logger.debug(
				"RESOLVER_FALLBACK conflict_id=%s metadata=%s",
				conflict.conflict_id,
				metadata,
				extra={"component": "MonitoringAgents"},
			)
			return _monitoring_fallback_decision(
				conflict,
				metadata=metadata,
				explanation="Model output could not be parsed after retries; no resolver action was executed.",
			)
		decision = _decision_from_parsed(conflict, parsed, metadata)
		logger.debug(
			"RESOLVER_DECISION conflict_id=%s selected_diagnosis_id=%s selected_action=%s selected_diagnosis=%s metadata=%s explanation=%s",
			conflict.conflict_id,
			decision.selected_diagnosis_id,
			decision.selected_action,
			decision.selected_diagnosis,
			metadata,
			decision.explanation,
			extra={"component": "MonitoringAgents"},
		)
		return decision
