from collections.abc import Iterable, Mapping, Sequence

from cps.agents.contracts import Conflict, ConflictType, EvidenceWindow, MonitoringReport
from cps.agents.report_selection import unique_actionable_reports
from cps.agents.resolution import LLMCompletion


class MockLLMClient:
	"""Scripted LLM client for tests."""

	def __init__(
		self,
		responses: Sequence[str],
		*,
		model: str = "mock-model",
		prompt_tokens: int = 0,
		completion_tokens: int = 0,
		latency_ms: float = 0.0,
	) -> None:
		if not responses:
			raise ValueError("MockLLMClient requires at least one scripted response.")
		self._responses = list(responses)
		self._model = model
		self._prompt_tokens = prompt_tokens
		self._completion_tokens = completion_tokens
		self._latency_ms = latency_ms
		self.calls: list[tuple[str, str, float]] = []
		self.response_formats: list[Mapping[str, object] | None] = []

	def complete(
		self,
		system: str,
		user: str,
		*,
		temperature: float,
		response_format: Mapping[str, object],
	) -> LLMCompletion:
		index = min(len(self.calls), len(self._responses) - 1)
		self.calls.append((system, user, temperature))
		self.response_formats.append(response_format)
		return LLMCompletion(
			text=self._responses[index],
			model=self._model,
			prompt_tokens=self._prompt_tokens,
			completion_tokens=self._completion_tokens,
			latency_ms=self._latency_ms,
		)


class RuleBasedDetector:
	"""Test fake that preserves the legacy deterministic detector behavior."""

	def detect(self, reports: Iterable[MonitoringReport], *, window: EvidenceWindow) -> tuple[Conflict, ...]:
		groups: dict[str | None, list[MonitoringReport]] = {}
		for report in unique_actionable_reports(reports):
			key = None if (report.diagnosis_id or "").startswith("network:") else report.machine_id
			groups.setdefault(key, []).append(report)
		conflicts: list[Conflict] = []
		for index, (machine_id, group) in enumerate(groups.items(), start=1):
			if len(group) < 2:
				continue
			conflict_types = _conflict_types(group)
			if not conflict_types:
				continue
			conflicts.append(
				Conflict(
					conflict_id=f"conflict-{machine_id or 'line'}-t{window.start_time:g}-{index}",
					machine_id=machine_id,
					window=window,
					conflict_types=conflict_types,
					reports=tuple(group),
					description="Monitoring reports in the same evidence window disagree on diagnosis, action, or confidence.",
				)
			)
		return tuple(conflicts)


def _conflict_types(reports: Sequence[MonitoringReport]) -> tuple[ConflictType, ...]:
	types: list[ConflictType] = []
	if len({report.diagnosis_id for report in reports}) > 1:
		types.append("diagnosis")
	if len({report.recommended_action for report in reports}) > 1:
		types.append("action")
	if len({report.confidence for report in reports}) > 1:
		types.append("confidence")
	return tuple(types)
