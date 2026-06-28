"""The LLM-backed conflict detector."""

import logging
from collections.abc import Iterable

from cps.agents.contracts import Conflict, EvidenceWindow, MonitoringReport
from cps.agents.detection.parser import ParsedConflict, parse_detection
from cps.agents.detection.prompts import format_reports, response_format, system_prompt
from cps.agents.llm.client import LLMCompletion, LLMClient, LLMRetryError, complete_with_retries
from cps.agents.report_selection import unique_actionable_reports

__all__ = ["ConflictDetector"]

DEFAULT_DETECTOR_RETRY_TEMPERATURE_STEP = 0.2
logger = logging.getLogger(__name__)


def _machine_id_for_reports(reports: tuple[MonitoringReport, ...]) -> str | None:
	machine_ids = {report.machine_id for report in reports}
	return next(iter(machine_ids)) if len(machine_ids) == 1 else None


def _conflict_from_parsed(
	parsed: ParsedConflict,
	*,
	candidates: tuple[MonitoringReport, ...],
	window: EvidenceWindow,
	index: int,
) -> Conflict:
	ids = set(parsed.report_ids)
	reports = tuple(report for report in candidates if report.report_id in ids)
	machine_id = _machine_id_for_reports(reports)
	return Conflict(
		conflict_id=f"conflict-{machine_id or 'line'}-t{window.start_time:g}-{index}",
		machine_id=machine_id,
		window=window,
		conflict_types=parsed.conflict_types,
		reports=reports,
		description=parsed.description
		or "Monitoring reports in the same evidence window offer competing explanations or recovery actions.",
	)


class ConflictDetector:
	"""Detect report conflicts with one model call per polling cycle."""

	def __init__(
		self,
		client: LLMClient,
		*,
		temperature: float = 0.0,
		max_retries: int = 2,
		retry_temperature_step: float = DEFAULT_DETECTOR_RETRY_TEMPERATURE_STEP,
	) -> None:
		self._client = client
		self._temperature = temperature
		self._max_retries = max(max_retries, 0)
		self._retry_temperature_step = retry_temperature_step
		self.traces: list[dict[str, object]] = []

	def detect(self, reports: Iterable[MonitoringReport], *, window: EvidenceWindow) -> tuple[Conflict, ...]:
		candidates = tuple(unique_actionable_reports(reports))
		if len(candidates) < 2:
			return ()
		system = system_prompt()
		user = format_reports(candidates, window=window)
		detection_schema = response_format()
		logger.debug(
			"DETECTOR_PROMPT window=%.1f-%.1f reports=%s system=%s user=%s",
			window.start_time,
			window.end_time,
			[report.report_id for report in candidates],
			system,
			user,
			extra={"component": "MonitoringAgents"},
		)
		valid_report_ids = frozenset(report.report_id for report in candidates)
		def log_completion(attempts: int, attempt_temperature: float, completion: LLMCompletion) -> None:
			logger.debug(
				"DETECTOR_COMPLETION window=%.1f-%.1f attempt=%d temperature=%.2f model=%s prompt_tokens=%s completion_tokens=%s latency_ms=%.2f text=%s",
				window.start_time,
				window.end_time,
				attempts,
				attempt_temperature,
				completion.model,
				completion.prompt_tokens,
				completion.completion_tokens,
				completion.latency_ms,
				completion.text,
				extra={"component": "MonitoringAgents"},
			)

		try:
			retry_result = complete_with_retries(
				self._client,
				system,
				user,
				temperature=self._temperature,
				max_retries=self._max_retries,
				retry_temperature_step=self._retry_temperature_step,
				response_format=detection_schema,
				parse=lambda text: parse_detection(text, valid_report_ids=valid_report_ids),
				on_completion=log_completion,
			)
		except LLMRetryError as exc:
			metadata = exc.result.metadata()
			metadata.update(
				{
					"window_start": window.start_time,
					"window_end": window.end_time,
					"conflict_ids": [],
					"fell_back": True,
					"error": repr(exc.error),
				}
			)
			self.traces.append(metadata)
			logger.exception(
				"DETECTOR_ERROR window=%.1f-%.1f metadata=%s",
				window.start_time,
				window.end_time,
				metadata,
				extra={"component": "MonitoringAgents"},
			)
			return ()
		except Exception as exc:
			metadata = {
				"window_start": window.start_time,
				"window_end": window.end_time,
				"conflict_ids": [],
				"model": "",
				"temperature": self._temperature,
				"attempts": 0,
				"latency_ms": 0.0,
				"prompt_tokens": 0,
				"completion_tokens": 0,
				"fell_back": True,
				"error": repr(exc),
			}
			self.traces.append(metadata)
			logger.exception(
				"DETECTOR_ERROR window=%.1f-%.1f metadata=%s",
				window.start_time,
				window.end_time,
				metadata,
				extra={"component": "MonitoringAgents"},
			)
			return ()
		parsed = retry_result.parsed
		metadata: dict[str, object] = {
			"window_start": window.start_time,
			"window_end": window.end_time,
			"conflict_ids": [],
			**retry_result.metadata(),
		}
		if parsed is None:
			self.traces.append(metadata)
			logger.debug(
				"DETECTOR_FALLBACK window=%.1f-%.1f metadata=%s",
				window.start_time,
				window.end_time,
				metadata,
				extra={"component": "MonitoringAgents"},
			)
			return ()
		conflicts = tuple(
			_conflict_from_parsed(conflict, candidates=candidates, window=window, index=index)
			for index, conflict in enumerate(parsed, start=1)
		)
		metadata["conflict_ids"] = [conflict.conflict_id for conflict in conflicts]
		self.traces.append(metadata)
		logger.debug(
			"DETECTOR_CONFLICTS window=%.1f-%.1f conflict_ids=%s metadata=%s",
			window.start_time,
			window.end_time,
			metadata["conflict_ids"],
			metadata,
			extra={"component": "MonitoringAgents"},
		)
		return conflicts
