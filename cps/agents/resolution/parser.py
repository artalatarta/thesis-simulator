"""JSON parser for model-produced conflict resolution decisions."""

from dataclasses import dataclass
from typing import TypeVar

from cps.agents.contracts import (
	CONFIDENCE_LEVELS,
	ActionLabel,
	ConfidenceLevel,
	DiagnosisLabel,
	MonitoringReport,
)
from cps.agents.llm.json import extract_json_object


LabelT = TypeVar("LabelT", DiagnosisLabel, ActionLabel, ConfidenceLevel)


def _parse_label(value: object, allowed: tuple[LabelT, ...]) -> LabelT | None:
	if isinstance(value, str) and value in allowed:
		return value
	return None


@dataclass(frozen=True)
class ParsedDecision:
	selected_diagnosis: DiagnosisLabel
	selected_action: ActionLabel
	confidence: ConfidenceLevel
	selected_diagnosis_id: str | None
	explanation: str
	selected_report_index: int


def _parse_report_index(value: object, *, conflict_reports: tuple[MonitoringReport, ...]) -> int | None:
	if not isinstance(value, int) or isinstance(value, bool):
		return None
	if value < 1 or value > len(conflict_reports):
		return None
	return value


def parse_decision(text: str, *, conflict_reports: tuple[MonitoringReport, ...] = ()) -> ParsedDecision | None:
	payload = extract_json_object(text)
	if payload is None:
		return None
	if not conflict_reports:
		return None
	selected_report_index = _parse_report_index(payload.get("selected_report_index"), conflict_reports=conflict_reports)
	confidence = _parse_label(payload.get("confidence"), CONFIDENCE_LEVELS)
	if selected_report_index is None or confidence is None:
		return None
	selected_report = conflict_reports[selected_report_index - 1]
	explanation = payload.get("explanation")
	return ParsedDecision(
		selected_diagnosis=selected_report.diagnosis,
		selected_action=selected_report.recommended_action,
		confidence=confidence,
		selected_diagnosis_id=selected_report.diagnosis_id,
		explanation=str(explanation) if explanation is not None else "",
		selected_report_index=selected_report_index,
	)
