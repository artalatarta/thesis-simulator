from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from cps.agents.contracts import Conflict, ResolutionDecision


def dataclass_to_record(instance: object, *, list_fields: frozenset[str] = frozenset()) -> dict[str, object]:
	if not is_dataclass(instance):
		raise TypeError("instance must be a dataclass.")
	record: dict[str, object] = {}
	for field_info in fields(instance):
		value = getattr(instance, field_info.name)
		if isinstance(value, Mapping):
			value = dict(value)
		elif field_info.name in list_fields:
			value = list(value)
		record[field_info.name] = value
	return record


def resolution_decision_to_record(decision: ResolutionDecision) -> dict[str, object]:
	return dataclass_to_record(decision, list_fields=frozenset({"supporting_report_ids"}))


def conflict_to_record(conflict: Conflict) -> dict[str, object]:
	return {
		"conflict_id": conflict.conflict_id,
		"machine_id": conflict.machine_id,
		"window": {
			"start": conflict.window.start_time,
			"end": conflict.window.end_time,
		},
		"conflict_types": list(conflict.conflict_types),
		"report_ids": [report.report_id for report in conflict.reports],
		"diagnoses": [report.diagnosis_id for report in conflict.reports],
		"actions": [report.recommended_action for report in conflict.reports],
		"evidence": [report.evidence[0] if report.evidence else report.report_id for report in conflict.reports],
		"description": conflict.description,
	}
