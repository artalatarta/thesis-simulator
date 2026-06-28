"""JSON parser for model-produced conflict detection results."""

from dataclasses import dataclass

from cps.agents.contracts import CONFLICT_TYPES, ConflictType
from cps.agents.llm.json import extract_json_object


@dataclass(frozen=True)
class ParsedConflict:
	report_ids: tuple[str, ...]
	conflict_types: tuple[ConflictType, ...]
	description: str


def _parse_conflict_types(values: object) -> tuple[ConflictType, ...] | None:
	if not isinstance(values, list):
		return None
	if not values:
		return ("diagnosis",)
	parsed: list[ConflictType] = []
	for value in values:
		if not isinstance(value, str) or value not in CONFLICT_TYPES:
			return None
		parsed.append(value)  # type: ignore[arg-type]
	return tuple(parsed)


def _parse_report_ids(values: object, *, valid_report_ids: frozenset[str]) -> tuple[str, ...] | None:
	if not isinstance(values, list):
		return None
	report_ids: list[str] = []
	seen: set[str] = set()
	for value in values:
		if not isinstance(value, str) or value not in valid_report_ids or value in seen:
			return None
		seen.add(value)
		report_ids.append(value)
	if len(report_ids) < 2:
		return None
	return tuple(report_ids)


def parse_detection(text: str, *, valid_report_ids: frozenset[str]) -> tuple[ParsedConflict, ...] | None:
	payload = extract_json_object(text)
	if payload is None:
		return None
	conflicts = payload.get("conflicts")
	if not isinstance(conflicts, list):
		return None
	parsed_conflicts: list[ParsedConflict] = []
	assigned_report_ids: set[str] = set()
	for item in conflicts:
		if not isinstance(item, dict):
			return None
		report_ids = _parse_report_ids(item.get("report_ids"), valid_report_ids=valid_report_ids)
		conflict_types = _parse_conflict_types(item.get("conflict_types"))
		if report_ids is None or conflict_types is None:
			return None
		if any(report_id in assigned_report_ids for report_id in report_ids):
			return None
		assigned_report_ids.update(report_ids)
		description = item.get("description")
		parsed_conflicts.append(
			ParsedConflict(
				report_ids=report_ids,
				conflict_types=conflict_types,
				description=str(description) if description is not None else "",
			)
		)
	return tuple(parsed_conflicts)
