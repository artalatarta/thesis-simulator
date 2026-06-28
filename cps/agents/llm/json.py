"""JSON extraction helpers for LLM responses."""

import json


def extract_json_object(text: str) -> dict[str, object] | None:
	"""Parse an LLM response as a JSON object.

	Responses are requested with a JSON ``response_format`` so providers
	return (and heal) valid JSON, so a plain parse suffices; malformed output
	yields ``None`` and the caller raises a clear error.
	"""
	try:
		parsed = json.loads(text.strip())
	except json.JSONDecodeError:
		return None
	return parsed if isinstance(parsed, dict) else None
