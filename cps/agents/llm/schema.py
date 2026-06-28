"""Helpers for building strict JSON-schema ``response_format`` payloads.

OpenRouter/OpenAI structured outputs let us constrain a model to a JSON schema
instead of describing the shape in the prompt. Strict mode requires every object
to set ``additionalProperties: false`` and list every property in ``required``.
"""

from collections.abc import Iterable, Mapping


def enum(values: Iterable[str]) -> dict[str, object]:
	"""A schema property constrained to a fixed set of string values."""
	return {"type": "string", "enum": list(values)}


def string_array() -> dict[str, object]:
	"""A schema property holding an array of strings (ids copied from the prompt)."""
	return {"type": "array", "items": {"type": "string"}}


def enum_array(values: Iterable[str]) -> dict[str, object]:
	"""A schema property holding an array whose items are drawn from a fixed set."""
	return {"type": "array", "items": enum(values)}


def strict_object(properties: Mapping[str, object]) -> dict[str, object]:
	"""A strict object schema: all listed properties required, none extra allowed."""
	return {
		"type": "object",
		"additionalProperties": False,
		"required": list(properties),
		"properties": dict(properties),
	}


def object_array(properties: Mapping[str, object]) -> dict[str, object]:
	"""An array of strict objects with the given property schema."""
	return {"type": "array", "items": strict_object(properties)}


def response_format(name: str, schema: Mapping[str, object]) -> dict[str, object]:
	"""Wrap a schema in the ``json_schema`` ``response_format`` envelope."""
	return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": dict(schema)}}
