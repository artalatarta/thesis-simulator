"""LLM client adapters used by agentic simulator components."""

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from openai import OpenAI

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Route every request to the fastest provider serving the model; the run's
# wall-clock time is dominated by these calls, not the simulation itself.
DEFAULT_OPENROUTER_PROVIDER_SORT = "throughput"
MAX_COMPLETION_TOKENS = 2048
REQUEST_TIMEOUT_SECONDS = 60.0
ParsedT = TypeVar("ParsedT")


@dataclass(frozen=True)
class LLMCompletion:
	"""A single model response plus metadata needed by traces."""

	text: str
	model: str
	prompt_tokens: int = 0
	completion_tokens: int = 0
	latency_ms: float = 0.0


@dataclass(frozen=True)
class LLMRetryResult(Generic[ParsedT]):
	"""Parsed model output plus aggregate retry metadata."""

	parsed: ParsedT | None
	model: str
	temperature: float
	attempts: int
	parse_failures: int
	latency_ms: float
	prompt_tokens: int
	completion_tokens: int

	def metadata(self) -> dict[str, object]:
		return {
			"model": self.model,
			"temperature": self.temperature,
			"attempts": self.attempts,
			"parse_failures": self.parse_failures,
			"latency_ms": self.latency_ms,
			"prompt_tokens": self.prompt_tokens,
			"completion_tokens": self.completion_tokens,
			"fell_back": self.parsed is None,
		}


class LLMRetryError(Exception):
	"""Raised when an LLM retry loop fails before producing parseable output."""

	def __init__(self, error: Exception, result: LLMRetryResult[object]) -> None:
		super().__init__(str(error))
		self.error = error
		self.result = result


class LLMClient(Protocol):
	"""The minimal surface the LLM agents and resolver depend on."""

	def complete(
		self,
		system: str,
		user: str,
		*,
		temperature: float,
		response_format: Mapping[str, object],
	) -> LLMCompletion: ...


def complete_with_retries(
	client: LLMClient,
	system: str,
	user: str,
	*,
	temperature: float,
	max_retries: int,
	retry_temperature_step: float,
	response_format: Mapping[str, object],
	parse: Callable[[str], ParsedT | None],
	on_completion: Callable[[int, float, LLMCompletion], None] | None = None,
) -> LLMRetryResult[ParsedT]:
	model = ""
	latency_ms = 0.0
	prompt_tokens = 0
	completion_tokens = 0
	attempts = 0
	attempt_temperature = temperature
	for attempt in range(max(max_retries, 0) + 1):
		attempt_temperature = temperature + attempt * retry_temperature_step
		attempts = attempt + 1
		try:
			completion = client.complete(system, user, temperature=attempt_temperature, response_format=response_format)
		except Exception as exc:
			raise LLMRetryError(
				exc,
				LLMRetryResult(
					parsed=None,
					model=model,
					temperature=attempt_temperature,
					attempts=attempts,
					parse_failures=attempt,
					latency_ms=latency_ms,
					prompt_tokens=prompt_tokens,
					completion_tokens=completion_tokens,
				),
			) from exc
		if on_completion is not None:
			on_completion(attempts, attempt_temperature, completion)
		model = completion.model
		latency_ms += completion.latency_ms
		prompt_tokens += completion.prompt_tokens
		completion_tokens += completion.completion_tokens
		parsed = parse(completion.text)
		if parsed is not None:
			return LLMRetryResult(
				parsed=parsed,
				model=model,
				temperature=attempt_temperature,
				attempts=attempts,
				parse_failures=attempt,
				latency_ms=latency_ms,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
			)
	return LLMRetryResult(
		parsed=None,
		model=model,
		temperature=attempt_temperature,
		attempts=attempts,
		parse_failures=attempts,
		latency_ms=latency_ms,
		prompt_tokens=prompt_tokens,
		completion_tokens=completion_tokens,
	)


class OpenRouterClient:
	""":class:`LLMClient` backed by the OpenAI SDK pointed at OpenRouter."""

	def __init__(
		self,
		*,
		api_key: str,
		model: str = DEFAULT_OPENROUTER_MODEL,
		base_url: str = DEFAULT_OPENROUTER_BASE_URL,
		provider_sort: str | None = DEFAULT_OPENROUTER_PROVIDER_SORT,
	) -> None:
		self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS)
		self._model = model
		# Only route to providers that honor response_format so the JSON request
		# below is actually enforced (and healed) rather than silently ignored by
		# whichever provider the throughput sort would otherwise pick.
		provider: dict[str, object] = {"require_parameters": True}
		if provider_sort:
			provider["sort"] = provider_sort
		# The response-healing plugin validates and repairs malformed JSON
		# (markdown-fenced output, trailing commas, missing brackets, prose mixed
		# with JSON) before it reaches us. It requires the per-call json_schema
		# response_format and only applies to non-streaming requests.
		self._extra_body = {"provider": provider, "plugins": [{"id": "response-healing"}]}

	def complete(
		self,
		system: str,
		user: str,
		*,
		temperature: float,
		response_format: Mapping[str, object],
	) -> LLMCompletion:
		start = time.perf_counter()
		response = self._client.chat.completions.create(
			model=self._model,
			messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
			temperature=temperature,
			max_tokens=MAX_COMPLETION_TOKENS,
			response_format=cast(Any, response_format),
			extra_body=self._extra_body,
		)
		usage = response.usage
		return LLMCompletion(
			text=response.choices[0].message.content or "",
			model=response.model or self._model,
			prompt_tokens=usage.prompt_tokens if usage else 0,
			completion_tokens=usage.completion_tokens if usage else 0,
			latency_ms=(time.perf_counter() - start) * 1000.0,
		)


def openrouter_client_from_env(*, model: str | None = None) -> OpenRouterClient:
	"""Build an :class:`OpenRouterClient` from ``OPENROUTER_*`` environment variables."""
	api_key = os.environ.get("OPENROUTER_API_KEY")
	if not api_key:
		raise RuntimeError("OPENROUTER_API_KEY is not set. Export it (e.g. `set -a; source .env`) to use the LLM agents.")
	provider_sort = os.environ.get("OPENROUTER_PROVIDER_SORT", DEFAULT_OPENROUTER_PROVIDER_SORT)
	return OpenRouterClient(
		api_key=api_key,
		model=model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
		base_url=os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
		provider_sort=provider_sort or None,
	)
