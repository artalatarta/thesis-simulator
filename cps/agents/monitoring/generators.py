"""LLM-backed report generator used by role-scoped monitoring agents."""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeVar

from cps.agents.contracts import (
	ACTION_LABELS,
	COMPONENT_LABELS,
	CONFIDENCE_LEVELS,
	DIAGNOSIS_LABELS,
	ActionLabel,
	ConfidenceLevel,
	DiagnosisLabel,
	MonitoringReport,
)
from cps.agents.diagnosis import classify_identifier
from cps.agents.identifiers import machine_id_from_identifier
from cps.agents.llm import schema
from cps.agents.llm.json import extract_json_object
from cps.agents.monitoring.debug_log import debug, report_summary
from cps.agents.report_selection import PASSIVE_ACTIONS
from cps.agents.resolution import LLMClient
from cps.core.reporting import ReportedEvent

if TYPE_CHECKING:
	from cps.agents.monitoring.base import MonitoringAgent
	from cps.agents.monitoring.context import MonitoringContext

LabelT = TypeVar("LabelT", DiagnosisLabel, ActionLabel, ConfidenceLevel)
DEFAULT_MONITORING_MAX_RETRIES = 2
DEFAULT_MONITORING_RETRY_TEMPERATURE_STEP = 0.2


def _label_or_default(value: object, default: LabelT, allowed: tuple[LabelT, ...]) -> LabelT:
	return value if isinstance(value, str) and value in allowed else default


def _action_or_default_for_diagnosis_id(action: ActionLabel, diagnosis_id: str | None, default: ActionLabel) -> ActionLabel:
	if diagnosis_id is None:
		return action
	expected_action = classify_identifier(diagnosis_id).recommended_action
	if action == expected_action:
		return action
	if action in PASSIVE_ACTIONS and expected_action in PASSIVE_ACTIONS:
		return action
	return default


def _evidence_tuple(value: object) -> tuple[str, ...]:
	if isinstance(value, list | tuple):
		return tuple(identifier for item in value if (identifier := _evidence_identifier(item)) is not None)
	identifier = _evidence_identifier(value)
	return (identifier,) if identifier is not None else ()


def _evidence_identifier(value: object) -> str | None:
	if isinstance(value, str):
		return value
	if isinstance(value, Mapping):
		identifier = value.get("identifier")
		return identifier if isinstance(identifier, str) else None
	return None


def _report_time(value: object, default: object) -> float:
	candidate = value if isinstance(value, int | float) else default
	return float(candidate) if isinstance(candidate, int | float) else 0.0


def _response_format(allowed_actions: tuple[str, ...]) -> dict[str, object]:
	"""Strict json_schema for this agent's reports array.

	``recommended_action`` is scoped to the actions the agent may take, so the
	schema enforces per-agent action scoping that the prompt used to describe.
	"""
	return schema.response_format(
		"monitoring_reports",
		schema.strict_object(
			{
				"reports": schema.object_array(
					{
						"diagnosis": schema.enum(DIAGNOSIS_LABELS),
						"component": schema.enum(COMPONENT_LABELS),
						"recommended_action": schema.enum(allowed_actions),
						"confidence": schema.enum(CONFIDENCE_LEVELS),
						"evidence": schema.string_array(),
						"rationale": {"type": "string"},
					}
				),
			}
		),
	)


def _system_prompt(
	agent_name: str,
	focus: str,
	action_guidance: str = "",
	diagnosis_id_templates: tuple[str, ...] = (),
) -> str:
	action_guidance_text = f"{action_guidance} " if action_guidance else ""
	diagnosis_scope = f" The system will derive internal catalog ids from these evidence patterns: {'; '.join(diagnosis_id_templates)}. " if diagnosis_id_templates else ""
	return (
		f"You are {agent_name}, a role-scoped monitoring agent for a cyber-physical manufacturing line. "
		f"{focus} "
		"Use only the evidence provided in this prompt. Do not infer hidden root faults. "
		'Every evidence item must be an exact "identifier" string copied verbatim from the provided events or '
		"active diagnostics; never modify or invent identifiers. "
		'Values in "supplementary_context" other than "active_diagnostics" entries are background context, not evidence. '
		'If "decision_history" is present, use it only for continuity with your own prior diagnoses, actions, and action outcomes; '
		"past evidence in that history cannot justify a new report without current-window evidence. "
		'Your reply is constrained to a JSON schema: a "reports" array whose items hold diagnosis, '
		"component, recommended_action, confidence, evidence, and rationale. "
		"diagnosis must be a concrete observed condition such as no_signal, stuck, slow_response, low_battery, overheating, jammed_workpiece, or belt_jam. "
		"component must name the affected component class such as PowerSensor, ActuatorSensor, Battery, Actuator, Machine, or BeltSegment. "
		"Do not output fault-catalog ids; the system derives them deterministically from current-window evidence. "
		'Prefer a supported corrective action when the evidence identifies one. Treat "wait_for_more_evidence" '
		"as a last-resort choice: use it only when the scoped evidence does not support a corrective action. "
		f"{action_guidance_text}"
		f"{diagnosis_scope}"
	)


def _user_prompt(agent_name: str, role: str, context: "MonitoringContext", events: tuple[ReportedEvent, ...], supplementary_context: Mapping[str, object]) -> str:
	event_payload = [
		{
			"identifier": event.identifier,
			"kind": event.kind,
			"component": event.component,
			"context": dict(event.context),
		}
		for event in events
	]
	payload = {
		"agent": agent_name,
		"role": role,
		"window": {"start": context.window.start_time, "end": context.window.end_time},
		"events": event_payload,
		"supplementary_context": dict(supplementary_context),
	}
	return json.dumps(payload, sort_keys=True)


class LLMMonitoringReportGenerator:
	"""Prompt the agent's scoped LLM and parse strict JSON into reports.

	Parse and model failures propagate to the caller rather than silently
	degrading to a deterministic stub.
	"""

	def __init__(
		self,
		llm_client: LLMClient,
		*,
		max_retries: int = DEFAULT_MONITORING_MAX_RETRIES,
		retry_temperature_step: float = DEFAULT_MONITORING_RETRY_TEMPERATURE_STEP,
	) -> None:
		self.llm_client = llm_client
		self.max_retries = max(max_retries, 0)
		self.retry_temperature_step = retry_temperature_step

	def generate(self, agent: "MonitoringAgent", context: "MonitoringContext") -> tuple[MonitoringReport, ...]:
		scoped_events = tuple(event for event in context.observable_events() if agent.owns_event(event))
		if not scoped_events and not agent._llm_should_use_supplementary_without_events(context):
			return ()
		supplementary_context = agent._llm_supplementary_context(context)
		decision_history = context.agent_decision_history.get(agent.identity_name, ())
		if decision_history:
			supplementary_context = supplementary_context | {"decision_history": [dict(entry) for entry in decision_history]}
		if not scoped_events and not supplementary_context:
			return ()
		agent_identity = agent.identity_name
		system = _system_prompt(
			agent_identity,
			agent.system_prompt_focus,
			agent.system_prompt_action_guidance,
			agent.system_prompt_diagnosis_ids,
		)
		user = _user_prompt(agent_identity, agent.role, context, scoped_events, supplementary_context)
		report_schema = _response_format(agent._llm_allowed_actions())
		debug(
			"MONITORING_PROMPT",
			agent=agent_identity,
			role=agent.role,
			window=f"{context.window.start_time:.2f}-{context.window.end_time:.2f}",
			scoped_events=[event.identifier for event in scoped_events],
			supplementary_context=supplementary_context,
			system=system,
			user=user,
		)
		for attempt in range(self.max_retries + 1):
			temperature = attempt * self.retry_temperature_step
			completion = self.llm_client.complete(system, user, temperature=temperature, response_format=report_schema)
			model = completion.model or agent.model
			debug(
				"MONITORING_COMPLETION",
				agent=agent_identity,
				attempt=attempt + 1,
				temperature=temperature,
				model=model,
				prompt_tokens=completion.prompt_tokens,
				completion_tokens=completion.completion_tokens,
				latency_ms=completion.latency_ms,
				text=completion.text,
			)
			try:
				return self._parse_llm_reports(agent, completion.text, scoped_events, supplementary_context, context, model)
			except ValueError:
				debug("MONITORING_PARSE_FAILURE", exc_info=True, agent=agent_identity, attempt=attempt + 1, text=completion.text)
				if attempt == self.max_retries:
					raise
		raise AssertionError("monitoring retry loop exited unexpectedly")

	def _parse_llm_reports(
		self,
		agent: "MonitoringAgent",
		text: str,
		events: tuple[ReportedEvent, ...],
		supplementary_context: Mapping[str, object],
		context: "MonitoringContext",
		model: str,
	) -> tuple[MonitoringReport, ...]:
		payload = extract_json_object(text)
		if payload is None:
			raise ValueError(f"{agent.name} monitoring LLM returned non-JSON output.")
		raw_reports = payload.get("reports")
		if raw_reports is None and "diagnosis" in payload:
			raw_reports = [payload]
		if not isinstance(raw_reports, list):
			raise ValueError(f"{agent.name} monitoring LLM output must contain a reports array.")
		critical_ids = context.critical_overheating_ids()
		allowed_evidence = {event.identifier for event in events}
		active_diagnostics = supplementary_context.get("active_diagnostics")
		if isinstance(active_diagnostics, list | tuple):
			allowed_evidence.update(identifier for identifier in active_diagnostics if isinstance(identifier, str))
		reports: list[MonitoringReport] = []
		for index, raw_report in enumerate(raw_reports, start=1):
			if not isinstance(raw_report, Mapping):
				raise ValueError(f"{agent.name} monitoring LLM report must be an object.")
			raw_evidence = _evidence_tuple(raw_report.get("evidence", ()))
			evidence = tuple(identifier for identifier in raw_evidence if identifier in allowed_evidence)
			dropped_evidence = tuple(identifier for identifier in raw_evidence if identifier not in allowed_evidence)
			if dropped_evidence:
				debug(
					"MONITORING_EVIDENCE_FILTER",
					agent=agent.name,
					report_index=index,
					dropped=list(dropped_evidence),
					allowed=sorted(allowed_evidence),
					raw_report=dict(raw_report),
				)
			identifier = evidence[0] if evidence else (events[index - 1].identifier if index <= len(events) else min(allowed_evidence, default=agent.identity_name))
			evidence = evidence or (identifier,)
			event = next((candidate for candidate in events if candidate.identifier == identifier), None)
			defaults = classify_identifier(
				identifier,
				critical_overheating_ids=critical_ids,
			)
			machine_id = agent._report_machine_id(identifier)
			if machine_id != machine_id_from_identifier(identifier):
				raise ValueError(f"{agent.name} monitoring LLM evidence does not match the agent machine scope.")
			raw_diagnosis = raw_report.get("diagnosis")
			raw_component = raw_report.get("component")
			raw_action = raw_report.get("recommended_action")
			raw_confidence = raw_report.get("confidence")
			diagnosis = defaults.diagnosis
			component = defaults.component
			action = _label_or_default(raw_action, defaults.recommended_action, ACTION_LABELS)
			if action not in agent._llm_allowed_actions():
				action = defaults.recommended_action
			selected_diagnosis_id = defaults.diagnosis_id
			action = _action_or_default_for_diagnosis_id(action, selected_diagnosis_id, defaults.recommended_action)
			confidence = _label_or_default(raw_confidence, "medium", CONFIDENCE_LEVELS)
			repairs = {
				"dropped_evidence": list(dropped_evidence),
				"diagnosis_repaired": diagnosis != raw_diagnosis,
				"component_repaired": component != raw_component,
				"recommended_action_repaired": action != raw_action,
				"confidence_repaired": confidence != raw_confidence,
			}
			repairs = {key: value for key, value in repairs.items() if value not in (False, [])}
			default_time = event.context.get("time", context.window.end_time) if event is not None else context.window.end_time
			report = MonitoringReport(
				report_id=agent._report_id(machine_id, context.window.start_time, index),
				agent_role=agent.role,
				component=component,  # type: ignore[arg-type]
				machine_id=machine_id,
				time=_report_time(raw_report.get("time"), default_time),
				diagnosis=diagnosis,
				recommended_action=action,
				confidence=confidence,
				evidence=evidence,
				rationale=str(raw_report.get("rationale", f"{agent.name} analyzed scoped evidence.")),
				diagnosis_id=selected_diagnosis_id,
				agent_name=agent.identity_name,
				agent_model=model,
				agent_kind="llm_agent",
				metadata={"persona": agent._persona(), "implementation": "llm", "canonicalization_repairs": repairs},
			)
			debug("MONITORING_REPORT_CANONICALIZED", agent=agent.identity_name, raw_report=dict(raw_report), canonical_report=report_summary(report))
			reports.append(report)
		return tuple(reports)
