"""Prompt builders for LLM-backed conflict detection."""

from cps.agents.contracts import CONFLICT_TYPES, EvidenceWindow, MonitoringReport
from cps.agents.llm import schema
from cps.core.reporting import ReportedEvent


def response_format() -> dict[str, object]:
	"""Strict json_schema constraining the detector's conflicts object."""
	return schema.response_format(
		"conflict_detection",
		schema.strict_object(
			{
				"conflicts": schema.object_array(
					{
						"report_ids": schema.string_array(),
						"conflict_types": schema.enum_array(CONFLICT_TYPES),
						"description": {"type": "string"},
					}
				),
			}
		),
	)


def system_prompt() -> str:
	return (
		"You are the conflict-detection layer of a cyber-physical manufacturing "
		"line monitor. A polling cycle produced monitoring reports from several "
		"role-specific agents plus raw observable events. Decide which reports, if "
		"any, conflict with each other.\n\n"
		"Your reply is constrained to a JSON schema with a conflicts array. Each "
		"conflict's report_ids must be two or more ids copied exactly from the "
		"reports below; description is one short sentence.\n\n"
		"Assign each report id to at most one conflict; when several reports "
		"compete over the same situation, group them into a single conflict. "
		"Group reports into one conflict only when they offer competing explanations "
		"or competing recovery actions for the same underlying situation -- e.g. a "
		"network-wide cause versus a machine-local cause for the same symptoms, or "
		"an upstream belt fault versus downstream symptom reports. Independent "
		"unrelated faults on different machines are not a conflict; include in "
		"conflicts only reports that compete, and return an empty conflicts list "
		"when no reports compete."
	)


def _observable_event_record(event: ReportedEvent) -> dict[str, object]:
	return {
		"identifier": event.identifier,
		"kind": event.kind,
		"component": event.component,
		"context": dict(event.context),
	}


def format_reports(reports: tuple[MonitoringReport, ...] | list[MonitoringReport], *, window: EvidenceWindow) -> str:
	lines = [
		f"Evidence window: {window.start_time:.1f}-{window.end_time:.1f}",
		"",
		"Monitoring reports:",
	]
	for report in reports:
		lines.append(
			f"- report_id={report.report_id} machine={report.machine_id or 'line-wide'} "
			f"[{report.agent_name or report.agent_role}; kind={report.agent_kind}; "
			f"model={report.agent_model}; role={report.agent_role}] component={report.component} diagnosis={report.diagnosis} "
			f"(id={report.diagnosis_id}) action={report.recommended_action} "
			f"confidence={report.confidence} "
			f"evidence={list(report.evidence)} :: {report.rationale}"
		)
	lines.extend(["", "Observable events in this window:"])
	if window.events:
		for event in window.events:
			lines.append(f"- {_observable_event_record(event)}")
	else:
		lines.append("- none")
	return "\n".join(lines)
