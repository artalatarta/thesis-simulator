"""Prompt builders for LLM-backed conflict resolution."""

from cps.agents.contracts import CONFIDENCE_LEVELS, Conflict
from cps.agents.llm import schema


def response_format() -> dict[str, object]:
	"""Strict json_schema constraining the resolver's decision object."""
	return schema.response_format(
		"conflict_resolution",
		schema.strict_object(
			{
				"selected_report_index": {"type": "integer", "minimum": 1},
				"confidence": schema.enum(CONFIDENCE_LEVELS),
				"explanation": {"type": "string"},
			}
		),
	)


def system_prompt() -> str:
	return (
		"You are the conflict-resolution layer of a cyber-physical manufacturing "
		"line monitor. Several role-specific LLM monitoring agents have produced "
		"disagreeing reports about the same machine within one evidence window. "
		"Choose exactly one monitoring report as the winner. Return only its "
		"1-based selected_report_index from the numbered Monitoring reports list: "
		"use 1 for the first report, 2 for the second report, and so on. "
		"Do not modify, generalize, combine, or infer a new diagnosis/action/id; "
		"the system will copy diagnosis, action, and diagnosis_id from the selected report.\n\n"
		"Your reply is constrained to a JSON schema. Set explanation to one short "
		"sentence explaining why the chosen report is the best supported report.\n\n"
		"Prefer a supported corrective action when the evidence clearly identifies "
		"one, but only by choosing a report that already proposes that action. "
		'The action "wait_for_more_evidence" is allowed only by choosing a report '
		"that already proposed it."
	)


def format_conflict(conflict: Conflict) -> str:
	lines = [
		f"Machine: {conflict.machine_id or 'line-wide'}",
		f"Evidence window: {conflict.window.start_time:.1f}-{conflict.window.end_time:.1f}",
		f"Conflict types: {', '.join(conflict.conflict_types)}",
		"",
		"Monitoring reports:",
	]
	for index, report in enumerate(conflict.reports, start=1):
		lines.append(
			f"{index}. report_id={report.report_id} "
			f"[{report.agent_name or report.agent_role}; kind={report.agent_kind}; "
			f"model={report.agent_model}; role={report.agent_role}] component={report.component} diagnosis={report.diagnosis} "
			f"(id={report.diagnosis_id}) action={report.recommended_action} "
			f"confidence={report.confidence} "
			f"evidence={list(report.evidence)} :: {report.rationale}"
		)
	return "\n".join(lines)
