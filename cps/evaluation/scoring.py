from collections.abc import Iterable
from typing import Any, cast

from cps.agents.identifiers import parse_identifier
from cps.agents.report_selection import PASSIVE_ACTIONS
from cps.core.reporting import ReportedEvent
from cps.evaluation.event_records import (
	DerivedIssueAttribution,
	derived_issue_attribution,
	root_fault_injection_times,
	root_fault_resolution_times,
)
from cps.evaluation.ground_truth import (
	evaluable_fault_ground_truth,
	evaluable_ground_truth,
	physical_state_response,
	root_fault_ground_truth,
	string_list,
)


def score_report_diagnoses(
	reports: Iterable[dict[str, Any]],
	ground_truth: Iterable[dict[str, Any]],
) -> dict[str, Any]:
	truth = evaluable_fault_ground_truth(ground_truth)
	report_list = list(reports)
	expected_diagnoses = {str(item["diagnosis"]) for item in truth}
	reported_diagnoses = {
		diagnosis_id
		for report in report_list
		if isinstance((diagnosis_id := report.get("diagnosis_id")), str)
		if parse_identifier(diagnosis_id).kind in {"sensor", "actuator", "network"} or diagnosis_id in expected_diagnoses
	}
	detected_diagnoses = reported_diagnoses & expected_diagnoses
	per_fault = [_score_detection(item, report_list) for item in truth]
	latencies = sorted(
		float(latency)
		for result in per_fault
		if isinstance((latency := result["detection_latency"]), int | float)
	)
	return {
		"per_fault": per_fault,
		"metrics": {
			"evaluable_faults": len(truth),
			"diagnosis_precision": len(detected_diagnoses) / len(reported_diagnoses) if reported_diagnoses else None,
			"mean_detection_latency": sum(latencies) / len(latencies) if latencies else None,
			"median_detection_latency": _median(latencies),
		},
	}


def score_agent_decisions(
	agent_actions: Iterable[dict[str, Any]],
	resolution_decisions: Iterable[dict[str, Any]],
	ground_truth: Iterable[dict[str, Any]],
	generated_reports: Iterable[dict[str, Any]] = (),
	detected_conflicts: Iterable[dict[str, Any]] = (),
	events: Iterable[ReportedEvent] = (),
) -> dict[str, Any]:
	decision_truth = evaluable_ground_truth(ground_truth)
	truth = evaluable_fault_ground_truth(decision_truth)
	actions = list(agent_actions)
	decisions = list(resolution_decisions)
	reports = list(generated_reports)
	event_list = list(events)
	if not truth:
		metrics = _per_fault_outcome_metrics([], diagnosis_metrics={})
		return {
			"missing_diagnoses": [],
			"missing_actions": [],
			"unexpected_actions": _selected_agent_actions(actions),
			"missing_action_pairs": [],
			"unexpected_action_pairs": [],
			"per_fault": [],
			"resolution_correctness": _score_resolution_decisions(decisions, decision_truth, detected_conflicts, event_list, actions),
			"metrics": metrics,
		}
	expected_diagnoses = {str(item["diagnosis"]) for item in truth}
	expected_action_pairs = {
		(str(item["diagnosis"]), str(item["required_action"]))
		for item in truth
		if isinstance(item.get("required_action"), str)
	}
	# Chained actions are reported against the chain-effect diagnosis the agent
	# actually observed (e.g. start_cooling against temperature:<m>:overheating),
	# not against the root-fault diagnosis, so tolerate them under both.
	tolerated_action_pairs = expected_action_pairs | {
		(diagnosis, action)
		for item in truth
		for action in _actions_for_chain_effects(item)
		for diagnosis in (str(item["diagnosis"]), *string_list(item.get("chain_effects")))
	}
	detection = score_report_diagnoses(reports, truth)
	detection_results = cast(list[dict[str, object]], detection["per_fault"])
	actual_diagnoses = {result["diagnosis"] for result in detection_results if result["detected"]}
	actual_diagnoses.update(diagnosis_id for decision in decisions if isinstance((diagnosis_id := decision.get("selected_diagnosis_id")), str))
	attributed_decisions = _attributed_decision_actions(decisions, decision_truth, detected_conflicts, event_list)
	actual_diagnoses.update(
		diagnosis_id
		for decision in attributed_decisions
		if isinstance((diagnosis_id := decision.get("diagnosis_id")), str)
	)
	actual_action_pairs = _selected_action_pairs(actions, attributed_decisions)
	missing_diagnoses = sorted(expected_diagnoses - actual_diagnoses)
	missing_action_pairs = sorted(expected_action_pairs - actual_action_pairs)
	unexpected_action_pairs = sorted(
		pair
		for pair in actual_action_pairs - tolerated_action_pairs
		if not _is_operational_state_response(*pair)
	)
	missing_actions = sorted({action for _diagnosis, action in missing_action_pairs})
	unexpected_actions = sorted({action for _diagnosis, action in unexpected_action_pairs})
	per_fault = [_score_fault(item, actions, decisions, attributed_decisions, reports) for item in truth]
	diagnosis_metrics = cast(dict[str, object], detection["metrics"])
	attempted_actions = [action for action in actions if action.get("execution_attempted") is True]
	executed_required_pairs = _executed_action_pairs(actions) & expected_action_pairs
	resolution_correctness = _score_resolution_decisions(decisions, decision_truth, detected_conflicts, event_list, actions)
	metrics = _per_fault_outcome_metrics(
		per_fault,
		diagnosis_metrics=diagnosis_metrics,
		executed_required_pairs=executed_required_pairs,
		expected_action_pairs=expected_action_pairs,
		attempted_actions=attempted_actions,
	)
	return {
		"missing_diagnoses": missing_diagnoses,
		"missing_actions": missing_actions,
		"unexpected_actions": unexpected_actions,
		"missing_action_pairs": _action_pair_records(missing_action_pairs),
		"unexpected_action_pairs": _action_pair_records(unexpected_action_pairs),
		"per_fault": per_fault,
		"resolution_correctness": resolution_correctness,
		"metrics": metrics,
	}


def _score_fault(
	truth: dict[str, object],
	agent_actions: list[dict[str, object]],
	resolution_decisions: list[dict[str, object]],
	attributed_decisions: list[dict[str, object]],
	generated_reports: list[dict[str, object]],
) -> dict[str, object]:
	diagnosis = str(truth["diagnosis"])
	required_action = str(truth["required_action"])
	detection = _score_detection(truth, generated_reports)
	selected_actions = {
		action
		for decision in attributed_decisions
		if decision.get("diagnosis_id") == diagnosis
		if isinstance((action := decision.get("selected_action")), str)
		if action not in PASSIVE_ACTIONS
	}
	executed_actions = {
		str(action["recommended_action"])
		for action in agent_actions
		if action.get("diagnosis_id") == diagnosis and _execution_succeeded(action)
	}
	selected_actions.update(
		str(action["recommended_action"])
		for action in agent_actions
		if action.get("diagnosis_id") == diagnosis
		if isinstance(action.get("recommended_action"), str)
	)
	diagnosis_selected = any(
		decision.get("selected_diagnosis_id") == diagnosis or decision.get("diagnosis_id") == diagnosis
		for decision in [*resolution_decisions, *attributed_decisions]
	)
	return {
		**detection,
		"truth_id": truth.get("truth_id"),
		"evaluation_role": truth.get("evaluation_role", truth.get("source", "root_fault")),
		"source": truth.get("source", "root_fault"),
		"required_action": required_action,
		"required_actions": [required_action],
		"selected_actions": sorted(selected_actions),
		"executed_actions": sorted(executed_actions),
		"diagnosis_correct": detection["detected"] or diagnosis_selected,
		"action_correct": required_action in selected_actions,
		"action_selected": required_action in selected_actions,
		"action_executed": required_action in executed_actions,
	}


def _actions_for_chain_effects(truth: dict[str, object]) -> list[str]:
	actions: list[str] = []
	for effect in string_list(truth.get("chain_effects")):
		parsed = parse_identifier(effect)
		state = parsed.state_or_issue or ""
		response = physical_state_response(parsed.kind, state)
		if response is None:
			continue
		action = response["required_action"].strip()
		if action:
			actions.append(action)
	return actions


def _per_fault_outcome_metrics(
	per_fault: list[dict[str, object]],
	*,
	diagnosis_metrics: dict[str, object],
	executed_required_pairs: set[tuple[str, str]] | None = None,
	expected_action_pairs: set[tuple[str, str]] | None = None,
	attempted_actions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
	executed_required_pairs = executed_required_pairs or set()
	expected_action_pairs = expected_action_pairs or set()
	attempted_actions = attempted_actions or []
	diagnosis_recall = _rate(per_fault, "diagnosis_correct")
	action_selected_rate = _rate(per_fault, "action_correct")
	root_faults = _source_results(per_fault, "root_fault")
	physical_state_faults = _source_results(per_fault, "physical_state")
	metrics: dict[str, object] = {
		"evaluable_faults": len(per_fault),
		"root_faults": len(root_faults),
		"physical_state_faults": len(physical_state_faults),
		"diagnosis_precision": diagnosis_metrics.get("diagnosis_precision"),
		"per_fault_diagnosis_recall": diagnosis_recall,
		"per_fault_action_selected_rate": action_selected_rate,
		"root_per_fault_diagnosis_recall": _rate(root_faults, "diagnosis_correct"),
		"root_per_fault_action_selected_rate": _rate(root_faults, "action_correct"),
		"physical_state_per_fault_diagnosis_recall": _rate(physical_state_faults, "diagnosis_correct"),
		"physical_state_per_fault_action_selected_rate": _rate(physical_state_faults, "action_correct"),
		"fault_action_execution_rate": sum(result["action_executed"] is True for result in per_fault) / len(per_fault) if per_fault else None,
		"required_action_execution_rate": len(executed_required_pairs) / len(expected_action_pairs) if expected_action_pairs else None,
		"action_attempt_success_rate": _action_attempt_success_rate(attempted_actions),
		"mean_detection_latency": diagnosis_metrics.get("mean_detection_latency"),
		"median_detection_latency": diagnosis_metrics.get("median_detection_latency"),
	}
	return metrics


def _score_detection(
	truth: dict[str, object],
	reports: list[dict[str, object]],
) -> dict[str, object]:
	diagnosis = str(truth["diagnosis"])
	matching_reports = [report for report in reports if report.get("diagnosis_id") == diagnosis]
	report_times = [float(time) for report in matching_reports if isinstance((time := report.get("time")), int | float)]
	injected_at = truth.get("injected_at")
	first_report_at = min(report_times) if report_times else None
	detection_latency = (
		max(0.0, first_report_at - float(injected_at))
		if first_report_at is not None and isinstance(injected_at, int | float)
		else None
	)
	return {
		"truth_id": truth.get("truth_id"),
		"evaluation_role": truth.get("evaluation_role", truth.get("source", "root_fault")),
		"diagnosis": diagnosis,
		"injected_at": injected_at,
		"evaluable": True,
		"detected": bool(matching_reports),
		"first_report_at": first_report_at,
		"detection_latency": detection_latency,
	}


def _median(values: list[float]) -> float | None:
	if not values:
		return None
	middle = len(values) // 2
	if len(values) % 2:
		return values[middle]
	return (values[middle - 1] + values[middle]) / 2


def _action_pair_records(pairs: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
	return [{"diagnosis": diagnosis, "action": action} for diagnosis, action in pairs]


def _selected_action_pairs(
	agent_actions: Iterable[dict[str, object]],
	attributed_decisions: Iterable[dict[str, object]],
) -> set[tuple[str, str]]:
	pairs: set[tuple[str, str]] = set()
	for action_record in agent_actions:
		action = action_record.get("recommended_action")
		if not isinstance(action, str):
			continue
		diagnosis = action_record.get("diagnosis_id")
		pairs.add((diagnosis if isinstance(diagnosis, str) else "<unattributed>", action))
	for decision in attributed_decisions:
		action = decision.get("selected_action")
		if not isinstance(action, str) or action in PASSIVE_ACTIONS:
			continue
		diagnosis = decision.get("diagnosis_id")
		pairs.add((diagnosis if isinstance(diagnosis, str) else "<unattributed>", action))
	return pairs


def _executed_action_pairs(agent_actions: Iterable[dict[str, object]]) -> set[tuple[str, str]]:
	pairs: set[tuple[str, str]] = set()
	for action_record in agent_actions:
		action = action_record.get("recommended_action")
		if not _execution_succeeded(action_record) or not isinstance(action, str):
			continue
		diagnosis = action_record.get("diagnosis_id")
		pairs.add((diagnosis if isinstance(diagnosis, str) else "<unattributed>", action))
	return pairs


def _attributed_diagnosis_id(
	diagnosis: str,
	action: str,
	root_fault: str | None,
	root_truth: list[dict[str, object]],
) -> str:
	"""Re-attribute a derived-issue decision to the root-fault diagnosis it repairs."""
	if root_fault is None:
		return diagnosis
	for item in root_truth:
		if item.get("root_fault") != root_fault:
			continue
		if item.get("required_action") == action and isinstance(item.get("diagnosis"), str):
			return str(item["diagnosis"])
	return diagnosis


def _attributed_decision_actions(
	resolution_decisions: Iterable[dict[str, object]],
	ground_truth: Iterable[dict[str, object]],
	detected_conflicts: Iterable[dict[str, object]],
	events: Iterable[ReportedEvent],
) -> list[dict[str, object]]:
	# Build the lookups once rather than per decision: both the attributed
	# diagnosis and its metadata derive from the same truth/conflict maps and the
	# same single attribution per derived-issue decision.
	truth_by_diagnosis = {str(item["diagnosis"]): item for item in evaluable_ground_truth(ground_truth)}
	conflicts_by_id = {
		str(conflict["conflict_id"]): conflict
		for conflict in detected_conflicts
		if isinstance(conflict.get("conflict_id"), str)
	}
	root_truth = root_fault_ground_truth(ground_truth)
	attributed: list[dict[str, object]] = []
	for decision in resolution_decisions:
		action = decision.get("selected_action")
		diagnosis = decision.get("selected_diagnosis_id")
		if not isinstance(action, str) or not isinstance(diagnosis, str):
			continue
		truth = truth_by_diagnosis.get(diagnosis)
		if truth is None or truth.get("source") != "derived_issue":
			attributed_diagnosis = diagnosis
			metadata: dict[str, object] = {
				"attributed_root_fault": None,
				"attribution_status": "not_derived_issue",
			}
		else:
			conflict_id = decision.get("conflict_id")
			conflict = conflicts_by_id.get(conflict_id) if isinstance(conflict_id, str) else None
			attribution = _derived_issue_attribution_for_decision(events, diagnosis, conflict)
			attributed_diagnosis = _attributed_diagnosis_id(diagnosis, action, attribution.root_fault, root_truth)
			metadata = {
				"attributed_root_fault": attribution.root_fault,
				"attribution_status": attribution.attribution_status,
				"matched_derived_issue_event_time": attribution.matched_event_time,
			}
		attributed.append(
			{
				"diagnosis_id": attributed_diagnosis,
				"selected_diagnosis_id": diagnosis,
				"attributed_diagnosis_id": attributed_diagnosis,
				"selected_action": action,
				**metadata,
			}
		)
	return attributed


def _selected_agent_actions(agent_actions: Iterable[dict[str, object]]) -> list[str]:
	actions: list[str] = []
	for action in agent_actions:
		if not _execution_succeeded(action):
			continue
		selected_action = action.get("recommended_action")
		if not isinstance(selected_action, str) or selected_action in actions:
			continue
		diagnosis = action.get("diagnosis_id")
		if isinstance(diagnosis, str) and _is_operational_state_response(diagnosis, selected_action):
			continue
		actions.append(selected_action)
	return actions


def _is_operational_state_response(diagnosis: str, action: str) -> bool:
	parsed = parse_identifier(diagnosis)
	if len(parsed.parts) != 3 or parsed.state_or_issue is None:
		return False
	response = physical_state_response(parsed.kind, parsed.state_or_issue)
	return response is not None and action == response["required_action"]


def _execution_succeeded(action: dict[str, object]) -> bool:
	return action.get("execution_succeeded") is True


def _execution_already_resolved(action: dict[str, object]) -> bool:
	return action.get("execution_outcome") in {"already_resolved", "obsolete"} or action.get("failure_reason") == "condition_already_resolved"


def _action_attempt_success_rate(actions: list[dict[str, object]]) -> float | None:
	scored_actions = [action for action in actions if not _execution_already_resolved(action)]
	if not scored_actions:
		return None
	return sum(_execution_succeeded(action) for action in scored_actions) / len(scored_actions)


def _rate(records: Iterable[dict[str, object]], field: str) -> float | None:
	values = [record.get(field) is True for record in records]
	return sum(values) / len(values) if values else None


def _source_results(records: Iterable[dict[str, object]], source: str) -> list[dict[str, object]]:
	return [record for record in records if record.get("source") == source]


def _score_resolution_decisions(
	resolution_decisions: Iterable[dict[str, object]],
	ground_truth: Iterable[dict[str, object]],
	detected_conflicts: Iterable[dict[str, object]],
	events: Iterable[ReportedEvent],
	agent_actions: Iterable[dict[str, object]],
) -> dict[str, object]:
	truth_by_diagnosis = {str(item["diagnosis"]): item for item in evaluable_ground_truth(ground_truth)}
	root_truth_by_fault = {str(item["root_fault"]): item for item in root_fault_ground_truth(ground_truth)}
	conflicts_by_id = {
		str(conflict["conflict_id"]): conflict
		for conflict in detected_conflicts
		if isinstance(conflict.get("conflict_id"), str)
	}
	event_list = list(events)
	injection_times = root_fault_injection_times(event_list)
	resolution_times = root_fault_resolution_times(event_list)
	actions = list(agent_actions)
	per_decision: list[dict[str, object]] = []
	for decision in resolution_decisions:
		diagnosis = decision.get("selected_diagnosis_id")
		truth = truth_by_diagnosis.get(diagnosis) if isinstance(diagnosis, str) else None
		conflict_id = decision.get("conflict_id")
		conflict = conflicts_by_id.get(conflict_id) if isinstance(conflict_id, str) else None
		conflict_types = string_list(conflict.get("conflict_types")) if conflict is not None else []
		conflict_window = conflict.get("window") if conflict is not None and isinstance(conflict.get("window"), dict) else None
		is_derived_issue = truth is not None and truth.get("source") == "derived_issue" and isinstance(diagnosis, str)
		attribution = (
			_derived_issue_attribution_for_decision(event_list, diagnosis, conflict)
			if is_derived_issue and isinstance(diagnosis, str)
			else DerivedIssueAttribution(
				issue_id=str(diagnosis) if isinstance(diagnosis, str) else "",
				matched_event_time=None,
				root_fault=None,
				attribution_status="not_derived_issue",
			)
		)
		root_fault = attribution.root_fault
		root_truth = root_truth_by_fault.get(root_fault) if root_fault is not None else None
		expected_root_diagnosis = root_truth.get("diagnosis") if root_truth is not None else None
		required_action = root_truth.get("required_action") if is_derived_issue and root_truth is not None else truth.get("required_action") if truth is not None else None
		selected_action = decision.get("selected_action")
		root_action_already_handled = (
			is_derived_issue
			and selected_action in PASSIVE_ACTIONS
			and root_fault is not None
			and root_truth is not None
			and _root_is_being_handled(
				root_fault,
				str(root_truth["diagnosis"]),
				str(root_truth["required_action"]),
				actions,
				resolution_decisions,
				conflicts_by_id,
				injection_times,
				resolution_times,
				conflict,
			)
		)
		if is_derived_issue:
			diagnosis_correct = root_fault is not None
			action_correct = (isinstance(required_action, str) and selected_action == required_action) or root_action_already_handled
		else:
			diagnosis_correct = truth is not None
			action_correct = truth is not None and selected_action == required_action
		matched_truth_id = truth.get("truth_id") if truth is not None else None
		matched_truth_source = truth.get("source", "root_fault") if truth is not None else None
		matched_truth_role = truth.get("evaluation_role", matched_truth_source) if truth is not None else None
		# Separate a genuine model self-assessment from the parser fallback: the
		# fallback path stamps confidence="low" + wait_for_more_evidence when no
		# valid decision could be parsed, so "low" alone conflates "model unsure"
		# with "parse failed". metadata.fell_back is the authoritative signal.
		metadata = decision.get("metadata")
		fell_back = bool(metadata.get("fell_back")) if isinstance(metadata, dict) else False
		decision_source = "fallback" if fell_back else "model"
		# Serialized per-decision schema: persisted under resolver_correctness in
		# runs.jsonl and consumed by analysis/run_loader.py (and older notebooks).
		# Several fields look like duplicates but differ by source: e.g.
		# matched_symptom_diagnosis is set only for derived issues, and
		# expected_diagnosis vs expected_root_diagnosis diverge for non-derived
		# rows. Treat field names as a stable contract, not collapsible aliases.
		record = {
			"decision_id": decision.get("decision_id"),
			"conflict_id": conflict_id,
			"conflict_types": conflict_types,
			"conflict_window": dict(conflict_window) if isinstance(conflict_window, dict) else None,
			"selected_diagnosis_id": diagnosis,
			"selected_action": selected_action,
			"matched_truth_id": matched_truth_id,
			"matched_truth_source": matched_truth_source,
			"matched_truth_role": matched_truth_role,
			"matched_diagnosis": diagnosis if truth is not None else None,
			"matched_symptom_diagnosis": diagnosis if is_derived_issue and truth is not None else None,
			"selected_diagnosis_in_ground_truth": truth is not None,
			"selected_diagnosis_matches_truth": truth is not None,
			"derived_issue_attribution_status": attribution.attribution_status,
			"derived_issue_matched_event_time": attribution.matched_event_time,
			"derived_issue_attributed_root_fault": root_fault,
			"expected_diagnosis": expected_root_diagnosis if is_derived_issue else diagnosis if truth is not None else None,
			"expected_root_diagnosis": expected_root_diagnosis,
			"expected_action": required_action,
			"expected_root_fault": root_fault,
			"root_action_already_handled": root_action_already_handled,
			"passive_action_credit_reason": "root_action_already_handled" if root_action_already_handled else None,
			"confidence": decision.get("confidence"),
			"decision_source": decision_source,
			"diagnosis_correct": diagnosis_correct,
			"action_correct": action_correct,
		}
		per_decision.append(record)
	return {
		"per_decision": per_decision,
		"overall": _resolution_metrics(per_decision),
		"by_conflict_type": {
			conflict_type: _resolution_metrics([record for record in per_decision if conflict_type in cast(list[str], record["conflict_types"])])
			for conflict_type in ("diagnosis", "action", "confidence")
		},
	}


def _derived_issue_attribution_for_decision(
	events: Iterable[ReportedEvent],
	diagnosis: str,
	conflict: dict[str, object] | None,
) -> DerivedIssueAttribution:
	window = conflict.get("window") if conflict is not None else None
	window_start = window.get("start") if isinstance(window, dict) else None
	window_end = window.get("end") if isinstance(window, dict) else None
	return derived_issue_attribution(
		events,
		diagnosis,
		window_start=float(window_start) if isinstance(window_start, int | float) else None,
		window_end=float(window_end) if isinstance(window_end, int | float) else None,
	)


def _root_is_being_handled(
	root_fault: str,
	root_diagnosis: str,
	required_action: str,
	agent_actions: Iterable[dict[str, object]],
	resolution_decisions: Iterable[dict[str, object]],
	conflicts_by_id: dict[str, dict[str, object]],
	injection_times: dict[str, float],
	resolution_times: dict[str, float],
	decision_conflict: dict[str, object] | None,
) -> bool:
	if any(
		action.get("diagnosis_id") == root_diagnosis
		and action.get("recommended_action") == required_action
		and _execution_succeeded(action)
		and _action_within_root_episode(action, root_fault, injection_times, resolution_times, decision_conflict)
		for action in agent_actions
	):
		return True
	for decision in resolution_decisions:
		if decision.get("selected_diagnosis_id") != root_diagnosis or decision.get("selected_action") != required_action:
			continue
		conflict_id = decision.get("conflict_id")
		conflict = conflicts_by_id.get(conflict_id) if isinstance(conflict_id, str) else None
		if _within_root_episode(conflict, root_fault, injection_times, resolution_times) and _conflict_at_or_before_decision(conflict, decision_conflict):
			return True
	return False


def _action_within_root_episode(
	action: dict[str, object],
	root_fault: str,
	injection_times: dict[str, float],
	resolution_times: dict[str, float],
	decision_conflict: dict[str, object] | None,
) -> bool:
	time = action.get("time")
	if not isinstance(time, int | float):
		return False
	if not _time_within_root_episode(float(time), root_fault, injection_times, resolution_times):
		return False
	window = decision_conflict.get("window") if decision_conflict is not None else None
	if not isinstance(window, dict):
		return True
	start = window.get("start")
	end = window.get("end")
	if isinstance(start, int | float) and float(time) < float(start):
		return False
	if isinstance(end, int | float) and float(time) > float(end):
		return False
	return True


def _within_root_episode(
	conflict: dict[str, object] | None,
	root_fault: str,
	injection_times: dict[str, float],
	resolution_times: dict[str, float],
) -> bool:
	injected_at = injection_times.get(root_fault)
	if injected_at is None:
		return True
	window = conflict.get("window") if conflict is not None else None
	if not isinstance(window, dict):
		return True
	time = window.get("end")
	if not isinstance(time, int | float):
		time = window.get("start")
	if not isinstance(time, int | float):
		return True
	return _time_within_root_episode(float(time), root_fault, injection_times, resolution_times)


def _conflict_at_or_before_decision(
	conflict: dict[str, object] | None,
	decision_conflict: dict[str, object] | None,
) -> bool:
	decision_end = _conflict_window_time(decision_conflict, prefer="end")
	if decision_end is None:
		return True
	conflict_time = _conflict_window_time(conflict, prefer="end")
	return conflict_time is not None and conflict_time <= decision_end


def _conflict_window_time(conflict: dict[str, object] | None, *, prefer: str) -> float | None:
	window = conflict.get("window") if conflict is not None else None
	if not isinstance(window, dict):
		return None
	value = window.get(prefer)
	if not isinstance(value, int | float):
		value = window.get("start" if prefer == "end" else "end")
	return float(value) if isinstance(value, int | float) else None


def _time_within_root_episode(
	time: float,
	root_fault: str,
	injection_times: dict[str, float],
	resolution_times: dict[str, float],
) -> bool:
	injected_at = injection_times.get(root_fault)
	if injected_at is None:
		return True
	resolved_at = resolution_times.get(root_fault)
	return injected_at <= time <= (resolved_at if resolved_at is not None else float("inf"))


def _resolution_metrics(records: list[dict[str, object]]) -> dict[str, object]:
	if not records:
		return {
			"decisions": 0,
			"diagnosis_accuracy": None,
			"action_accuracy": None,
		}
	return {
		"decisions": len(records),
		"diagnosis_accuracy": _rate(records, "diagnosis_correct"),
		"action_accuracy": _rate(records, "action_correct"),
	}
