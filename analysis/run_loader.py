from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import pandas as pd
from matplotlib.figure import Figure

from cps.core.reporting import ReportedEvent
from cps.evaluation.cascade import score_cascades

_REPORTED_EVENT_FIELDS = frozenset(field.name for field in fields(ReportedEvent))

@dataclass(frozen=True)
class RunData:
	run_dir: Path
	run_id: str
	raw: dict[str, Any]
	ground_truth: pd.DataFrame
	reports: pd.DataFrame
	runtime_reports: pd.DataFrame
	conflicts: pd.DataFrame
	decisions: pd.DataFrame
	decisions_with_conflicts: pd.DataFrame
	agent_actions: pd.DataFrame
	per_fault: pd.DataFrame
	detection_per_fault: pd.DataFrame
	resolution_per_decision: pd.DataFrame
	cascade_per_fault: pd.DataFrame
	events: pd.DataFrame
	runtime_metrics: dict[str, Any]
	detection_metrics: dict[str, Any]
	runtime_detection_metrics: dict[str, Any]
	per_fault_metrics: dict[str, Any]
	resolver_metrics: dict[str, Any]
	resolution_correctness: dict[str, Any]
	ground_truth_audit: dict[str, Any]
	cascade: dict[str, Any]


def load_run(run_dir: Path) -> RunData:
	resolved = run_dir.expanduser().resolve()
	record = _read_single_record(resolved / "runs.jsonl")
	runtime_correctness = record.get("runtime_correctness") or {}
	detection = record.get("detection_metrics") or {}
	runtime_detection = record.get("runtime_detection") or {
		"per_fault": runtime_correctness.get("per_fault"),
		"metrics": runtime_correctness.get("metrics"),
	}
	per_fault_outcomes = record.get("per_fault_outcomes") or runtime_correctness
	resolution_correctness = record.get("resolver_correctness") or runtime_correctness.get("resolution_correctness") or {}
	cascade = _derive_cascade(record)
	conflicts = _dedupe_conflict_types(_frame(record.get("detected_conflicts")))
	decisions = _frame(record.get("runtime_llm_decisions"))
	decisions_with_conflicts = join_decisions_conflicts(decisions, conflicts)

	return RunData(
		run_dir=resolved,
		run_id=resolved.name,
		raw=record,
		ground_truth=_frame(record.get("ground_truth")),
		reports=_frame(record.get("generated_reports")),
		runtime_reports=_frame(record.get("runtime_llm_reports")),
		conflicts=conflicts,
		decisions=decisions,
		decisions_with_conflicts=decisions_with_conflicts,
		agent_actions=_frame(record.get("agent_actions")),
		per_fault=_frame(per_fault_outcomes.get("per_fault")),
		detection_per_fault=_frame(detection.get("per_fault")),
		resolution_per_decision=_frame(resolution_correctness.get("per_decision")),
		cascade_per_fault=_frame(cascade.get("per_fault")),
		events=_frame(record.get("events")),
		runtime_metrics=runtime_correctness.get("metrics") or {},
		detection_metrics=detection.get("metrics") or {},
		runtime_detection_metrics=runtime_detection.get("metrics") or {},
		per_fault_metrics=per_fault_outcomes.get("metrics") or {},
		resolver_metrics=resolution_correctness.get("overall") or {},
		resolution_correctness=resolution_correctness,
		ground_truth_audit=record.get("ground_truth_audit") or {},
		cascade=cascade,
	)


def _derive_cascade(record: dict[str, Any]) -> dict[str, Any]:
	saved = record.get("cascade")
	if isinstance(saved, dict) and saved:
		derived = _score_cascade_from_record(record)
		if not derived:
			return saved
		saved_metrics = saved.get("metrics") if isinstance(saved.get("metrics"), dict) else {}
		derived_metrics = derived.get("metrics") if isinstance(derived.get("metrics"), dict) else {}
		saved_metrics = cast(dict[str, Any], saved_metrics)
		derived_metrics = cast(dict[str, Any], derived_metrics)
		return {
			**derived,
			**saved,
			"metrics": {**derived_metrics, **saved_metrics},
			"per_fault": saved.get("per_fault") or derived.get("per_fault"),
		}

	return _score_cascade_from_record(record)


def _score_cascade_from_record(record: dict[str, Any]) -> dict[str, Any]:
	events = record.get("events")
	ground_truth = record.get("ground_truth")
	event_window = record.get("event_window")
	if isinstance(events, list) and isinstance(ground_truth, list) and isinstance(event_window, dict):
		window_end = event_window.get("end")
		if isinstance(window_end, int | float):
			return score_cascades(
				[_reported_event_from_record(event) for event in events if isinstance(event, dict)],
				ground_truth,
				window_end=float(window_end),
			)
	return {}


def _reported_event_from_record(record: dict[str, Any]) -> ReportedEvent:
	return ReportedEvent(**{key: value for key, value in record.items() if key in _REPORTED_EVENT_FIELDS})


def run_summary(run: RunData) -> dict[str, Any]:
	_, throughput = parse_simulation_log(run.run_dir)
	completed_products = len(throughput) if not throughput.empty else None
	cascade_metrics = run.cascade.get("metrics") or {}
	ground_truth_source_counts = _ground_truth_source_counts(run)
	
	return {
		"run_id": run.run_id,
		"duration": _event_window_duration(run.raw.get("event_window")),
		"completed_products": completed_products,
		"root_faults": _root_fault_count(run),
		"all_faults": len(run.ground_truth),
		"physical_state_issues": ground_truth_source_counts.get("physical_state", 0),
		"derived_issues": ground_truth_source_counts.get("derived_issue", 0),
		"runtime_reports": len(run.runtime_reports),
		"conflicts": len(run.conflicts),
		"resolver_decisions": len(run.decisions),
		"agent_actions": len(run.agent_actions),
		"per_fault_diagnosis_rate": _metric(run.per_fault_metrics, "per_fault_diagnosis_recall", "diagnosis_correct_rate", "root_diagnosis_correct_rate"),
		"per_fault_action_rate": _metric(run.per_fault_metrics, "per_fault_action_selected_rate", "action_selected_rate", "root_action_selected_rate"),
		"runtime_mean_detection_latency": _metric(run.runtime_detection_metrics, "mean_detection_latency"),
		"runtime_median_detection_latency": _metric(run.runtime_detection_metrics, "median_detection_latency"),
		"detection_mean_detection_latency": _metric(run.detection_metrics, "mean_detection_latency"),
		"detection_median_detection_latency": _metric(run.detection_metrics, "median_detection_latency"),
		"resolver_diagnosis_accuracy": _metric(run.resolver_metrics, "diagnosis_accuracy"),
		"resolver_action_accuracy": _metric(run.resolver_metrics, "action_accuracy"),
		"action_attempt_success_rate": _action_attempt_success_rate(run.agent_actions),
		"required_action_execution_rate": _required_action_execution_rate(run.per_fault),
		"cascaded_root_faults": _metric(cascade_metrics, "n_cascaded", "cascaded_root_faults"),
		"contained_root_faults": _metric(cascade_metrics, "n_contained", "contained_root_faults"),
		"containable_root_faults": _metric(cascade_metrics, "n_containable"),
		"structurally_cascading_root_faults": _metric(cascade_metrics, "n_structurally_cascading"),
		"contained_given_containable_root_faults": _metric(cascade_metrics, "n_contained_given_containable"),
		"contained_rate_over_containable": _metric(cascade_metrics, "contained_rate_over_containable"),
	}


def _event_window_duration(event_window: object) -> float | None:
	if not isinstance(event_window, dict):
		return None
	start = event_window.get("start")
	end = event_window.get("end")
	if not isinstance(start, int | float) or not isinstance(end, int | float):
		return None
	return float(end) - float(start)


def join_decisions_conflicts(decisions: pd.DataFrame, conflicts: pd.DataFrame) -> pd.DataFrame:
	if decisions.empty or conflicts.empty or "conflict_id" not in decisions or "conflict_id" not in conflicts:
		return decisions.copy()
	keep = ["conflict_id", "conflict_types", "diagnoses", "actions", "report_ids", "window", "machine_id", "description"]
	available = [column for column in keep if column in conflicts.columns]
	return decisions.merge(conflicts[available], on="conflict_id", how="left", suffixes=("", "_conflict"))


def _dedupe_conflict_types(conflicts: pd.DataFrame) -> pd.DataFrame:
	"""Collapse repeated dimensions within a conflict's ``conflict_types`` list.

	The conflict detector can record the same dimension more than once for a
	single conflict (e.g. ``['confidence', 'confidence']``). Those duplicates are
	semantically a single-dimension conflict, so they are de-duplicated here to
	keep both the frequency and the exact-combination charts consistent. Order is
	preserved; downstream consumers sort when they need a canonical key.
	"""
	if conflicts.empty or "conflict_types" not in conflicts:
		return conflicts

	def _dedupe(values: object) -> object:
		if not isinstance(values, list):
			return values
		seen: list[object] = []
		for value in values:
			if value not in seen:
				seen.append(value)
		return seen

	conflicts = conflicts.copy()
	conflicts["conflict_types"] = conflicts["conflict_types"].map(_dedupe)
	return conflicts


def conflict_type_counts(conflicts: pd.DataFrame) -> pd.DataFrame:
	if conflicts.empty or "conflict_types" not in conflicts:
		return pd.DataFrame(data=[], columns=pd.Index(["conflict_type", "count"]))
	counts: dict[str, int] = {}
	for values in conflicts["conflict_types"].dropna():
		if isinstance(values, list):
			for value in values:
				counts[str(value)] = counts.get(str(value), 0) + 1
	return pd.DataFrame([{"conflict_type": key, "count": value} for key, value in counts.items()]).sort_values("count", ascending=False)


def decision_failure_taxonomy(decisions: pd.DataFrame) -> pd.DataFrame:
	columns = ["failure_mode", "decisions", "share"]
	if decisions.empty:
		return pd.DataFrame(data=[], columns=pd.Index(columns))
	classified = decisions.copy()
	classified["failure_mode"] = _decision_failure_modes(classified)
	counts = classified.groupby("failure_mode", dropna=False).size().to_frame("decisions").reset_index()
	counts["share"] = counts["decisions"] / len(classified)
	return cast(pd.DataFrame, counts.sort_values(["decisions", "failure_mode"], ascending=[False, True])[columns])


def example_decisions(run: RunData, n_per_mode: int = 1) -> pd.DataFrame:
	columns = [
		"failure_mode",
		"decision_id",
		"conflict_id",
		"conflict_types",
		"conflict_window",
		"selected_diagnosis_id",
		"selected_action",
		"expected_action",
		"matched_truth_source",
		"derived_issue_attribution_status",
	]
	if run.resolution_per_decision.empty:
		return pd.DataFrame(data=[], columns=pd.Index(columns))

	examples = run.resolution_per_decision.copy()
	examples["failure_mode"] = _decision_failure_modes(examples)
	if "conflict_window" not in examples and "window" in examples:
		examples["conflict_window"] = examples["window"]
	elif "conflict_window" not in examples and "conflict_id" in examples and "conflict_id" in run.conflicts and "window" in run.conflicts:
		conflict_windows: dict[Any, Any] = {}
		for conflict_id, window in zip(_series(run.conflicts, "conflict_id"), _series(run.conflicts, "window"), strict=False):
			conflict_windows.setdefault(conflict_id, window)
		examples["conflict_window"] = cast(pd.Series, examples["conflict_id"]).map(conflict_windows)
	elif "conflict_window" not in examples and "conflict_id" in examples and {"conflict_id", "window.start", "window.end"}.issubset(run.conflicts.columns):
		conflict_windows = {}
		for conflict_id, window_start, window_end in zip(
			_series(run.conflicts, "conflict_id"),
			_series(run.conflicts, "window.start"),
			_series(run.conflicts, "window.end"),
			strict=False,
		):
			conflict_windows.setdefault(
				conflict_id,
				{"start": float(window_start), "end": float(window_end)},
			)
		examples["conflict_window"] = cast(pd.Series, examples["conflict_id"]).map(conflict_windows)
	modes = [
		"active_action_correct",
		"passive_after_root_handled",
		"passive_without_root_handling",
		"unattributable_derived_issue",
	]
	rows = cast(pd.DataFrame, examples[examples["failure_mode"].isin(modes)].groupby("failure_mode", sort=False, dropna=False).head(n_per_mode))
	for column in columns:
		if column not in rows:
			rows[column] = None
	return cast(pd.DataFrame, rows[columns].reset_index(drop=True))


def _decision_failure_modes(decisions: pd.DataFrame) -> pd.Series:
	classified = decisions.copy()

	def column(name: str, default: Any) -> pd.Series:
		if name in classified.columns:
			return cast(pd.Series, classified[name])
		return pd.Series(default, index=classified.index)

	diagnosis_correct = column("diagnosis_correct", True)
	if "diagnosis_correct" not in classified.columns and "selected_diagnosis_matches_truth" in classified.columns:
		diagnosis_correct = column("selected_diagnosis_matches_truth", True)
	has_scorer_correctness = any(name in classified.columns for name in ("diagnosis_correct", "action_correct"))

	failure_mode = pd.Series("active_action_correct", index=classified.index)
	failure_mode.loc[
		column("selected_action", "").eq("wait_for_more_evidence")
		& column("root_action_already_handled", False).eq(True)
		& column("action_correct", False).eq(True)
	] = "passive_after_root_handled"
	failure_mode.loc[diagnosis_correct.eq(False)] = "wrong_diagnosis"
	failure_mode.loc[failure_mode.isin(["active_action_correct", "passive_after_root_handled"]) & column("action_correct", True).eq(False)] = "wrong_action"
	failure_mode.loc[
		column("derived_issue_attribution_status", "").isin(["no_matching_issue_event", "no_root_cause_found"])
	] = "unattributable_derived_issue"
	failure_mode.loc[
		column("selected_action", "").eq("wait_for_more_evidence")
		& column("root_action_already_handled", True).eq(False)
		& column("action_correct", True).eq(False)
	] = "passive_without_root_handling"
	failure_mode.loc[
		(~failure_mode.isin(["active_action_correct", "passive_after_root_handled"]) | (not has_scorer_correctness))
		& (column("selected_diagnosis_in_ground_truth", True).eq(False) | column("matched_truth_source", "present").isna())
	] = "missing_truth"
	return failure_mode


def duplicate_ground_truth_diagnosis_audit(run: RunData) -> pd.DataFrame:
	duplicate_diagnoses = run.ground_truth_audit.get("duplicate_diagnoses")
	if duplicate_diagnoses:
		return pd.json_normalize(duplicate_diagnoses)
	if run.ground_truth.empty or "diagnosis" not in run.ground_truth:
		return pd.DataFrame(columns=pd.Index(["diagnosis", "count"]))
	grouped = run.ground_truth.groupby("diagnosis", dropna=False).agg(count=("diagnosis", "size")).reset_index()
	duplicates = cast(pd.DataFrame, grouped[grouped["count"] > 1])
	return cast(pd.DataFrame, duplicates.sort_values("count", ascending=False))  # pyright: ignore[reportCallIssue]


def _ground_truth_source_counts(run: RunData) -> dict[str, int]:
	if "source" not in run.ground_truth:
		return {}
	return {str(key): int(value) for key, value in run.ground_truth["source"].value_counts(dropna=False).items()}


def ground_truth_audit_table(run: RunData) -> pd.DataFrame:
	rows = [{"measure": "all_faults", "value": len(run.ground_truth)}]
	if "source" in run.ground_truth:
		source_measure_names = {
			"root_fault": "root_faults",
			"physical_state": "physical_state_issues",
			"derived_issue": "derived_issues",
		}
		rows.extend(
			{"measure": source_measure_names.get(str(key), f"source:{key}"), "value": int(value)}
			for key, value in run.ground_truth["source"].value_counts(dropna=False).items()
		)
		detection_evaluable = cast(pd.Series, run.ground_truth["source"]).isin(["root_fault", "physical_state"])
		if "evaluable" in run.ground_truth:
			detection_evaluable &= cast(pd.Series, run.ground_truth["evaluable"]).eq(True)
		rows.append(
			{
				"measure": "detection_evaluable_faults",
				"value": int(detection_evaluable.sum()),
			}
		)
	if "evaluation_role" in run.ground_truth:
		rows.extend({"measure": f"evaluation_role:{key}", "value": value} for key, value in run.ground_truth["evaluation_role"].value_counts(dropna=False).items())
	if "evaluable" in run.ground_truth:
		rows.append({"measure": "evaluable_rows", "value": int(run.ground_truth["evaluable"].eq(True).sum())})
	rows.append({"measure": "duplicate_diagnosis_groups", "value": len(duplicate_ground_truth_diagnosis_audit(run))})
	return pd.DataFrame(rows)


def traceability_audit(run: RunData) -> pd.DataFrame:
	report_ids = set(_series(run.runtime_reports, "report_id").dropna())
	conflict_report_ids = _flatten(_series(run.conflicts, "report_ids"))
	decision_support_ids = _flatten(_series(run.decisions, "supporting_report_ids"))
	action_report_ids = list(_series(run.agent_actions, "report_id").dropna())
	decision_conflicts = set(_series(run.decisions, "conflict_id").dropna())
	conflict_ids = set(_series(run.conflicts, "conflict_id").dropna())
	rows = [
		_rate_row("conflict_report_ids_resolve", conflict_report_ids, report_ids),
		_rate_row("decision_support_ids_resolve", decision_support_ids, report_ids),
		_rate_row("agent_action_report_ids_resolve", action_report_ids, report_ids),
		{"measure": "runtime_report_id_uniqueness", "value": len(report_ids) / len(run.runtime_reports) if len(run.runtime_reports) else None, "numerator": len(report_ids), "denominator": len(run.runtime_reports)},
		{"measure": "decision_to_conflict_match_rate", "value": len(decision_conflicts & conflict_ids) / len(decision_conflicts) if decision_conflicts else None, "numerator": len(decision_conflicts & conflict_ids), "denominator": len(decision_conflicts)},
	]
	return pd.DataFrame(rows)


def save_fig(fig: Figure, name: str, figures_dir: Path) -> None:
	figures_dir.mkdir(parents=True, exist_ok=True)
	fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight", dpi=200)


def parse_simulation_log(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	path = run_dir / "simulation.log"
	fault_columns = ["fault_key", "machine", "component", "start_t", "end_t", "repair_time"]
	throughput_columns = ["product_id", "time", "sequence"]
	if not path.exists():
		return pd.DataFrame(data=[], columns=pd.Index(fault_columns)), pd.DataFrame(data=[], columns=pd.Index(throughput_columns))
	starts: dict[str, float] = {}
	faults: list[dict[str, Any]] = []
	throughput: list[dict[str, Any]] = []
	sequence = 0
	for line in path.read_text(errors="replace").splitlines():
		if match := re.search(r"KPI: Fault started for (?P<key>.+?) at T=(?P<time>[0-9.]+)", line):
			starts[match.group("key")] = float(match.group("time"))
		elif match := re.search(r"KPI: Fault ended for (?P<key>.+?) at T=(?P<time>[0-9.]+)\. Repair time: (?P<repair>[0-9.]+)", line):
			key = match.group("key")
			machine, component = _split_fault_key(key)
			faults.append(
				{
					"fault_key": key,
					"machine": machine,
					"component": component,
					"start_t": starts.get(key),
					"end_t": float(match.group("time")),
					"repair_time": float(match.group("repair")),
				}
			)
		elif match := re.search(r"Stored completed product (?P<product>[\w-]+)(?: at T=(?P<time>[0-9]+\.[0-9]+))?", line):
			sequence += 1
			logged_time = match.group("time")
			throughput.append(
				{
					"product_id": match.group("product"),
					"time": float(logged_time) if logged_time is not None else float(sequence),
					"sequence": sequence,
				}
			)
	return pd.DataFrame(faults, columns=pd.Index(fault_columns)), pd.DataFrame(throughput, columns=pd.Index(throughput_columns))


def _read_single_record(path: Path) -> dict[str, Any]:
	with path.open() as handle:
		for line in handle:
			if line.strip():
				return json.loads(line)
	msg = f"No JSON records found in {path}"
	raise ValueError(msg)


def _frame(value: Any) -> pd.DataFrame:
	if isinstance(value, list):
		return pd.json_normalize(value)
	if isinstance(value, dict):
		return pd.json_normalize([value])
	return pd.DataFrame()


def _split_fault_key(key: str) -> tuple[str, str]:
	for suffix in ("ActuatorSensor", "Temperature", "Actuator", "Machine", "Power", "Belt", "Network"):
		marker = f"-{suffix}"
		if key.endswith(marker):
			return key[: -len(marker)], suffix
	return key, "unknown"


def _metric(metrics: dict[str, Any], *names: str) -> Any:
	for name in names:
		value = metrics.get(name)
		if value is not None:
			return value
	return None


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
	if column in frame:
		return cast(pd.Series, frame[column])
	return pd.Series(dtype=object)


def _root_fault_count(run: RunData) -> int:
	if "evaluation_role" in run.ground_truth:
		root_rows = run.ground_truth[run.ground_truth["evaluation_role"].isin(["root_fault", "root"])]
		if not root_rows.empty:
			return len(root_rows)
	if "root_fault" in run.ground_truth:
		return int(run.ground_truth["root_fault"].dropna().nunique())
	return len(run.ground_truth)


def _action_attempt_success_rate(actions: pd.DataFrame) -> float | None:
	if actions.empty or "execution_attempted" not in actions or "execution_succeeded" not in actions:
		return None
	attempted = actions[cast(pd.Series, actions["execution_attempted"]).eq(True)]
	if "execution_outcome" in attempted:
		outcome_mask = ~cast(pd.Series, attempted["execution_outcome"]).isin({"already_resolved", "obsolete"})
		attempted = attempted.loc[outcome_mask]
	if "failure_reason" in attempted:
		failure_mask = ~cast(pd.Series, attempted["failure_reason"]).eq("condition_already_resolved")
		attempted = attempted.loc[failure_mask]
	if attempted.empty:
		return None
	return float(cast(pd.Series, attempted["execution_succeeded"]).eq(True).mean())


def _required_action_execution_rate(per_fault: pd.DataFrame) -> float | None:
	if per_fault.empty:
		return None
	for column in ("action_executed", "required_action_executed"):
		if column in per_fault:
			return float(per_fault[column].fillna(False).mean())
	return None


def _flatten(series: pd.Series) -> list[Any]:
	values: list[Any] = []
	for item in series.dropna():
		if isinstance(item, list):
			values.extend(item)
		else:
			values.append(item)
	return values


def _rate_row(measure: str, values: list[Any], valid_values: set[Any]) -> dict[str, Any]:
	matches = sum(value in valid_values for value in values)
	return {
		"measure": measure,
		"value": matches / len(values) if values else None,
		"numerator": matches,
		"denominator": len(values),
	}
