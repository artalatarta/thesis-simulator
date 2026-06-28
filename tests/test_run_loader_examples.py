from pathlib import Path

import pandas as pd

import analysis.run_loader as run_loader
from analysis.run_loader import RunData, example_decisions, ground_truth_audit_table, run_summary


def _run_data(**overrides) -> RunData:
	defaults = {
		"run_dir": Path("missing-run-dir"),
		"run_id": "synthetic",
		"raw": {},
		"ground_truth": pd.DataFrame(),
		"reports": pd.DataFrame(),
		"runtime_reports": pd.DataFrame(),
		"conflicts": pd.DataFrame(),
		"decisions": pd.DataFrame(),
		"decisions_with_conflicts": pd.DataFrame(),
		"agent_actions": pd.DataFrame(),
		"per_fault": pd.DataFrame(),
		"detection_per_fault": pd.DataFrame(),
		"resolution_per_decision": pd.DataFrame(),
		"cascade_per_fault": pd.DataFrame(),
		"events": pd.DataFrame(),
		"runtime_metrics": {},
		"detection_metrics": {},
		"runtime_detection_metrics": {},
		"per_fault_metrics": {},
		"resolver_metrics": {},
		"resolution_correctness": {},
		"ground_truth_audit": {},
		"cascade": {},
	}
	defaults.update(overrides)
	return RunData(**defaults)


def test_example_decisions_returns_traceable_rows_for_present_modes() -> None:
	run = _run_data(
		resolution_per_decision=pd.DataFrame(
			[
				{
					"decision_id": "d-correct",
					"conflict_id": "c1",
					"conflict_types": ["diagnosis"],
					"selected_diagnosis_id": "sensor:M1:Power:stuck",
					"selected_action": "calibrate_sensor",
					"expected_action": "calibrate_sensor",
					"matched_truth_source": "root_fault",
					"derived_issue_attribution_status": "not_derived",
					"diagnosis_correct": True,
					"action_correct": True,
				},
				{
					"decision_id": "d-passive",
					"conflict_id": "c2",
					"conflict_types": ["action"],
					"selected_diagnosis_id": "battery:M1:dead_battery",
					"selected_action": "wait_for_more_evidence",
					"expected_action": "replace_battery",
					"matched_truth_source": "root_fault",
					"derived_issue_attribution_status": "not_derived",
					"diagnosis_correct": True,
					"action_correct": False,
					"root_action_already_handled": False,
				},
				{
					"conflict_id": "c3",
					"selected_diagnosis_id": "temperature:M1:overheating",
					"selected_action": "cool_down",
					"expected_action": "cool_down",
					"matched_truth_source": "physical_state",
					"derived_issue_attribution_status": "not_derived",
					"diagnosis_correct": True,
					"action_correct": True,
				},
				{
					"decision_id": "d-derived",
					"conflict_id": "c4",
					"conflict_types": ["diagnosis"],
					"selected_diagnosis_id": "machine:M1:production_blocked",
					"selected_action": "restart_machine",
					"expected_action": "clear_blockage",
					"matched_truth_source": "derived_issue",
					"derived_issue_attribution_status": "no_root_cause_found",
					"diagnosis_correct": False,
					"action_correct": False,
				},
			]
		),
		conflicts=pd.DataFrame(
			[
				{"conflict_id": "c1", "window.start": 1.0, "window.end": 2.0},
				{"conflict_id": "c2", "window.start": 3.0, "window.end": 4.0},
				{"conflict_id": "c3", "window.start": 5.0, "window.end": 6.0},
				{"conflict_id": "c4", "window.start": 7.0, "window.end": 8.0},
			]
		),
	)

	examples = example_decisions(run)

	assert set(examples["failure_mode"]) == {
		"active_action_correct",
		"passive_without_root_handling",
		"unattributable_derived_issue",
	}
	assert list(examples.columns) == [
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
	assert examples.loc[examples["decision_id"].eq("d-correct"), "conflict_window"].iloc[0] == {"start": 1.0, "end": 2.0}


def test_run_summary_labels_runtime_and_detection_metrics_latency() -> None:
	run = _run_data(
		run_dir=Path("missing-run-dir"),
		runtime_detection_metrics={"mean_detection_latency": 10.0, "median_detection_latency": 5.0},
		runtime_metrics={"mean_detection_latency": 99.0, "median_detection_latency": 88.0},
		detection_metrics={"mean_detection_latency": 3.0, "median_detection_latency": 1.0},
	)

	summary = run_summary(run)

	assert summary["runtime_mean_detection_latency"] == 10.0
	assert summary["runtime_median_detection_latency"] == 5.0
	assert summary["detection_mean_detection_latency"] == 3.0
	assert summary["detection_median_detection_latency"] == 1.0
	assert "mean_detection_latency" not in summary
	assert "median_detection_latency" not in summary


def test_derive_cascade_preserves_saved_rows_and_fills_new_metrics(monkeypatch) -> None:
	def fake_score_cascades(events, ground_truth, *, window_end):
		del events, ground_truth, window_end
		return {
			"metrics": {
				"n_cascaded": 1,
				"n_containable": 3,
				"n_structurally_cascading": 2,
				"n_contained_given_containable": 1,
				"contained_rate_over_containable": 1 / 3,
			},
		}

	monkeypatch.setattr(run_loader, "score_cascades", fake_score_cascades)
	record = {
		"cascade": {
			"metrics": {
				"n_cascaded": 1,
			},
		},
		"events": [],
		"ground_truth": [],
		"event_window": {"end": 10.0},
	}

	cascade = run_loader._derive_cascade(record)

	assert cascade["metrics"]["n_containable"] == 3
	assert cascade["metrics"]["contained_rate_over_containable"] == 1 / 3


def test_ground_truth_audit_table_distinguishes_detection_evaluable_faults() -> None:
	run = _run_data(
		ground_truth=pd.DataFrame(
			[
				{"source": "root_fault", "evaluable": True},
				{"source": "root_fault", "evaluable": True},
				{"source": "physical_state", "evaluable": True},
				{"source": "physical_state", "evaluable": False},
				{"source": "derived_issue", "evaluable": True},
			]
		),
	)

	audit = ground_truth_audit_table(run)

	values = dict(zip(audit["measure"], audit["value"], strict=True))
	assert values["all_faults"] == 5
	assert values["root_faults"] == 2
	assert values["physical_state_issues"] == 2
	assert values["derived_issues"] == 1
	assert values["evaluable_rows"] == 4
	assert values["detection_evaluable_faults"] == 3
