from typing import cast

from cps.core.reporting import ReportedEvent
from cps.evaluation.scoring import score_agent_decisions, score_report_diagnoses

def test_action_correctness_requires_actions_to_match_their_diagnoses() -> None:
	ground_truth = [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"required_action": "fix_stuck",
			"evaluable": True,
		},
		{
			"diagnosis": "actuator:M2:stuck",
			"required_action": "fix_slow_response",
			"evaluable": True,
		},
	]
	decisions = [
		{
			"selected_diagnosis_id": "sensor:M1:Power:stuck",
			"selected_action": "fix_slow_response",
		},
		{
			"selected_diagnosis_id": "actuator:M2:stuck",
			"selected_action": "fix_stuck",
		},
	]

	result = score_agent_decisions([], decisions, ground_truth)

	assert result["missing_actions"] == ["fix_slow_response", "fix_stuck"]
	assert result["unexpected_actions"] == ["fix_slow_response", "fix_stuck"]
	assert result["missing_action_pairs"] == [
		{"diagnosis": "actuator:M2:stuck", "action": "fix_slow_response"},
		{"diagnosis": "sensor:M1:Power:stuck", "action": "fix_stuck"},
	]
	assert result["metrics"]["per_fault_action_selected_rate"] == 0.0


def test_agent_decision_metrics_use_canonical_per_fault_labels() -> None:
	ground_truth = [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"required_action": "fix_stuck",
			"evaluable": True,
		}
	]
	reports = [{"diagnosis_id": "sensor:M1:Power:stuck", "time": 2.0}]
	actions = [
		{
			"diagnosis_id": "sensor:M1:Power:stuck",
			"recommended_action": "fix_stuck",
			"execution_succeeded": True,
			"execution_attempted": True,
		}
	]

	result = score_agent_decisions(actions, [], ground_truth, reports)
	metrics = cast(dict[str, object], result["metrics"])

	assert metrics["per_fault_diagnosis_recall"] == 1.0
	assert metrics["per_fault_action_selected_rate"] == 1.0
	assert "diagnosis_correct_rate" not in metrics
	assert "action_selected_rate" not in metrics
	assert "resolution_correctness" in result


def test_report_diagnosis_metrics_exclude_symptoms_from_precision() -> None:
	ground_truth = [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"injected_at": 1.0,
			"evaluable": True,
		}
	]
	reports = [
		{"diagnosis_id": "sensor:M1:Power:stuck", "time": 2.0},
		{"diagnosis_id": "temperature:M1:overheating", "time": 2.0},
		{"diagnosis_id": "machine:M1:production_slowdown", "time": 2.0},
	]

	result = score_report_diagnoses(reports, ground_truth)
	metrics = cast(dict[str, object], result["metrics"])

	assert metrics["diagnosis_precision"] == 1.0
	assert "diagnosis_recall" not in metrics
	assert metrics["mean_detection_latency"] == 1.0


def test_report_diagnosis_metrics_exclude_derived_issues_from_targets() -> None:
	ground_truth = [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"injected_at": 1.0,
			"source": "root_fault",
			"evaluable": True,
		},
		{
			"diagnosis": "machine:M1:production_blocked",
			"source": "derived_issue",
			"evaluable": True,
		},
	]
	reports = [
		{"diagnosis_id": "sensor:M1:Power:stuck", "time": 2.0},
		{"diagnosis_id": "machine:M1:production_blocked", "time": 5.0},
	]

	result = score_report_diagnoses(reports, ground_truth)
	metrics = cast(dict[str, object], result["metrics"])
	per_fault = cast(list[dict[str, object]], result["per_fault"])

	assert metrics["evaluable_faults"] == 1
	assert metrics["mean_detection_latency"] == 1.0
	assert [fault["diagnosis"] for fault in per_fault] == ["sensor:M1:Power:stuck"]


def test_chain_effect_actions_are_tolerated_under_chain_effect_diagnoses() -> None:
	ground_truth = [
		{
			"diagnosis": "sensor:M1:Temperature:stuck",
			"required_action": "fix_stuck",
			"chain_effects": ["temperature:M1:overheating", "machine:M1:production_slowdown"],
			"evaluable": True,
		}
	]
	actions = [
		{
			"diagnosis_id": "sensor:M1:Temperature:stuck",
			"recommended_action": "fix_stuck",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
		{
			"diagnosis_id": "temperature:M1:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
	]

	result = score_agent_decisions(actions, [], ground_truth)

	assert result["unexpected_action_pairs"] == []
	assert result["metrics"]["per_fault_action_selected_rate"] == 1.0


def test_operational_state_responses_are_not_unexpected() -> None:
	ground_truth = [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"required_action": "fix_stuck",
			"evaluable": True,
		}
	]
	actions = [
		{
			"diagnosis_id": "sensor:M1:Power:stuck",
			"recommended_action": "fix_stuck",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
		# Housekeeping on machines without any injected fault.
		{
			"diagnosis_id": "temperature:M2:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
		{
			"diagnosis_id": "battery:M3:low_battery",
			"recommended_action": "replace_battery",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
		# A non-canonical response to a physical state stays unexpected.
		{
			"diagnosis_id": "temperature:M2:overheating",
			"recommended_action": "fix_packet_loss",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
	]

	result = score_agent_decisions(actions, [], ground_truth)

	assert result["unexpected_action_pairs"] == [
		{"diagnosis": "temperature:M2:overheating", "action": "fix_packet_loss"}
	]
	assert result["unexpected_actions"] == ["fix_packet_loss"]
	assert result["metrics"]["per_fault_action_selected_rate"] == 1.0


def test_recalibration_completes_stuck_actuator_action_pair() -> None:
	ground_truth = [
		{
			"diagnosis": "actuator:M1:stuck",
			"required_action": "fix_stuck",
			"evaluable": True,
		}
	]
	actions = [
		{
			"diagnosis_id": "actuator:M1:stuck",
			"recommended_action": "fix_stuck",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
	]

	result = score_agent_decisions(actions, [], ground_truth)

	assert result["metrics"]["per_fault_action_selected_rate"] == 1.0
	assert result["missing_action_pairs"] == []
	assert result["metrics"]["required_action_execution_rate"] == 1.0


def test_duplicate_diagnosis_roles_remain_distinguishable_in_per_fault_metrics() -> None:
	ground_truth = [
		{
			"truth_id": "root_fault:sensor:M1:Power:stuck",
			"diagnosis": "sensor:M1:Power:stuck",
			"required_action": "fix_stuck",
			"source": "root_fault",
			"evaluation_role": "root_fault",
			"evaluable": True,
		},
		{
			"truth_id": "audit:sensor:M1:Power:stuck",
			"diagnosis": "sensor:M1:Power:stuck",
			"required_action": "fix_stuck",
			"source": "physical_state",
			"evaluation_role": "physical_state_response",
			"evaluable": True,
		},
	]
	actions = [
		{
			"diagnosis_id": "sensor:M1:Power:stuck",
			"recommended_action": "fix_stuck",
			"execution_succeeded": True,
			"execution_attempted": True,
		},
	]

	result = score_agent_decisions(actions, [], ground_truth)
	per_fault = cast(list[dict[str, object]], result["per_fault"])

	assert [record["truth_id"] for record in per_fault] == [
		"root_fault:sensor:M1:Power:stuck",
		"audit:sensor:M1:Power:stuck",
	]
	assert [record["evaluation_role"] for record in per_fault] == ["root_fault", "physical_state_response"]
	assert [record["diagnosis"] for record in per_fault] == ["sensor:M1:Power:stuck", "sensor:M1:Power:stuck"]
	assert result["metrics"]["evaluable_faults"] == 2


def test_already_resolved_counts_as_correct_decision_not_execution() -> None:
	ground_truth = [
		{
			"diagnosis": "temperature:M1:overheating",
			"required_action": "start_cooling",
			"source": "physical_state",
			"evaluable": True,
		}
	]
	actions = [
		{
			"diagnosis_id": "temperature:M1:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": False,
			"execution_attempted": True,
			"execution_outcome": "already_resolved",
			"failure_reason": None,
		}
	]

	result = score_agent_decisions(actions, [], ground_truth)

	assert result["per_fault"][0]["action_correct"] is True
	assert result["per_fault"][0]["action_executed"] is False
	assert result["unexpected_action_pairs"] == []
	assert result["metrics"]["per_fault_action_selected_rate"] == 1.0
	assert result["metrics"]["fault_action_execution_rate"] == 0.0
	assert result["metrics"]["action_attempt_success_rate"] is None


def test_already_resolved_actions_are_excluded_from_action_attempt_success_rate() -> None:
	ground_truth = [
		{
			"diagnosis": "temperature:M1:overheating",
			"required_action": "start_cooling",
			"source": "physical_state",
			"evaluable": True,
		}
	]
	actions = [
		{
			"diagnosis_id": "temperature:M1:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": True,
			"execution_attempted": True,
			"execution_outcome": "succeeded",
			"failure_reason": None,
		},
		{
			"diagnosis_id": "temperature:M1:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": False,
			"execution_attempted": True,
			"execution_outcome": "failed",
			"failure_reason": "execution_failed",
		},
		{
			"diagnosis_id": "temperature:M1:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": False,
			"execution_attempted": True,
			"execution_outcome": "already_resolved",
			"failure_reason": None,
		},
	]

	result = score_agent_decisions(actions, [], ground_truth)

	assert result["metrics"]["action_attempt_success_rate"] == 0.5


def test_legacy_obsolete_actions_are_excluded_from_action_attempt_success_rate() -> None:
	ground_truth = [
		{
			"diagnosis": "temperature:M1:overheating",
			"required_action": "start_cooling",
			"source": "physical_state",
			"evaluable": True,
		}
	]
	actions = [
		{
			"diagnosis_id": "temperature:M1:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": True,
			"execution_attempted": True,
			"execution_outcome": "succeeded",
			"failure_reason": None,
		},
		{
			"diagnosis_id": "temperature:M1:overheating",
			"recommended_action": "start_cooling",
			"execution_succeeded": False,
			"execution_attempted": True,
			"execution_outcome": "obsolete",
			"failure_reason": "condition_already_resolved",
		},
	]

	result = score_agent_decisions(actions, [], ground_truth)

	assert result["metrics"]["action_attempt_success_rate"] == 1.0


def test_resolution_correctness_is_reported_by_conflict_type() -> None:
	ground_truth = [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"required_action": "fix_stuck",
			"evaluable": True,
		}
	]
	decisions = [
		{
			"decision_id": "d1",
			"conflict_id": "c1",
			"selected_diagnosis_id": "sensor:M1:Power:stuck",
			"selected_action": "fix_stuck",
			"confidence": "high",
		}
	]
	conflicts = [
		{
			"conflict_id": "c1",
			"conflict_types": ["action"],
		}
	]

	result = score_agent_decisions([], decisions, ground_truth, detected_conflicts=conflicts)

	assert result["resolution_correctness"]["overall"]["decisions"] == 1
	assert result["resolution_correctness"]["overall"]["diagnosis_accuracy"] == 1.0
	assert result["resolution_correctness"]["overall"]["action_accuracy"] == 1.0
	assert result["resolution_correctness"]["by_conflict_type"]["action"]["action_accuracy"] == 1.0
	assert result["resolution_correctness"]["by_conflict_type"]["confidence"]["decisions"] == 0


	ground_truth = [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"required_action": "fix_stuck",
			"evaluable": True,
		},
		{
			"diagnosis": "temperature:M1:overheating",
			"required_action": "start_cooling",
			"evaluable": True,
		},
		{
			"diagnosis": "battery:M1:low_battery",
			"required_action": "replace_battery",
			"evaluable": True,
		},
	]
	decisions = [
		{
			"decision_id": "d1",
			"selected_diagnosis_id": "sensor:M1:Power:stuck",
			"selected_action": "fix_stuck",
		},
		{
			"decision_id": "d2",
			"selected_diagnosis_id": "temperature:M1:overheating",
			"selected_action": "start_cooling",
		},
		{
			"decision_id": "d3",
			"selected_diagnosis_id": "battery:M1:low_battery",
			"selected_action": "replace_battery",
		},
	]

	result = score_agent_decisions([], decisions, ground_truth)
	resolution = cast(dict[str, object], result["resolution_correctness"])
	overall = cast(dict[str, object], resolution["overall"])
	assert set(overall) == {"decisions", "diagnosis_accuracy", "action_accuracy"}


def test_derived_issue_decision_with_correct_root_action_is_correct() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_belt_slippage",
			}
		],
		_derived_issue_ground_truth(),
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is True
	assert decision["selected_diagnosis_matches_truth"] is True
	assert decision["matched_symptom_diagnosis"] == "belt:M1:M2:transfer_rate_degraded"
	assert decision["derived_issue_attribution_status"] == "attributed_to_root"
	assert decision["derived_issue_attributed_root_fault"] == "belt:M1:M2:belt_slippage"
	assert decision["expected_diagnosis"] == "belt:M1:M2:belt_slippage"
	assert decision["expected_root_diagnosis"] == "belt:M1:M2:belt_slippage"
	assert decision["expected_action"] == "fix_belt_slippage"
	assert decision["expected_root_fault"] == "belt:M1:M2:belt_slippage"


def test_derived_issue_root_action_counts_in_per_fault_metrics() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_belt_slippage",
			}
		],
		_derived_issue_ground_truth(),
		events=_derived_issue_events(),
	)
	metrics = cast(dict[str, object], result["metrics"])
	per_fault = cast(list[dict[str, object]], result["per_fault"])

	assert metrics["per_fault_diagnosis_recall"] == 1.0
	assert metrics["per_fault_action_selected_rate"] == 1.0
	assert result["missing_diagnoses"] == []
	assert per_fault[0]["diagnosis_correct"] is True
	assert per_fault[0]["action_correct"] is True
	assert per_fault[0]["selected_actions"] == ["fix_belt_slippage"]


def test_derived_issue_root_action_is_not_reported_missing_or_unexpected() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"conflict_id": "c1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_belt_slippage",
			}
		],
		_derived_issue_ground_truth(),
		detected_conflicts=[_conflict("c1", 2.0, 3.0)],
		events=_derived_issue_events(),
	)

	assert result["missing_action_pairs"] == []
	assert result["unexpected_action_pairs"] == []
	assert result["missing_actions"] == []
	assert result["unexpected_actions"] == []


def test_derived_issue_passive_action_is_correct_when_root_is_handled() -> None:
	result = score_agent_decisions(
		[
			{
				"diagnosis_id": "belt:M1:M2:belt_slippage",
				"recommended_action": "fix_belt_slippage",
				"execution_succeeded": True,
				"time": 2.5,
			}
		],
		[
			{
				"decision_id": "d1",
				"conflict_id": "c1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "wait_for_more_evidence",
			}
		],
		_derived_issue_ground_truth(),
		detected_conflicts=[_conflict("c1", 2.0, 3.0)],
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is True
	assert decision["root_action_already_handled"] is True
	assert decision["passive_action_credit_reason"] == "root_action_already_handled"


def test_derived_issue_passive_action_is_wrong_when_root_action_is_outside_conflict_window() -> None:
	result = score_agent_decisions(
		[
			{
				"diagnosis_id": "belt:M1:M2:belt_slippage",
				"recommended_action": "fix_belt_slippage",
				"execution_succeeded": True,
				"time": 5.0,
			}
		],
		[
			{
				"decision_id": "d1",
				"conflict_id": "c1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "wait_for_more_evidence",
			}
		],
		_derived_issue_ground_truth(),
		detected_conflicts=[_conflict("c1", 2.0, 3.0)],
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is False


def test_derived_issue_passive_action_is_wrong_when_root_decision_is_later() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"conflict_id": "c1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "wait_for_more_evidence",
			},
			{
				"decision_id": "d2",
				"conflict_id": "c2",
				"selected_diagnosis_id": "belt:M1:M2:belt_slippage",
				"selected_action": "fix_belt_slippage",
			},
		],
		_derived_issue_ground_truth(),
		detected_conflicts=[_conflict("c1", 2.0, 3.0), _conflict("c2", 5.0, 6.0)],
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is False


def test_derived_issue_passive_action_allows_same_window_root_decision() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"conflict_id": "c1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "wait_for_more_evidence",
			},
			{
				"decision_id": "d2",
				"conflict_id": "c2",
				"selected_diagnosis_id": "belt:M1:M2:belt_slippage",
				"selected_action": "fix_belt_slippage",
			},
		],
		_derived_issue_ground_truth(),
		detected_conflicts=[_conflict("c1", 2.0, 3.0), _conflict("c2", 3.0, 3.0)],
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is True


def test_derived_issue_passive_action_is_wrong_when_root_is_unhandled() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "wait_for_more_evidence",
			}
		],
		_derived_issue_ground_truth(),
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is False


def test_derived_issue_wrong_direct_action_is_wrong() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_belt_jam",
			}
		],
		_derived_issue_ground_truth(),
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is False


	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_belt_slippage",
			}
		],
		_derived_issue_ground_truth(),
		events=_derived_issue_events(),
	)

	decision = _first_resolution_decision(result)


def test_unattributable_derived_issue_does_not_credit_action() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_belt_slippage",
			}
		],
		_derived_issue_ground_truth(),
		events=[
			ReportedEvent(
				identifier="belt:M1:M2:transfer_rate_degraded",
				kind="derived_issue",
				component="Flow",
				cause_id="missing-cause",
				context={"time": 2.0},
			)
		],
	)

	decision = _first_resolution_decision(result)
	assert decision["diagnosis_correct"] is False
	assert decision["selected_diagnosis_in_ground_truth"] is True
	assert decision["selected_diagnosis_matches_truth"] is True
	assert decision["matched_symptom_diagnosis"] == "belt:M1:M2:transfer_rate_degraded"
	assert decision["derived_issue_attribution_status"] == "no_root_cause_found"
	assert decision["derived_issue_attributed_root_fault"] is None
	assert decision["expected_diagnosis"] is None
	assert decision["expected_root_fault"] is None
	assert decision["action_correct"] is False


def test_missing_derived_issue_event_reports_no_matching_issue_event() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_belt_slippage",
			}
		],
		_derived_issue_ground_truth(),
		events=[
			ReportedEvent(
				identifier="belt:M1:M2:belt_slippage",
				kind="root_fault",
				component="FaultInjector",
				context={"time": 1.0},
			)
		],
	)

	decision = _first_resolution_decision(result)
	assert decision["selected_diagnosis_matches_truth"] is True
	assert decision["derived_issue_attribution_status"] == "no_matching_issue_event"
	assert decision["derived_issue_matched_event_time"] is None
	assert decision["derived_issue_attributed_root_fault"] is None
	assert decision["expected_diagnosis"] is None
	assert decision["diagnosis_correct"] is False
	assert decision["action_correct"] is False


def test_repeated_derived_issue_uses_decision_window_root_attribution() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"conflict_id": "c1",
				"selected_diagnosis_id": "machine:M2:production_blocked",
				"selected_action": "fix_jammed_workpiece",
			}
		],
		[
			{
				"root_fault": "belt:M1:M2:belt_jam",
				"diagnosis": "belt:M1:M2:belt_jam",
				"required_action": "fix_belt_jam",
				"source": "root_fault",
				"evaluable": True,
			},
			{
				"root_fault": "machine:M2:jammed_workpiece",
				"diagnosis": "machine:M2:jammed_workpiece",
				"required_action": "fix_jammed_workpiece",
				"source": "root_fault",
				"evaluable": True,
			},
			{
				"diagnosis": "machine:M2:production_blocked",
				"required_action": "",
				"source": "derived_issue",
				"evaluable": True,
			},
		],
		detected_conflicts=[_conflict("c1", 10.0, 11.0)],
		events=[
			ReportedEvent(
				identifier="belt:M1:M2:belt_jam",
				kind="root_fault",
				component="FaultInjector",
				context={"time": 1.0},
			),
			ReportedEvent(
				identifier="machine:M2:production_blocked",
				kind="derived_issue",
				component="Flow",
				cause_id="belt:M1:M2:belt_jam",
				context={"time": 2.0},
			),
			ReportedEvent(
				identifier="machine:M2:jammed_workpiece",
				kind="root_fault",
				component="FaultInjector",
				context={"time": 9.0},
			),
			ReportedEvent(
				identifier="machine:M2:production_blocked",
				kind="derived_issue",
				component="Flow",
				cause_id="machine:M2:jammed_workpiece",
				context={"time": 10.5},
			),
		],
	)

	decision = _first_resolution_decision(result)
	assert decision["conflict_window"] == {"start": 10.0, "end": 11.0}
	assert decision["derived_issue_matched_event_time"] == 10.5
	assert decision["derived_issue_attribution_status"] == "attributed_to_root"
	assert decision["expected_root_fault"] == "machine:M2:jammed_workpiece"
	assert decision["expected_action"] == "fix_jammed_workpiece"
	assert decision["action_correct"] is True


def test_network_observation_cause_attributes_derived_issue_to_network_root_fault() -> None:
	result = score_agent_decisions(
		[],
		[
			{
				"decision_id": "d1",
				"selected_diagnosis_id": "belt:M1:M2:transfer_rate_degraded",
				"selected_action": "fix_latency",
			}
		],
		[
			{
				"root_fault": "network:latency",
				"diagnosis": "network:latency",
				"required_action": "fix_latency",
				"source": "root_fault",
				"evaluable": True,
			},
			{
				"diagnosis": "belt:M1:M2:transfer_rate_degraded",
				"required_action": "",
				"source": "derived_issue",
				"evaluable": True,
			},
		],
		events=[
			ReportedEvent(
				identifier="network:latency",
				kind="root_fault",
				component="Network",
				context={"time": 1.0},
			),
			ReportedEvent(
				identifier="network:network_latency_detected",
				kind="observation",
				component="Network",
				context={"time": 2.0},
			),
			ReportedEvent(
				identifier="belt:M1:M2:transfer_rate_degraded",
				kind="derived_issue",
				component="Flow",
				cause_id="network:network_latency_detected",
				context={"time": 2.5},
			),
		],
	)

	decision = _first_resolution_decision(result)
	assert decision["expected_root_fault"] == "network:latency"
	assert decision["expected_action"] == "fix_latency"
	assert decision["diagnosis_correct"] is True
	assert decision["action_correct"] is True


def _derived_issue_ground_truth() -> list[dict[str, object]]:
	return [
		{
			"root_fault": "belt:M1:M2:belt_slippage",
			"diagnosis": "belt:M1:M2:belt_slippage",
			"required_action": "fix_belt_slippage",
			"source": "root_fault",
			"evaluable": True,
		},
		{
			"diagnosis": "belt:M1:M2:transfer_rate_degraded",
			"required_action": "",
			"source": "derived_issue",
			"evaluable": True,
		},
	]


def _first_resolution_decision(result: dict[str, object]) -> dict[str, object]:
	resolution = cast(dict[str, object], result["resolution_correctness"])
	per_decision = cast(list[dict[str, object]], resolution["per_decision"])
	return per_decision[0]


def _conflict(conflict_id: str, start: float, end: float) -> dict[str, object]:
	return {
		"conflict_id": conflict_id,
		"window": {"start": start, "end": end},
		"conflict_types": ["action"],
	}


def _derived_issue_events() -> list[ReportedEvent]:
	return [
		ReportedEvent(
			identifier="belt:M1:M2:belt_slippage",
			kind="root_fault",
			component="FaultInjector",
			context={"time": 1.0},
		),
		ReportedEvent(
			identifier="belt:M1:M2:persistent_queue_pressure",
			kind="derived_issue",
			component="Flow",
			cause_id="belt:M1:M2:belt_slippage",
			context={"time": 1.5},
		),
		ReportedEvent(
			identifier="belt:M1:M2:transfer_rate_degraded",
			kind="derived_issue",
			component="Flow",
			cause_id="belt:M1:M2:persistent_queue_pressure",
			context={"time": 2.0},
		),
	]
