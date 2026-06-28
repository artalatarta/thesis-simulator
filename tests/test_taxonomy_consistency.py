import pandas as pd

from analysis.run_loader import decision_failure_taxonomy


def test_taxonomy_correct_matches_scorer_correctness_flags() -> None:
	decisions = pd.DataFrame(
		[
			{"diagnosis_correct": True, "action_correct": True},
			{"diagnosis_correct": False, "action_correct": True},
			{"diagnosis_correct": True, "action_correct": False},
		]
	)
	correct_rows = decisions["diagnosis_correct"] & decisions["action_correct"]

	taxonomy = decision_failure_taxonomy(decisions)
	correct_count = int(taxonomy.loc[taxonomy["failure_mode"].eq("active_action_correct"), "decisions"].sum())

	assert correct_count == int(correct_rows.sum())


def test_taxonomy_never_labels_incorrect_scorer_flags_as_correct() -> None:
	decisions = pd.DataFrame(
		[
			{"diagnosis_correct": False, "action_correct": True},
			{"diagnosis_correct": True, "action_correct": False},
		]
	)

	taxonomy = decision_failure_taxonomy(decisions)

	assert "active_action_correct" not in set(taxonomy["failure_mode"])
	assert "passive_after_root_handled" not in set(taxonomy["failure_mode"])


def test_taxonomy_splits_active_correct_from_credited_passive() -> None:
	decisions = pd.DataFrame(
		[
			{
				"diagnosis_correct": True,
				"action_correct": True,
				"selected_action": "fix_stuck",
			},
			{
				"diagnosis_correct": True,
				"action_correct": True,
				"selected_action": "wait_for_more_evidence",
				"root_action_already_handled": True,
			},
		]
	)

	taxonomy = decision_failure_taxonomy(decisions)

	assert dict(zip(taxonomy["failure_mode"], taxonomy["decisions"], strict=True)) == {
		"active_action_correct": 1,
		"passive_after_root_handled": 1,
	}


def test_taxonomy_uses_selected_diagnosis_match_as_legacy_fallback_only() -> None:
	legacy_decisions = pd.DataFrame(
		[
			{"selected_diagnosis_matches_truth": False, "action_correct": True},
		]
	)
	scored_decisions = pd.DataFrame(
		[
			{
				"diagnosis_correct": True,
				"selected_diagnosis_matches_truth": False,
				"action_correct": True,
			},
		]
	)

	legacy_taxonomy = decision_failure_taxonomy(legacy_decisions)
	scored_taxonomy = decision_failure_taxonomy(scored_decisions)

	assert dict(zip(legacy_taxonomy["failure_mode"], legacy_taxonomy["decisions"], strict=True)) == {"wrong_diagnosis": 1}
	assert dict(zip(scored_taxonomy["failure_mode"], scored_taxonomy["decisions"], strict=True)) == {"active_action_correct": 1}


def test_taxonomy_uses_missing_truth_as_legacy_fallback_only() -> None:
	legacy_decisions = pd.DataFrame(
		[
			{"selected_diagnosis_in_ground_truth": False, "matched_truth_source": None},
		]
	)
	scored_decisions = pd.DataFrame(
		[
			{
				"diagnosis_correct": True,
				"action_correct": True,
				"selected_diagnosis_in_ground_truth": False,
				"matched_truth_source": None,
			},
		]
	)

	legacy_taxonomy = decision_failure_taxonomy(legacy_decisions)
	scored_taxonomy = decision_failure_taxonomy(scored_decisions)

	assert dict(zip(legacy_taxonomy["failure_mode"], legacy_taxonomy["decisions"], strict=True)) == {"missing_truth": 1}
	assert dict(zip(scored_taxonomy["failure_mode"], scored_taxonomy["decisions"], strict=True)) == {"active_action_correct": 1}
