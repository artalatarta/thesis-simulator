import csv
import json
from typing import Any, cast

from cps.agents.diagnosis import resolver_visible_cause_id
from cps.agents.identifiers import (
	BATTERY_OBSERVATIONS,
	BELT_PRODUCTION_ISSUES,
	MACHINE_PRODUCTION_ISSUES,
	TEMPERATURE_OBSERVATIONS,
)
from cps.core.reporting import EventReporter, ReportedEvent
from cps.evaluation.event_records import monitoring_report_from_event
from cps.evaluation.ground_truth import (
	GROUND_TRUTH_CSV,
	duplicate_diagnosis_groups,
	physical_state_response,
)
from cps.evaluation.scenarios import (
	build_experiment_record as _build_experiment_record,
	ground_truth_for_root_faults,
	physical_state_ground_truth,
	write_jsonl,
)
from tests.fakes import RuleBasedDetector
from tests.ground_truth_catalog import catalog_rows_by_category, validate_ground_truth_catalog

reporter = EventReporter()


def build_experiment_record(**kwargs: Any):
	return _build_experiment_record(events=reporter.events, **kwargs)


def test_build_experiment_record_scores_actual_agent_actions_against_injected_faults() -> None:
	reporter.clear()
	reporter.root_fault("sensor:Sheet-Metal-Press:Power:stuck", context={"time": 1.0})
	reporter.observation(
		"sensor:Sheet-Metal-Press:Power:sensor_stuck_detected",
		component="Sheet-Metal-Press",
		context={"time": 2.0},
	)
	reporter.fault_resolved("sensor:Sheet-Metal-Press:Power:stuck", component="Sheet-Metal-Press", context={"time": 6.0})

	record = build_experiment_record(
		window_start=0.0,
		window_end=10.0,
		detector=RuleBasedDetector(),
		runtime_llm_decisions=[],
		runtime_llm_reports=[
			{
				"report_id": "PowerSensor-Sheet-Metal-Press-1",
				"diagnosis_id": "sensor:Sheet-Metal-Press:Power:stuck",
				"time": 2.0,
			}
		],
		agent_actions=[
			{
				"diagnosis_id": "sensor:Sheet-Metal-Press:Power:stuck",
				"recommended_action": "fix_stuck",
				"execution_attempted": True,
				"execution_succeeded": True,
			}
		],
	)

	assert record.injected_root_faults == ["sensor:Sheet-Metal-Press:Power:stuck"]
	assert record.runtime_llm_reports == [
		{
			"report_id": "PowerSensor-Sheet-Metal-Press-1",
			"diagnosis_id": "sensor:Sheet-Metal-Press:Power:stuck",
			"time": 2.0,
		}
	]
	assert record.ground_truth == [
		{
			"truth_id": "root_fault:sensor:Sheet-Metal-Press:Power:stuck",
			"root_fault": "sensor:Sheet-Metal-Press:Power:stuck",
			"diagnosis": "sensor:Sheet-Metal-Press:Power:stuck",
			"required_action": "fix_stuck",
			"chain_effects": [
				"battery:Sheet-Metal-Press:low_battery",
				"battery:Sheet-Metal-Press:dead_battery",
				"machine:Sheet-Metal-Press:production_blocked",
				"belt:<from_node_id>:Sheet-Metal-Press:handoff_blocked",
				"machine:<from_node_id>:production_blocked",
				"belt:<from_node_id>:Sheet-Metal-Press:persistent_queue_pressure",
			],
			"source": "root_fault",
			"evaluation_role": "root_fault",
			"injected_at": 1.0,
			"available_observation_time": 9.0,
			"evaluable": True,
		}
	]
	assert record.runtime_correctness == {
		"missing_diagnoses": [],
		"missing_actions": [],
		"unexpected_actions": [],
		"missing_action_pairs": [],
		"unexpected_action_pairs": [],
		"per_fault": [
			{
				"truth_id": "root_fault:sensor:Sheet-Metal-Press:Power:stuck",
				"evaluation_role": "root_fault",
				"diagnosis": "sensor:Sheet-Metal-Press:Power:stuck",
				"injected_at": 1.0,
				"evaluable": True,
				"detected": True,
				"first_report_at": 2.0,
				"detection_latency": 1.0,
				"source": "root_fault",
				"required_action": "fix_stuck",
				"required_actions": ["fix_stuck"],
				"selected_actions": ["fix_stuck"],
				"executed_actions": ["fix_stuck"],
				"diagnosis_correct": True,
				"action_correct": True,
				"action_selected": True,
				"action_executed": True,
			}
		],
		"resolution_correctness": {
			"per_decision": [],
			"overall": {
				"decisions": 0,
				"diagnosis_accuracy": None,
				"action_accuracy": None,
			},
			"by_conflict_type": {
				"diagnosis": {
					"decisions": 0,
					"diagnosis_accuracy": None,
					"action_accuracy": None,
				},
				"action": {
					"decisions": 0,
					"diagnosis_accuracy": None,
					"action_accuracy": None,
					},
					"confidence": {
					"decisions": 0,
					"diagnosis_accuracy": None,
					"action_accuracy": None,
				},
			},
		},
		"metrics": {
			"evaluable_faults": 1,
			"root_faults": 1,
			"physical_state_faults": 0,
			"diagnosis_precision": 1.0,
			"per_fault_diagnosis_recall": 1.0,
			"per_fault_action_selected_rate": 1.0,
			"root_per_fault_diagnosis_recall": 1.0,
			"root_per_fault_action_selected_rate": 1.0,
			"physical_state_per_fault_diagnosis_recall": None,
			"physical_state_per_fault_action_selected_rate": None,
			"fault_action_execution_rate": 1.0,
			"required_action_execution_rate": 1.0,
			"action_attempt_success_rate": 1.0,
			"mean_detection_latency": 1.0,
			"median_detection_latency": 1.0,
		},
	}
	detection_metrics = cast(dict[str, object], record.detection_metrics["metrics"])
	assert "diagnosis_recall" not in detection_metrics
	assert record.runtime_detection["metrics"]["mean_detection_latency"] == 1.0
	assert record.per_fault_outcomes["metrics"]["per_fault_diagnosis_recall"] == 1.0
	assert record.resolver_correctness == record.runtime_correctness["resolution_correctness"]
	assert record.cascade == {
		"per_fault": [
			{
				"root_fault": "sensor:Sheet-Metal-Press:Power:stuck",
				"injected_at": 1.0,
				"resolved_at": 6.0,
				"first_effect_at": None,
				"manifestation_latency": None,
				"containable": True,
				"polls_to_resolution": 2,
				"depth": 0,
				"contained": True,
				"reached_effects": [],
			}
		],
		"metrics": {
			"root_faults": 1,
			"n_contained": 1,
			"n_cascaded": 0,
			"n_containable": 1,
			"n_structurally_cascading": 0,
			"n_contained_given_containable": 1,
			"contained_rate_over_containable": 1.0,
		},
	}


def test_build_experiment_record_preserves_conflict_report_citations_in_runtime_reports() -> None:
	reporter.clear()
	runtime_conflicts = [
		{
			"conflict_id": "conflict-M1-t1-1",
			"machine_id": "M1",
			"window": {"start": 1.0, "end": 2.0},
			"conflict_types": ["diagnosis", "action"],
			"report_ids": ["flow-t2", "sensor-t2"],
			"diagnoses": ["machine:M1:production_blocked", "sensor:M1:Power:stuck"],
			"actions": ["wait_for_more_evidence", "fix_stuck"],
			"evidence": ["flow-t2", "sensor-t2"],
			"description": "test conflict",
		}
	]
	runtime_llm_reports = [
		{
			"report_id": "flow-t2",
			"diagnosis_id": "machine:M1:production_blocked",
			"recommended_action": "wait_for_more_evidence",
			"time": 2.0,
		},
		{
			"report_id": "sensor-t2",
			"diagnosis_id": "sensor:M1:Power:stuck",
			"recommended_action": "fix_stuck",
			"time": 2.0,
		},
	]

	record = build_experiment_record(
		window_start=0.0,
		window_end=3.0,
		runtime_conflicts=runtime_conflicts,
		runtime_llm_decisions=[],
		runtime_llm_reports=runtime_llm_reports,
		agent_actions=[],
	)

	conflict_report_ids = {report_id for conflict in record.detected_conflicts for report_id in conflict["report_ids"]}
	runtime_report_ids = {report["report_id"] for report in record.runtime_llm_reports}
	assert conflict_report_ids <= runtime_report_ids


def test_build_experiment_record_separates_detection_metrics_and_runtime_detection_latency() -> None:
	reporter.clear()
	reporter.root_fault("sensor:Sheet-Metal-Press:Power:stuck", context={"time": 1.0})
	reporter.observation(
		"sensor:Sheet-Metal-Press:Power:sensor_stuck_detected",
		component="Sheet-Metal-Press",
		context={"time": 2.0},
	)

	record = build_experiment_record(
		window_start=0.0,
		window_end=10.0,
		detector=RuleBasedDetector(),
		runtime_llm_decisions=[],
		runtime_llm_reports=[
			{
				"report_id": "runtime-late",
				"diagnosis_id": "sensor:Sheet-Metal-Press:Power:stuck",
				"time": 5.0,
			}
		],
		agent_actions=[],
	)

	detection_metrics = cast(dict[str, object], record.detection_metrics["metrics"])
	runtime_metrics = cast(dict[str, object], record.runtime_detection["metrics"])

	assert detection_metrics["mean_detection_latency"] == 1.0
	assert runtime_metrics["mean_detection_latency"] == 4.0
	assert record.runtime_correctness["metrics"]["mean_detection_latency"] == 4.0


def test_ground_truth_for_random_root_faults_comes_from_fault_catalog() -> None:
	ground_truth = ground_truth_for_root_faults(
		[
			"sensor:Sheet-Metal-Press:Power:stuck",
			"actuator:Body-Welding-Cell:stuck",
			"network:packet_loss",
		]
	)

	assert ground_truth == [
		{
			"truth_id": "root_fault:sensor:Sheet-Metal-Press:Power:stuck",
			"root_fault": "sensor:Sheet-Metal-Press:Power:stuck",
			"diagnosis": "sensor:Sheet-Metal-Press:Power:stuck",
			"required_action": "fix_stuck",
			"chain_effects": [
				"battery:Sheet-Metal-Press:low_battery",
				"battery:Sheet-Metal-Press:dead_battery",
				"machine:Sheet-Metal-Press:production_blocked",
				"belt:<from_node_id>:Sheet-Metal-Press:handoff_blocked",
				"machine:<from_node_id>:production_blocked",
				"belt:<from_node_id>:Sheet-Metal-Press:persistent_queue_pressure",
			],
			"source": "root_fault",
			"evaluation_role": "root_fault",
		},
		{
			"truth_id": "root_fault:actuator:Body-Welding-Cell:stuck",
			"root_fault": "actuator:Body-Welding-Cell:stuck",
			"diagnosis": "actuator:Body-Welding-Cell:stuck",
			"required_action": "fix_stuck",
			"chain_effects": [
				"machine:Body-Welding-Cell:production_blocked",
				"belt:<from_node_id>:Body-Welding-Cell:handoff_blocked",
				"machine:<from_node_id>:production_blocked",
				"belt:<from_node_id>:Body-Welding-Cell:persistent_queue_pressure",
			],
			"source": "root_fault",
			"evaluation_role": "root_fault",
		},
		{
			"truth_id": "root_fault:network:packet_loss",
			"root_fault": "network:packet_loss",
			"diagnosis": "network:packet_loss",
			"required_action": "fix_packet_loss",
			"chain_effects": [
				"belt:<from_node_id>:<to_node_id>:handoff_blocked",
				"belt:<from_node_id>:<to_node_id>:transfer_rate_degraded",
				"machine:<from_node_id>:production_blocked",
				"belt:<from_node_id>:<to_node_id>:persistent_queue_pressure",
			],
			"source": "root_fault",
			"evaluation_role": "root_fault",
		},
	]


def test_ground_truth_for_belt_root_fault_names_the_concrete_belt() -> None:
	ground_truth = ground_truth_for_root_faults(["belt:Sheet-Metal-Press:Body-Welding-Cell:belt_jam"])

	assert ground_truth == [
		{
			"truth_id": "root_fault:belt:Sheet-Metal-Press:Body-Welding-Cell:belt_jam",
			"root_fault": "belt:Sheet-Metal-Press:Body-Welding-Cell:belt_jam",
			"diagnosis": "belt:Sheet-Metal-Press:Body-Welding-Cell:belt_jam",
			"required_action": "fix_belt_jam",
			"chain_effects": [
				"belt:Sheet-Metal-Press:Body-Welding-Cell:handoff_blocked",
				"machine:Sheet-Metal-Press:production_blocked",
				"belt:Sheet-Metal-Press:Body-Welding-Cell:persistent_queue_pressure",
			],
			"source": "root_fault",
			"evaluation_role": "root_fault",
		}
	]


def test_ground_truth_catalog_validates_against_fault_taxonomy() -> None:
	assert validate_ground_truth_catalog() == []


def test_ground_truth_csv_has_one_required_action_per_root_fault() -> None:
	with GROUND_TRUTH_CSV.open(newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))

	assert "required_action" in rows[0]
	assert "required_actions" not in rows[0]
	root_fault_rows = [row for row in rows if row["category"] == "root_fault"]
	assert root_fault_rows
	assert all(row["required_action"] and "|" not in row["required_action"] for row in root_fault_rows)
	assert all("reboot_machine_process" not in str(row) for row in rows)


def test_ground_truth_csv_has_physical_state_and_derived_issue_severities() -> None:
	physical_states = catalog_rows_by_category("physical_state")
	derived_issues = catalog_rows_by_category("derived_issue")

	expected_physical_states = {
		("battery", observation.removesuffix("_detected")) for observation in BATTERY_OBSERVATIONS
	} | {
		("temperature", observation.removesuffix("_detected")) for observation in TEMPERATURE_OBSERVATIONS
	}
	expected_derived_issues = {("machine", issue) for issue in MACHINE_PRODUCTION_ISSUES} | {
		("belt", issue) for issue in BELT_PRODUCTION_ISSUES
	}

	assert {(row["domain"], row["fault"]) for row in physical_states} == expected_physical_states
	assert {(row["domain"], row["fault"]) for row in derived_issues} == expected_derived_issues
	assert all(row["required_action"] for row in physical_states)
	assert all(not row["required_action"] for row in derived_issues)
	overheating_response = physical_state_response("temperature", "overheating")
	low_battery_response = physical_state_response("battery", "low_battery")
	assert overheating_response is not None
	assert low_battery_response is not None
	assert physical_state_response("actuator", "stuck") is None


def test_physical_state_ground_truth_only_includes_observed_catalog_states() -> None:
	ground_truth = physical_state_ground_truth(
		[
			"temperature:M1:overheating",
			"temperature:M1:critical_overheating",
			"battery:M2:low_battery",
			"battery:M2:dead_battery",
			"actuator:M3:stuck",
			"temperature:M1:overheating",
		]
	)

	assert ground_truth == [
		{
			"truth_id": "physical_state:temperature:M1:overheating",
			"diagnosis": "temperature:M1:overheating",
			"required_action": "start_cooling",
			"source": "physical_state",
			"evaluation_role": "physical_state_response",
			"evaluable": True,
		},
		{
			"truth_id": "physical_state:temperature:M1:critical_overheating",
			"diagnosis": "temperature:M1:critical_overheating",
			"required_action": "start_intense_cooling",
			"source": "physical_state",
			"evaluation_role": "physical_state_response",
			"evaluable": True,
		},
		{
			"truth_id": "physical_state:battery:M2:low_battery",
			"diagnosis": "battery:M2:low_battery",
			"required_action": "replace_battery",
			"source": "physical_state",
			"evaluation_role": "physical_state_response",
			"evaluable": True,
		},
		{
			"truth_id": "physical_state:battery:M2:dead_battery",
			"diagnosis": "battery:M2:dead_battery",
			"required_action": "replace_battery",
			"source": "physical_state",
			"evaluation_role": "physical_state_response",
			"evaluable": True,
		},
	]


def test_constructed_ground_truth_rows_have_truth_identity_and_evaluation_role() -> None:
	rows = [
		*ground_truth_for_root_faults(["sensor:M1:Power:stuck"]),
		*physical_state_ground_truth(["temperature:M1:overheating"]),
	]

	assert {row["truth_id"] for row in rows} == {
		"root_fault:sensor:M1:Power:stuck",
		"physical_state:temperature:M1:overheating",
	}
	assert {row["evaluation_role"] for row in rows} == {"root_fault", "physical_state_response"}


def test_duplicate_diagnosis_groups_are_auditable_for_true_duplicate_rows() -> None:
	groups = duplicate_diagnosis_groups(
		[
			{
				"truth_id": "root_fault:sensor:M1:Power:stuck",
				"diagnosis": "sensor:M1:Power:stuck",
				"source": "root_fault",
				"evaluation_role": "root_fault",
			},
			{
				"truth_id": "audit:sensor:M1:Power:stuck",
				"diagnosis": "sensor:M1:Power:stuck",
				"source": "physical_state",
				"evaluation_role": "physical_state_response",
			},
		]
	)

	assert groups == [
		{
			"diagnosis": "sensor:M1:Power:stuck",
			"count": 2,
			"truth_ids": ["root_fault:sensor:M1:Power:stuck", "audit:sensor:M1:Power:stuck"],
			"sources": ["physical_state", "root_fault"],
			"evaluation_roles": ["physical_state_response", "root_fault"],
		}
	]


def test_build_experiment_record_adds_observed_physical_state_ground_truth() -> None:
	reporter.clear()
	reporter.physical_state(
		"temperature:Sheet-Metal-Press:overheating",
		component="Temperature",
		context={"time": 3.0},
	)

	record = build_experiment_record(
		window_start=0.0,
		window_end=10.0,
		detector=RuleBasedDetector(),
		runtime_llm_decisions=[],
		agent_actions=[
			{
				"diagnosis_id": "temperature:Sheet-Metal-Press:overheating",
				"recommended_action": "start_cooling",
				"execution_attempted": True,
				"execution_succeeded": True,
			}
		],
	)

	assert record.ground_truth == [
		{
			"truth_id": "physical_state:temperature:Sheet-Metal-Press:overheating",
			"diagnosis": "temperature:Sheet-Metal-Press:overheating",
			"required_action": "start_cooling",
			"source": "physical_state",
			"evaluation_role": "physical_state_response",
			"evaluable": True,
		}
	]
	assert record.runtime_correctness["metrics"]["per_fault_action_selected_rate"] == 1.0


def test_build_experiment_record_detection_metrics_ignore_derived_issue_ground_truth() -> None:
	reporter.clear()
	reporter.root_fault("actuator:Sheet-Metal-Press:stuck", context={"time": 1.0})
	reporter.derived_issue(
		"machine:Sheet-Metal-Press:production_blocked",
		component="Actuator",
		cause_id="actuator:Sheet-Metal-Press:stuck",
		context={"time": 2.0},
	)

	record = build_experiment_record(
		window_start=0.0,
		window_end=10.0,
		detector=RuleBasedDetector(),
		runtime_llm_decisions=[],
		agent_actions=[],
	)
	detection_metrics = cast(dict[str, object], record.detection_metrics["metrics"])
	runtime_metrics = cast(dict[str, object], record.runtime_correctness["metrics"])
	detection_per_fault = cast(list[dict[str, object]], record.detection_metrics["per_fault"])

	assert {item["source"] for item in record.ground_truth} == {"root_fault", "derived_issue"}
	assert detection_metrics["evaluable_faults"] == runtime_metrics["evaluable_faults"] == 1
	assert [fault["diagnosis"] for fault in detection_per_fault] == ["actuator:Sheet-Metal-Press:stuck"]


def test_build_experiment_record_passes_only_observable_events_to_detector() -> None:
	reporter.clear()
	reporter.root_fault("sensor:Sheet-Metal-Press:Power:stuck", context={"time": 1.0})
	reporter.physical_state("battery:Sheet-Metal-Press:low_battery", component="Sheet-Metal-Press", context={"time": 2.0})
	reporter.observation(
		"sensor:Sheet-Metal-Press:Power:sensor_stuck_detected",
		component="Sheet-Metal-Press",
		context={"time": 3.0},
	)
	reporter.derived_issue(
		"machine:Sheet-Metal-Press:production_blocked",
		component="Sheet-Metal-Press",
		context={"time": 4.0},
	)

	class _RecordingDetector:
		def __init__(self):
			self.windows = []

		def detect(self, reports, *, window):
			del reports
			self.windows.append(window)
			return ()

	detector = _RecordingDetector()
	build_experiment_record(
		window_start=0.0,
		window_end=10.0,
		detector=detector,
		runtime_llm_decisions=[],
	)

	assert [[event.kind for event in window.events] for window in detector.windows] == [["observation", "derived_issue"]]
	assert [[event.identifier for event in window.events] for window in detector.windows] == [
		[
			"sensor:Sheet-Metal-Press:Power:sensor_stuck_detected",
			"machine:Sheet-Metal-Press:production_blocked",
		]
	]


def test_build_experiment_record_marks_late_fault_as_not_evaluable() -> None:
	reporter.clear()
	reporter.root_fault("sensor:Sheet-Metal-Press:Power:stuck", context={"time": 8.0})

	record = build_experiment_record(
		window_start=0.0,
		window_end=10.0,
		detector=RuleBasedDetector(),
		runtime_llm_decisions=[],
	)

	assert record.ground_truth[0]["injected_at"] == 8.0
	assert record.ground_truth[0]["available_observation_time"] == 2.0
	assert record.ground_truth[0]["evaluable"] is False
	assert record.runtime_correctness["per_fault"] == []
	assert record.runtime_correctness["metrics"]["evaluable_faults"] == 0


def test_monitoring_report_keeps_actuator_root_cause_visible_to_resolver() -> None:
	report = monitoring_report_from_event(
		ReportedEvent(
			identifier="machine:Sheet-Metal-Press:production_blocked",
			kind="derived_issue",
			component="Actuator",
			cause_id="actuator:Sheet-Metal-Press:stuck",
			context={"time": 5.0},
		),
		1,
		visible_cause_id=resolver_visible_cause_id("actuator:Sheet-Metal-Press:stuck"),
	)

	assert report.agent_role == "machine_health"
	assert report.diagnosis_id == "actuator:Sheet-Metal-Press:stuck"


def test_write_jsonl_outputs_one_record_per_line(tmp_path) -> None:
	reporter.clear()
	reporter.root_fault("network:packet_loss", context={"time": 1.0})
	record = build_experiment_record(
		window_start=0.0,
		window_end=2.0,
		detector=RuleBasedDetector(),
		runtime_llm_decisions=[],
	)
	output_path = tmp_path / "runs.jsonl"

	write_jsonl([record], output_path)

	lines = output_path.read_text(encoding="utf-8").splitlines()
	assert len(lines) == 1
	assert "scenario" not in json.loads(lines[0])
