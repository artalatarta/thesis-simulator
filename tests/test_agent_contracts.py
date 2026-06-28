from collections.abc import MutableMapping
from typing import cast

import pytest

from cps.agents.contracts import (
	ACTION_LABELS,
	CONFIDENCE_LEVELS,
	CONFLICT_TYPES,
	DIAGNOSIS_LABELS,
	ActionLabel,
	Conflict,
	ConflictType,
	DiagnosisLabel,
	EvidenceWindow,
	MonitoringAgentKind,
	MonitoringReport,
	ResolutionDecision,
	executed_action_to_record,
)
from cps.core.reporting import ReportedEvent


def test_action_record_exposes_unhandled_execution_lifecycle() -> None:
	report = MonitoringReport(
		report_id="sensor-machine1-10",
		agent_role="power",
		machine_id="machine1",
		time=10.0,
		diagnosis="stuck",
		recommended_action="fix_stuck",
		confidence="high",
		evidence=("sensor:machine1:Power:sensor_stuck_detected",),
		diagnosis_id="sensor:machine1:Power:stuck",
	)

	record = executed_action_to_record(report, None, selected_by_resolver=True)

	assert record["selected_by_resolver"] is True
	assert record["execution_attempted"] is False
	assert record["execution_succeeded"] is False
	assert record["failure_reason"] == "no_action_handler"


def test_action_record_treats_already_resolved_as_neutral_outcome() -> None:
	report = MonitoringReport(
		report_id="temperature-machine1-10",
		agent_role="temperature",
		machine_id="machine1",
		time=10.0,
		diagnosis="overheating",
		recommended_action="start_cooling",
		confidence="high",
		evidence=("temperature:machine1:overheating",),
		diagnosis_id="temperature:machine1:overheating",
	)

	record = executed_action_to_record(report, "already_resolved", selected_by_resolver=False)

	assert record["execution_attempted"] is True
	assert record["execution_outcome"] == "already_resolved"
	assert record["execution_succeeded"] is False
	assert record["failure_reason"] is None


def test_monitoring_report_contract_captures_agent_output() -> None:
	report = MonitoringReport(
		report_id="sensor-machine1-10",
		agent_role="power",
		machine_id="machine1",
		time=10.0,
		diagnosis="stuck",
		recommended_action="fix_stuck",
		confidence="high",
		evidence=("sensor:machine1:Power:sensor_stuck_detected",),
		rationale="Power readings diverged from expected battery state.",
		diagnosis_id="sensor:machine1:Power:stuck",
	)

	assert report.agent_role == "power"
	assert report.diagnosis in DIAGNOSIS_LABELS
	assert report.recommended_action in ACTION_LABELS
	assert report.confidence in CONFIDENCE_LEVELS
	assert report.diagnosis_id == "sensor:machine1:Power:stuck"
	assert report.agent_kind == "deterministic_stub"
	assert report.agent_model == "deterministic-llm-agent-stub"


def test_conflict_and_resolution_contract_link_reports_to_decision() -> None:
	sensor_report = MonitoringReport(
		report_id="sensor-machine1-20",
		agent_role="power",
		machine_id="machine1",
		time=20.0,
		diagnosis="stuck",
		recommended_action="fix_stuck",
		confidence="high",
	)
	machine_report = MonitoringReport(
		report_id="machine-health-machine1-20",
		agent_role="machine_health",
		machine_id="machine1",
		time=20.0,
		diagnosis="low_battery",
		recommended_action="replace_battery",
		confidence="medium",
	)
	conflict = Conflict(
		conflict_id="conflict-machine1-20",
		machine_id="machine1",
		window=EvidenceWindow(start_time=15.0, end_time=20.0),
		conflict_types=("diagnosis", "action"),
		reports=(sensor_report, machine_report),
	)
	decision = ResolutionDecision(
		decision_id="decision-machine1-20",
		conflict_id=conflict.conflict_id,
		selected_diagnosis="low_battery",
		selected_action="replace_battery",
		confidence="medium",
		supporting_report_ids=(machine_report.report_id,),
		selected_diagnosis_id="battery:machine1:low_battery",
	)

	assert set(conflict.conflict_types).issubset(CONFLICT_TYPES)
	assert decision.conflict_id == conflict.conflict_id
	assert decision.supporting_report_ids == ("machine-health-machine1-20",)
	assert decision.selected_diagnosis_id == "battery:machine1:low_battery"


def test_monitoring_report_rejects_unknown_contract_labels() -> None:
	with pytest.raises(ValueError, match="diagnosis must be one of"):
		MonitoringReport(
			report_id="sensor-machine1-10",
			agent_role="power",
			machine_id="machine1",
			time=10.0,
			diagnosis=cast(DiagnosisLabel, "not_a_diagnosis"),
			recommended_action="fix_stuck",
			confidence="high",
		)

	with pytest.raises(ValueError, match="agent_kind must be one of"):
		MonitoringReport(
			report_id="sensor-machine1-10",
			agent_role="power",
			machine_id="machine1",
			time=10.0,
			diagnosis="stuck",
			recommended_action="fix_stuck",
			confidence="high",
			agent_kind=cast(MonitoringAgentKind, "rule_engine"),
		)


def test_concrete_diagnosis_ids_must_match_fault_catalog() -> None:
	with pytest.raises(ValueError, match="diagnosis_id must match"):
		MonitoringReport(
			report_id="sensor-machine1-10",
			agent_role="power",
			machine_id="machine1",
			time=10.0,
			diagnosis="stuck",
			recommended_action="fix_stuck",
			confidence="high",
			diagnosis_id="sensor:machine1:ActuatorSensor:stuck",
		)

	with pytest.raises(ValueError, match="selected_diagnosis_id must match"):
		ResolutionDecision(
			decision_id="decision-machine1-10",
			conflict_id="conflict-machine1-10",
			selected_diagnosis="packet_loss",
			selected_action="fix_packet_loss",
			confidence="high",
			selected_diagnosis_id="network:packetloss",
		)


def test_tuple_fields_are_normalized_to_immutable_tuples() -> None:
	report = MonitoringReport(
		report_id="sensor-machine1-10",
		agent_role="power",
		machine_id="machine1",
		time=10.0,
		diagnosis="stuck",
		recommended_action="fix_stuck",
		confidence="high",
		evidence=cast(tuple[str, ...], ["sensor:machine1:Power:sensor_stuck_detected"]),
	)
	window = EvidenceWindow(start_time=5.0, end_time=10.0, events=cast(tuple[ReportedEvent, ...], []))
	conflict = Conflict(
		conflict_id="conflict-machine1-10",
		machine_id="machine1",
		window=window,
		conflict_types=cast(tuple[ConflictType, ...], ["diagnosis"]),
		reports=cast(
			tuple[MonitoringReport, ...],
			[
				report,
				MonitoringReport(
					report_id="machine-health-machine1-10",
					agent_role="machine_health",
					machine_id="machine1",
					time=10.0,
					diagnosis="low_battery",
					recommended_action="replace_battery",
					confidence="medium",
				),
			],
		),
	)
	decision = ResolutionDecision(
		decision_id="decision-machine1-10",
		conflict_id=conflict.conflict_id,
		selected_diagnosis="low_battery",
		selected_action="replace_battery",
		confidence="medium",
		supporting_report_ids=cast(tuple[str, ...], ["machine-health-machine1-10"]),
	)

	assert report.evidence == ("sensor:machine1:Power:sensor_stuck_detected",)
	assert window.events == ()
	assert conflict.conflict_types == ("diagnosis",)
	assert isinstance(conflict.reports, tuple)
	assert decision.supporting_report_ids == ("machine-health-machine1-10",)
	assert report.metadata == {}


def test_conflict_rejects_unknown_conflict_type() -> None:
	report = MonitoringReport(
		report_id="sensor-machine1-10",
		agent_role="power",
		machine_id="machine1",
		time=10.0,
		diagnosis="stuck",
		recommended_action="fix_stuck",
		confidence="high",
	)
	other_report = MonitoringReport(
		report_id="machine-health-machine1-10",
		agent_role="machine_health",
		machine_id="machine1",
		time=10.0,
		diagnosis="low_battery",
		recommended_action="replace_battery",
		confidence="medium",
	)

	with pytest.raises(ValueError, match="conflict_types must be one of"):
		Conflict(
			conflict_id="conflict-machine1-10",
			machine_id="machine1",
			window=EvidenceWindow(start_time=5.0, end_time=10.0),
			conflict_types=(cast(ConflictType, "not_a_conflict_type"),),
			reports=(report, other_report),
		)


def test_conflict_requires_at_least_one_type_and_two_reports() -> None:
	report = MonitoringReport(
		report_id="sensor-machine1-10",
		agent_role="power",
		machine_id="machine1",
		time=10.0,
		diagnosis="stuck",
		recommended_action="fix_stuck",
		confidence="high",
	)
	other_report = MonitoringReport(
		report_id="machine-health-machine1-10",
		agent_role="machine_health",
		machine_id="machine1",
		time=10.0,
		diagnosis="low_battery",
		recommended_action="replace_battery",
		confidence="medium",
	)

	with pytest.raises(ValueError, match="conflict_types must contain at least one conflict type"):
		Conflict(
			conflict_id="conflict-machine1-10",
			machine_id="machine1",
			window=EvidenceWindow(start_time=5.0, end_time=10.0),
			conflict_types=(),
			reports=(report, other_report),
		)

	with pytest.raises(ValueError, match="reports must contain at least two monitoring reports"):
		Conflict(
			conflict_id="conflict-machine1-10",
			machine_id="machine1",
			window=EvidenceWindow(start_time=5.0, end_time=10.0),
			conflict_types=("diagnosis",),
			reports=(report,),
		)


def test_evidence_window_rejects_reversed_time_range() -> None:
	with pytest.raises(ValueError, match="end_time must be greater than or equal to start_time"):
		EvidenceWindow(start_time=20.0, end_time=10.0)


def test_resolution_decision_rejects_unknown_contract_labels() -> None:
	with pytest.raises(ValueError, match="selected_action must be one of"):
		ResolutionDecision(
			decision_id="decision-machine1-20",
			conflict_id="conflict-machine1-20",
			selected_diagnosis="low_battery",
			selected_action=cast(ActionLabel, "not_an_action"),
			confidence="medium",
		)


def test_resolution_decision_metadata_is_immutable() -> None:
	source_metadata: dict[str, object] = {"model": "mock", "latency_seconds": 0.1}
	decision = ResolutionDecision(
		decision_id="decision-machine1-20",
		conflict_id="conflict-machine1-20",
		selected_diagnosis="low_battery",
		selected_action="replace_battery",
		confidence="medium",
		metadata=source_metadata,
	)

	source_metadata["model"] = "changed"

	assert decision.metadata["model"] == "mock"
	with pytest.raises(TypeError):
		cast(MutableMapping[str, object], decision.metadata)["model"] = "changed"
