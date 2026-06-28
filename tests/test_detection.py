import json
from collections.abc import Mapping

from cps.agents.contracts import EvidenceWindow, MonitoringReport
from cps.agents.detection import ConflictDetector
from cps.agents.detection.parser import parse_detection
from cps.agents.detection.prompts import format_reports, response_format
from cps.agents.diagnosis import component_label_for_identifier, diagnosis_label_for_catalog_id
from cps.agents.llm.client import LLMCompletion
from cps.core.reporting import ReportedEvent
from tests.fakes import MockLLMClient


def _report(
	report_id: str,
	*,
	machine_id: str | None = "M1",
	diagnosis_id: str = "sensor:M1:Power:stuck",
	action: str = "fix_stuck",
	agent_name: str = "PowerSensor",
) -> MonitoringReport:
	return MonitoringReport(
		report_id=report_id,
		agent_role="machine_health",
		machine_id=machine_id,
		time=10.0,
		diagnosis=diagnosis_label_for_catalog_id(diagnosis_id),
		recommended_action=action,  # type: ignore[arg-type]
		confidence="high",
		evidence=(f"event:{report_id}",),
		rationale=f"rationale {report_id}",
		diagnosis_id=diagnosis_id,
		component=component_label_for_identifier(diagnosis_id),
		agent_name=agent_name,
	)


def _detection_json(conflicts: list[dict[str, object]]) -> str:
	return json.dumps({"conflicts": conflicts})


def test_parse_detection_accepts_valid_multi_conflict_payload() -> None:
	parsed = parse_detection(
		_detection_json(
			[
				{"report_ids": ["a", "b"], "conflict_types": ["diagnosis"], "description": "one"},
				{"report_ids": ["c", "d"], "conflict_types": ["action", "confidence"]},
			]
		),
		valid_report_ids=frozenset({"a", "b", "c", "d"}),
	)

	assert parsed is not None
	assert parsed[0].report_ids == ("a", "b")
	assert parsed[1].conflict_types == ("action", "confidence")
	assert parsed[1].description == ""


def test_parse_detection_accepts_empty_conflicts() -> None:
	assert parse_detection('{"conflicts": []}', valid_report_ids=frozenset({"a", "b"})) == ()


def test_parse_detection_defaults_empty_conflict_types() -> None:
	parsed = parse_detection(
		_detection_json([{"report_ids": ["a", "b"], "conflict_types": [], "description": "competing reports"}]),
		valid_report_ids=frozenset({"a", "b"}),
	)

	assert parsed is not None
	assert parsed[0].conflict_types == ("diagnosis",)


def test_parse_detection_rejects_invalid_payloads() -> None:
	assert parse_detection(_detection_json([{"report_ids": ["a", "missing"], "conflict_types": ["diagnosis"]}]), valid_report_ids=frozenset({"a"})) is None
	assert parse_detection(_detection_json([{"report_ids": ["a"], "conflict_types": ["diagnosis"]}]), valid_report_ids=frozenset({"a"})) is None
	assert parse_detection(_detection_json([{"report_ids": ["a", "b"], "conflict_types": ["bad"]}]), valid_report_ids=frozenset({"a", "b"})) is None
	assert parse_detection('{"conflicts": ["bad"]}', valid_report_ids=frozenset({"a", "b"})) is None


def test_parse_detection_rejects_report_id_reused_across_conflicts() -> None:
	assert (
		parse_detection(
			_detection_json(
				[
					{"report_ids": ["a", "b"], "conflict_types": ["diagnosis"]},
					{"report_ids": ["b", "c"], "conflict_types": ["action"]},
				]
			),
			valid_report_ids=frozenset({"a", "b", "c"}),
		)
		is None
	)


def test_detection_schema_and_prompt_grouping_reports_and_events() -> None:
	properties = response_format()["json_schema"]["schema"]["properties"]  # type: ignore[index]
	conflict_properties = properties["conflicts"]["items"]["properties"]
	assert set(conflict_properties) == {"report_ids", "conflict_types", "description"}

	user = format_reports(
		[_report("a")],
		window=EvidenceWindow(
			start_time=1.0,
			end_time=2.0,
			events=(ReportedEvent("sensor:M1:Power:sensor_stuck_detected", "observation", "M1", cause_id="network:packet_loss"),),
		),
	)
	assert "report_id=a" in user
	assert "sensor:M1:Power:sensor_stuck_detected" in user
	assert "cause_id" not in user
	assert "network:packet_loss" not in user


def test_detector_happy_path_builds_conflicts_and_preserves_prompt_order() -> None:
	reports = (
		_report("a", machine_id="M1", diagnosis_id="sensor:M1:Power:stuck"),
		_report("b", machine_id="M1", diagnosis_id="battery:M1:low_battery", action="replace_battery"),
		_report("c", machine_id="M2", diagnosis_id="network:packet_loss", action="fix_packet_loss"),
		_report("d", machine_id="M1", diagnosis_id="belt:M1:M2:belt_jam", action="fix_belt_jam"),
	)
	client = MockLLMClient(
		[
			_detection_json(
				[
					{"report_ids": ["a", "b"], "conflict_types": ["diagnosis", "action"], "description": "local conflict"},
					{"report_ids": ["c", "d"], "conflict_types": ["diagnosis"], "description": "cross machine"},
				]
			)
		]
	)

	conflicts = ConflictDetector(client).detect(reports, window=EvidenceWindow(start_time=5.0, end_time=10.0))

	assert [conflict.conflict_id for conflict in conflicts] == ["conflict-M1-t5-1", "conflict-line-t5-2"]
	assert conflicts[0].machine_id == "M1"
	assert conflicts[1].machine_id is None
	assert [report.report_id for report in conflicts[1].reports] == ["c", "d"]
	assert client.calls[0][2] == 0.0
	assert conflicts[0].description == "local conflict"


def test_detector_retries_with_temperature_schedule_then_succeeds() -> None:
	client = MockLLMClient(["bad", "still bad", _detection_json([{"report_ids": ["a", "b"], "conflict_types": ["diagnosis"]}])])
	detector = ConflictDetector(client)

	conflicts = detector.detect(
		(
			_report("a", diagnosis_id="sensor:M1:Power:stuck"),
			_report("b", diagnosis_id="battery:M1:low_battery", action="replace_battery"),
		),
		window=EvidenceWindow(start_time=0.0, end_time=5.0),
	)

	assert len(conflicts) == 1
	assert [call[2] for call in client.calls] == [0.0, 0.2, 0.4]
	assert detector.traces[0]["attempts"] == 3
	assert detector.traces[0]["parse_failures"] == 2
	assert detector.traces[0]["fell_back"] is False


def test_detector_fallback_returns_no_conflicts_and_records_trace() -> None:
	client = MockLLMClient(["bad"])
	detector = ConflictDetector(client, max_retries=1)

	conflicts = detector.detect(
		(
			_report("a", diagnosis_id="sensor:M1:Power:stuck"),
			_report("b", diagnosis_id="battery:M1:low_battery", action="replace_battery"),
		),
		window=EvidenceWindow(start_time=0.0, end_time=5.0),
	)

	assert conflicts == ()
	assert len(client.calls) == 2
	assert detector.traces[0]["fell_back"] is True
	assert detector.traces[0]["conflict_ids"] == []


def test_detector_client_exception_returns_no_conflicts_and_records_trace() -> None:
	class _RaisingClient:
		def complete(self, system: str, user: str, *, temperature: float, response_format: Mapping[str, object] | None = None) -> LLMCompletion:
			raise RuntimeError("client failed")

	detector = ConflictDetector(_RaisingClient())

	conflicts = detector.detect(
		(
			_report("a", diagnosis_id="sensor:M1:Power:stuck"),
			_report("b", diagnosis_id="battery:M1:low_battery", action="replace_battery"),
		),
		window=EvidenceWindow(start_time=0.0, end_time=5.0),
	)

	assert conflicts == ()
	assert detector.traces[0]["fell_back"] is True
	assert detector.traces[0]["attempts"] == 1
	assert detector.traces[0]["temperature"] == 0.0
	assert "error" in detector.traces[0]
	assert "client failed" in str(detector.traces[0]["error"])


def test_detector_skips_llm_when_fewer_than_two_actionable_reports() -> None:
	client = MockLLMClient([_detection_json([])])

	conflicts = ConflictDetector(client).detect(
		(_report("a", diagnosis_id="sensor:M1:Power:stuck"),),
		window=EvidenceWindow(start_time=0.0, end_time=5.0),
	)

	assert conflicts == ()
	assert client.calls == []
