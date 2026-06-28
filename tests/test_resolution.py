import json

import pytest

from tests.fakes import MockLLMClient
from cps.agents.contracts import Conflict, ConflictType, EvidenceWindow, MonitoringReport
from cps.agents.diagnosis import component_label_for_identifier, diagnosis_label_for_catalog_id
from cps.agents.resolution import ConflictResolver, openrouter_client_from_env
from cps.agents.resolution.prompts import response_format, system_prompt


def _report(
	report_id: str,
	*,
	diagnosis: str,
	action: str,
	confidence: str = "medium",
	diagnosis_id: str | None = None,
	evidence: tuple[str, ...] = (),
) -> MonitoringReport:
	return MonitoringReport(
		report_id=report_id,
		agent_role="machine_health",
		machine_id="M1",
		time=10.0,
		diagnosis=diagnosis_label_for_catalog_id(diagnosis_id) if diagnosis_id is not None else diagnosis,  # type: ignore[arg-type]
		recommended_action=action,  # type: ignore[arg-type]
		confidence=confidence,  # type: ignore[arg-type]
		evidence=evidence,
		diagnosis_id=diagnosis_id,
		component=component_label_for_identifier(diagnosis_id) if diagnosis_id is not None else "Line",
	)


def _conflict(reports: tuple[MonitoringReport, ...], conflict_types: tuple[ConflictType, ...]) -> Conflict:
	return Conflict(
		conflict_id="conflict-M1-1",
		machine_id="M1",
		window=EvidenceWindow(start_time=0.0, end_time=20.0),
		conflict_types=conflict_types,
		reports=reports,
	)


def _decision_json(
	*,
	selected_report_index: object = 1,
	confidence: str = "high",
	explanation: str = "Sensor fault is the root cause.",
) -> str:
	return json.dumps(
		{
			"selected_report_index": selected_report_index,
			"confidence": confidence,
			"explanation": explanation,
		}
	)


def test_resolver_schema_selects_report_by_index() -> None:
	properties = response_format()["json_schema"]["schema"]["properties"]  # type: ignore[index]
	assert properties["selected_report_index"] == {"type": "integer", "minimum": 1}
	assert "selected_diagnosis" not in properties
	assert "selected_action" not in properties
	assert "selected_diagnosis_id" not in properties


def test_resolver_prompt_requires_copying_one_report_exactly() -> None:
	prompt = system_prompt()
	assert "Return only its 1-based selected_report_index" in prompt
	assert "use 1 for the first report, 2 for the second report, and so on" in prompt
	assert "Do not modify, generalize, combine, or infer a new diagnosis/action/id" in prompt


def test_resolver_accepts_injected_client() -> None:
	client = MockLLMClient([_decision_json()])
	resolver = ConflictResolver(client)
	decision = resolver.resolve(
		_conflict(
			(
				_report("a", diagnosis="sensor_fault", action="fix_stuck", diagnosis_id="sensor:M1:Power:stuck"),
				_report("b", diagnosis="battery_issue", action="replace_battery", diagnosis_id="battery:M1:low_battery"),
			),
			("diagnosis", "action"),
		)
	)
	assert decision.conflict_id == "conflict-M1-1"
	assert decision.selected_action == "fix_stuck"


def _two_way_conflict() -> Conflict:
	return _conflict(
		(
			_report("a", diagnosis="sensor_fault", action="fix_stuck", diagnosis_id="sensor:M1:Power:stuck"),
			_report("b", diagnosis="battery_issue", action="replace_battery", diagnosis_id="battery:M1:low_battery"),
		),
		("diagnosis", "action"),
	)


def test_single_llm_parses_valid_json_and_records_trace() -> None:
	client = MockLLMClient([_decision_json()], latency_ms=12.5, prompt_tokens=123, completion_tokens=45)
	decision = ConflictResolver(client, temperature=0.0).resolve(_two_way_conflict())

	assert decision.selected_action == "fix_stuck"
	assert decision.selected_diagnosis == "stuck"
	assert decision.selected_diagnosis_id == "sensor:M1:Power:stuck"
	assert decision.supporting_report_ids == ("a", "b")

	metadata = decision.metadata
	assert metadata["model"] == "mock-model"
	assert metadata["temperature"] == 0.0
	assert metadata["attempts"] == 1
	assert metadata["parse_failures"] == 0
	assert metadata["selected_report_index"] == 1
	assert metadata["latency_ms"] == 12.5
	assert metadata["fell_back"] is False
	assert metadata["prompt_tokens"] == 123
	assert metadata["completion_tokens"] == 45
	assert len(client.calls) == 1
	system, user, temperature = client.calls[0]
	assert temperature == 0.0
	assert system and user


def test_single_llm_retries_then_uses_waiting_fallback_on_unparseable_output() -> None:
	client = MockLLMClient(["not valid json at all"])
	decision = ConflictResolver(client, max_retries=2).resolve(_two_way_conflict())

	assert decision.selected_action == "wait_for_more_evidence"
	assert decision.selected_diagnosis == "stuck"
	assert decision.selected_diagnosis_id == "sensor:M1:Power:stuck"
	assert decision.metadata["attempts"] == 3
	assert decision.metadata["parse_failures"] == 3
	assert decision.metadata["fell_back"] is True
	assert len(client.calls) == 3


def test_single_llm_accepts_supported_wait_for_more_evidence_as_resolver_action() -> None:
	client = MockLLMClient([_decision_json()])
	conflict = _conflict(
		(
			_report("a", diagnosis="sensor_fault", action="wait_for_more_evidence", diagnosis_id="sensor:M1:Power:stuck"),
			_report("b", diagnosis="battery_issue", action="replace_battery", diagnosis_id="battery:M1:low_battery"),
		),
		("diagnosis", "action"),
	)
	decision = ConflictResolver(client, max_retries=0).resolve(conflict)

	assert decision.selected_action == "wait_for_more_evidence"
	assert decision.selected_diagnosis == "stuck"
	assert decision.selected_diagnosis_id == "sensor:M1:Power:stuck"
	assert decision.metadata["attempts"] == 1
	assert decision.metadata["parse_failures"] == 0
	assert decision.metadata["fell_back"] is False


def test_single_llm_selects_action_from_chosen_report() -> None:
	client = MockLLMClient([_decision_json(selected_report_index=2)])
	decision = ConflictResolver(client, max_retries=0).resolve(_two_way_conflict())

	assert decision.selected_action == "replace_battery"
	assert decision.selected_diagnosis == "low_battery"
	assert decision.selected_diagnosis_id == "battery:M1:low_battery"
	assert decision.metadata["fell_back"] is False
	assert decision.metadata["selected_report_index"] == 2


def test_single_llm_rejects_missing_selected_report_index() -> None:
	client = MockLLMClient([json.dumps({"confidence": "high", "explanation": "Missing index."})])
	decision = ConflictResolver(client, max_retries=0).resolve(_two_way_conflict())

	assert decision.selected_action == "wait_for_more_evidence"
	assert decision.selected_diagnosis_id == "sensor:M1:Power:stuck"
	assert decision.metadata["fell_back"] is True


def test_single_llm_rejects_out_of_range_selected_report_index() -> None:
	client = MockLLMClient([_decision_json(selected_report_index=3)])
	decision = ConflictResolver(client, max_retries=0).resolve(_two_way_conflict())

	assert decision.selected_action == "wait_for_more_evidence"
	assert decision.metadata["fell_back"] is True


def test_single_llm_rejects_non_integer_selected_report_index() -> None:
	client = MockLLMClient([_decision_json(selected_report_index="1")])
	decision = ConflictResolver(client, max_retries=0).resolve(_two_way_conflict())

	assert decision.selected_action == "wait_for_more_evidence"
	assert decision.metadata["fell_back"] is True


def test_single_llm_rejects_boolean_selected_report_index() -> None:
	client = MockLLMClient([_decision_json(selected_report_index=True)])
	decision = ConflictResolver(client, max_retries=0).resolve(_two_way_conflict())

	assert decision.selected_action == "wait_for_more_evidence"
	assert decision.metadata["fell_back"] is True


def test_single_llm_recovers_after_one_parse_failure() -> None:
	client = MockLLMClient(["garbage", _decision_json()])
	decision = ConflictResolver(client, max_retries=2).resolve(_two_way_conflict())
	assert decision.selected_action == "fix_stuck"
	assert decision.metadata["attempts"] == 2
	assert decision.metadata["parse_failures"] == 1


def test_openrouter_client_from_env_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
	with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
		openrouter_client_from_env()
