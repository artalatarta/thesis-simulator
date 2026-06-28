import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cps.agents.contracts import (
	Conflict,
	DetectsConflicts,
	EvidenceWindow,
)
from cps.agents.diagnosis import event_is_observable
from cps.agents.identifiers import parse_identifier
from cps.agents.resolution import ConflictResolver
from cps.config import AGENT_POLL_INTERVAL
from cps.core.reporting import ReportedEvent
from cps.evaluation.cascade import score_cascades
from cps.evaluation.event_records import (
	dedupe_events,
	event_reports,
	events_in_window,
	physical_state_ids,
	root_fault_ids,
	root_fault_injection_times,
	unique_identifiers,
)
from cps.evaluation.ground_truth import (
	derived_issue_ground_truth,
	duplicate_diagnosis_groups,
	ground_truth_for_root_faults,
	physical_state_ground_truth,
)
from cps.evaluation.scoring import score_agent_decisions, score_report_diagnoses
from cps.evaluation.serialization import conflict_to_record, dataclass_to_record, resolution_decision_to_record


@dataclass(frozen=True)
class ExperimentRunRecord:
	event_window: dict[str, float]
	injected_root_faults: list[str]
	ground_truth: list[dict[str, Any]]
	generated_reports: list[dict[str, Any]]
	detected_conflicts: list[dict[str, Any]]
	detection_metrics: dict[str, Any]
	cascade: dict[str, Any]
	events: list[dict[str, Any]]
	ground_truth_audit: dict[str, Any] = field(default_factory=dict)
	runtime_llm_reports: list[dict[str, Any]] = field(default_factory=list)
	agent_actions: list[dict[str, Any]] = field(default_factory=list)
	runtime_correctness: dict[str, Any] = field(default_factory=dict)
	runtime_detection: dict[str, Any] = field(default_factory=dict)
	per_fault_outcomes: dict[str, Any] = field(default_factory=dict)
	resolver_correctness: dict[str, Any] = field(default_factory=dict)
	runtime_llm_decisions: list[dict[str, Any]] = field(default_factory=list)

	def to_json(self) -> str:
		return json.dumps(asdict(self), sort_keys=True)


def build_experiment_record(
	*,
	window_start: float,
	window_end: float,
	detector: DetectsConflicts | None = None,
	resolver: ConflictResolver | None = None,
	runtime_llm_decisions: list[dict[str, Any]] | None = None,
	runtime_conflicts: list[dict[str, Any]] | None = None,
	runtime_llm_reports: list[dict[str, Any]] | None = None,
	agent_actions: list[dict[str, Any]] | None = None,
	events: Iterable[ReportedEvent],
) -> ExperimentRunRecord:
	"""Build an experiment record from the explicitly supplied simulation events.

	Pass a ``resolver`` to resolve the windowed conflicts here, or pass
	already-serialized ``runtime_llm_decisions`` to reuse decisions made during the
	live run instead of issuing a second round of model calls.
	"""
	window_events = tuple(dedupe_events(events_in_window(events, window_start, window_end)))
	generated_reports = event_reports(window_events)
	monitoring_reports = [generated.report for generated in generated_reports]
	injected_root_faults = root_fault_ids(window_events)
	ground_truth = ground_truth_for_root_faults(injected_root_faults, _neighbors_from_belt_events(window_events))
	ground_truth.extend(physical_state_ground_truth(physical_state_ids(window_events)))
	ground_truth.extend(derived_issue_ground_truth(unique_identifiers(window_events, "derived_issue")))
	injection_times = root_fault_injection_times(window_events)
	for item in ground_truth:
		root_fault = item.get("root_fault")
		if not isinstance(root_fault, str):
			continue
		injected_at = injection_times.get(root_fault)
		available_observation_time = None if injected_at is None else max(0.0, window_end - injected_at)
		item["injected_at"] = injected_at
		item["available_observation_time"] = available_observation_time
		item["evaluable"] = available_observation_time is not None and available_observation_time >= AGENT_POLL_INTERVAL
	serialized_agent_actions = list(agent_actions or [])
	# Conflicts come solely from the injected report-level detector, matching
	# the live MonitoringDriver path when runtime conflicts are not supplied.
	conflicts: tuple[Conflict, ...] = ()
	if runtime_conflicts is None:
		if detector is None:
			raise ValueError("build_experiment_record needs a detector or precomputed runtime_conflicts.")
		window = EvidenceWindow(
			start_time=window_start,
			end_time=window_end,
			events=tuple(event for event in window_events if event_is_observable(event)),
		)
		conflicts = detector.detect(monitoring_reports, window=window)
		runtime_conflicts = [conflict_to_record(conflict) for conflict in conflicts]
	if runtime_llm_decisions is None:
		if resolver is None:
			raise ValueError("build_experiment_record needs a resolver or precomputed runtime_llm_decisions.")
		runtime_llm_decisions = [resolution_decision_to_record(resolver.resolve(conflict)) for conflict in conflicts]
	serialized_reports = [generated.to_record() for generated in generated_reports]
	serialized_runtime_reports = list(runtime_llm_reports or [])
	detection_metrics = score_report_diagnoses(serialized_reports, ground_truth)
	runtime_detection = score_report_diagnoses(serialized_runtime_reports, ground_truth)
	runtime_correctness = score_agent_decisions(
		serialized_agent_actions,
		runtime_llm_decisions,
		ground_truth,
		serialized_runtime_reports,
		runtime_conflicts,
		window_events,
	)
	return ExperimentRunRecord(
		event_window={"start": window_start, "end": window_end},
		injected_root_faults=injected_root_faults,
		ground_truth=ground_truth,
		generated_reports=serialized_reports,
		detected_conflicts=runtime_conflicts,
		detection_metrics=detection_metrics,
		runtime_detection=runtime_detection,
		per_fault_outcomes={
			"per_fault": runtime_correctness.get("per_fault", []),
			"metrics": runtime_correctness.get("metrics", {}),
		},
		resolver_correctness=runtime_correctness.get("resolution_correctness", {}),
		cascade=score_cascades(window_events, ground_truth, window_end=window_end),
		runtime_llm_reports=serialized_runtime_reports,
		agent_actions=serialized_agent_actions,
		ground_truth_audit={"duplicate_diagnoses": duplicate_diagnosis_groups(ground_truth)},
		runtime_correctness=runtime_correctness,
		events=[dataclass_to_record(event) for event in window_events],
		runtime_llm_decisions=runtime_llm_decisions,
	)


def _neighbors_from_belt_events(events: Iterable[ReportedEvent]) -> dict[str, tuple[str | None, str | None]]:
	"""Recover each machine's belt neighbours from the belt events in the window.

	Belt identifiers have the form ``belt:{from_node_id}:{to_node_id}:{issue}``,
	so a belt between A and B makes A's downstream neighbour B and B's upstream
	neighbour A. The line is linear, so this reconstructs the topology needed to
	resolve ``<from_node_id>``/``<to_node_id>`` placeholders without any config.
	"""
	neighbors: dict[str, tuple[str | None, str | None]] = {}
	for event in events:
		parsed = parse_identifier(event.identifier)
		if parsed.kind != "belt" or parsed.from_node_id is None or parsed.to_node_id is None:
			continue
		from_node, to_node = parsed.from_node_id, parsed.to_node_id
		_, down_for_to = neighbors.get(to_node, (None, None))
		neighbors[to_node] = (from_node, down_for_to)
		up_for_from, _ = neighbors.get(from_node, (None, None))
		neighbors[from_node] = (up_for_from, to_node)
	return neighbors


def write_jsonl(records: Iterable[ExperimentRunRecord], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as output:
		for record in records:
			output.write(record.to_json())
			output.write("\n")
