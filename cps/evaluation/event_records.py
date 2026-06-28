from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass

from cps.agents.diagnosis import (
	classify_identifier,
	critical_overheating_machine_ids,
	event_is_observable,
	resolver_visible_cause_id,
)
from cps.agents.fault_catalog import ACTUATOR_SENSOR_TYPE
from cps.agents.identifiers import machine_id_from_identifier, parse_identifier
from cps.agents.contracts import AgentRole, ConfidenceLevel, MonitoringReport
from cps.core.reporting import ReportedEvent
from cps.evaluation.serialization import dataclass_to_record

NETWORK_OBSERVATION_ROOT_FAULTS = {
	"network:network_latency_detected": "network:latency",
	"network:network_packet_loss_detected": "network:packet_loss",
}

ROLE_BY_KIND: dict[str, AgentRole] = {
	"actuator": "actuator",
	"network": "network",
	"belt": "belt",
	"battery": "power",
	"temperature": "temperature",
}

ROLE_BY_SENSOR_TYPE: dict[str, AgentRole] = {
	"Power": "power",
	"Temperature": "temperature",
	ACTUATOR_SENSOR_TYPE: "actuator",
}


def event_time(event: ReportedEvent, *, default: float = 0.0) -> float:
	"""Return the simulation time stamped on ``event``, or ``default`` if absent."""
	time = event.context.get("time")
	return float(time) if isinstance(time, int | float) else default


def agent_role_for_identifier(identifier: str) -> AgentRole:
	parsed = parse_identifier(identifier)
	if parsed.kind == "sensor":
		return ROLE_BY_SENSOR_TYPE.get(parsed.sensor_type or "", "machine_health")
	return ROLE_BY_KIND.get(parsed.kind, "machine_health")


def confidence_for_event(event: ReportedEvent) -> ConfidenceLevel:
	return "high" if event.kind == "observation" else "medium"


def monitoring_report_from_event(
	event: ReportedEvent,
	index: int,
	critical_overheating_ids: Iterable[str] = (),
	visible_cause_id: str | None = None,
) -> MonitoringReport:
	"""Build a deterministic evaluation report from a simulator event."""
	identifier = event.identifier
	agent_role = agent_role_for_identifier(identifier)
	classification = classify_identifier(
		identifier,
		visible_cause_id=visible_cause_id,
		critical_overheating_ids=critical_overheating_ids,
		prefer_visible_cause=True,
	)
	return MonitoringReport(
		report_id=f"{agent_role}-{index}",
		agent_role=agent_role,
		component=classification.component,
		machine_id=machine_id_from_identifier(identifier),
		time=event_time(event),
		diagnosis=classification.diagnosis,
		recommended_action=classification.recommended_action,
		confidence=confidence_for_event(event),
		evidence=(identifier,),
		rationale=f"{event.kind} event {identifier}",
		diagnosis_id=classification.diagnosis_id,
		agent_name="",
		agent_model="deterministic-llm-agent-stub",
		agent_kind="deterministic_stub",
		metadata={},
	)


@dataclass(frozen=True)
class GeneratedReport:
	"""A monitoring report plus resolver-visible fields from its source event."""

	report: MonitoringReport
	source: str
	kind: str
	identifier: str
	cause_id: str | None
	context: Mapping[str, object]

	def __post_init__(self) -> None:
		object.__setattr__(self, "cause_id", resolver_visible_cause_id(self.cause_id))

	@property
	def machine_id(self) -> str | None:
		return self.report.machine_id if self.report.machine_id is not None else machine_id_from_identifier(self.identifier)

	def to_record(self) -> dict[str, object]:
		record = dataclass_to_record(self.report)
		record.update(
			{
				"source": self.source,
				"kind": self.kind,
				"identifier": self.identifier,
				"cause_id": self.cause_id,
				"context": dict(self.context),
			}
		)
		return record


@dataclass(frozen=True)
class DerivedIssueAttribution:
	"""Window-scoped explanation of a derived issue's root attribution."""

	issue_id: str
	matched_event_time: float | None
	root_fault: str | None
	attribution_status: str


def events_in_window(events: Iterable[ReportedEvent], start: float, end: float) -> list[ReportedEvent]:
	return [event for event in events if isinstance(time := event.context.get("time"), int | float) and start <= float(time) <= end]


def dedupe_events(events: Iterable[ReportedEvent]) -> list[ReportedEvent]:
	"""Collapse repeated per-tick re-emissions of the same event."""
	seen: set[tuple[str, str, str, str | None]] = set()
	unique: list[ReportedEvent] = []
	for event in events:
		key = (event.identifier, event.kind, event.component, event.cause_id)
		if key in seen:
			continue
		seen.add(key)
		unique.append(event)
	return unique


def event_reports(events: Iterable[ReportedEvent]) -> list[GeneratedReport]:
	event_list = list(events)
	critical_overheating_ids = critical_overheating_machine_ids((event.identifier, resolver_visible_cause_id(event.cause_id)) for event in event_list if event_is_observable(event))
	reports: list[GeneratedReport] = []
	for index, event in enumerate(event_list, start=1):
		if not event_is_observable(event):
			continue
		report = monitoring_report_from_event(event, index, critical_overheating_ids, resolver_visible_cause_id(event.cause_id))
		reports.append(
			GeneratedReport(
				report=report,
				source=event.component,
				kind=event.kind,
				identifier=event.identifier,
				cause_id=event.cause_id,
				context=event.context,
			)
		)
	return reports


def unique_identifiers(events: Iterable[ReportedEvent], kind: str) -> list[str]:
	"""Ordered, de-duplicated identifiers of every event of ``kind``."""
	identifiers: list[str] = []
	for event in events:
		if event.kind != kind or event.identifier in identifiers:
			continue
		identifiers.append(event.identifier)
	return identifiers


def root_fault_ids(events: Iterable[ReportedEvent]) -> list[str]:
	return unique_identifiers(events, "root_fault")


def physical_state_ids(events: Iterable[ReportedEvent]) -> list[str]:
	return unique_identifiers(events, "physical_state")


def first_event_times(events: Iterable[ReportedEvent], kind: str) -> dict[str, float]:
	"""First recorded simulation time for each identifier of ``kind``.

	Events of ``kind`` that carry no usable ``time`` are skipped entirely
	rather than defaulted, so the result only holds genuinely timed events.
	"""
	times: dict[str, float] = {}
	for event in events:
		if event.kind != kind or event.identifier in times:
			continue
		time = event.context.get("time")
		if isinstance(time, int | float):
			times[event.identifier] = float(time)
	return times


def root_fault_injection_times(events: Iterable[ReportedEvent]) -> dict[str, float]:
	"""Return the first recorded simulation time for each injected root fault."""
	return first_event_times(events, "root_fault")


def root_fault_resolution_times(events: Iterable[ReportedEvent]) -> dict[str, float]:
	"""Return the first recorded simulation time for each resolved root fault."""
	return first_event_times(events, "fault_resolved")


def derived_issue_attribution(
	events: Iterable[ReportedEvent],
	issue_id: str,
	*,
	window_start: float | None = None,
	window_end: float | None = None,
) -> DerivedIssueAttribution:
	"""Return a transparent root attribution for the issue occurrence in a decision window."""
	event_list = list(events)
	event = _derived_issue_for_window(event_list, issue_id, window_start=window_start, window_end=window_end)
	if event is None:
		return DerivedIssueAttribution(
			issue_id=issue_id,
			matched_event_time=None,
			root_fault=None,
			attribution_status="no_matching_issue_event",
		)
	root_fault = _root_fault_for_event(event, event_list)
	return DerivedIssueAttribution(
		issue_id=issue_id,
		matched_event_time=event_time(event),
		root_fault=root_fault,
		attribution_status="attributed_to_root" if root_fault is not None else "no_root_cause_found",
	)


def _derived_issue_for_window(
	events: list[ReportedEvent],
	issue_id: str,
	*,
	window_start: float | None,
	window_end: float | None,
) -> ReportedEvent | None:
	matches = [event for event in events if event.kind == "derived_issue" and event.identifier == issue_id]
	if not matches:
		return None
	if window_end is None:
		return max(matches, key=event_time)
	in_window = [
		event
		for event in matches
		if (window_start is None or event_time(event) >= window_start) and event_time(event) <= window_end
	]
	if in_window:
		return max(in_window, key=event_time)
	before_window_end = [event for event in matches if event_time(event) <= window_end]
	if before_window_end:
		return max(before_window_end, key=event_time)
	return min(matches, key=event_time)


def _root_fault_for_event(event: ReportedEvent, events: list[ReportedEvent]) -> str | None:
	cause_id = event.cause_id
	visited: set[str] = set()
	while cause_id is not None and cause_id not in visited:
		visited.add(cause_id)
		cause = _causal_event_before(events, cause_id, event_time(event))
		if cause is None:
			return None
		if cause.kind == "root_fault":
			return cause.identifier
		if (root_fault_id := NETWORK_OBSERVATION_ROOT_FAULTS.get(cause.identifier)) is not None:
			root_fault = _causal_event_before(events, root_fault_id, event_time(cause))
			if root_fault is not None and root_fault.kind == "root_fault":
				return root_fault.identifier
		cause_id = cause.cause_id
	return None


def _causal_event_before(events: list[ReportedEvent], identifier: str, time: float) -> ReportedEvent | None:
	candidates = [event for event in events if event.identifier == identifier and event_time(event) <= time]
	return max(candidates, key=event_time) if candidates else None
