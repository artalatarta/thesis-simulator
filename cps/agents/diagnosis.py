"""Shared, deterministic classification of reported events.

This module is the single source of truth for turning fault-catalog
identifiers and :class:`~cps.core.reporting.ReportedEvent` records into the
diagnosis labels and recommended actions used by role-specific monitoring
agents. Evaluation code reuses these domain classifications.

All functions are pure: they never read simulator state directly, only the
identifiers and metadata already captured on reported events.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeGuard, cast

from cps.agents.contracts import (
	ActionLabel,
	ComponentLabel,
	DiagnosisLabel,
	is_fault_catalog_diagnosis_id,
)
from cps.agents.fault_catalog import (
	ACTUATOR_OBSERVATION_FAULTS,
	ACTUATOR_SENSOR_TYPE,
	BATTERY_OBSERVATION_STATES,
	BELT_OBSERVATION_FAULTS,
	CLEAR_ACTION_BY_FAULT_TYPE,
	MACHINE_OBSERVATION_FAULTS,
	NETWORK_OBSERVATION_FAULTS,
	SENSOR_OBSERVATION_FAULTS,
	TEMPERATURE_OBSERVATION_STATES,
)
from cps.agents.identifiers import machine_id_from_identifier, parse_identifier
from cps.core.reporting import (
	ReportedEvent,
	actuator_state_id,
	battery_state_id,
	belt_issue_id,
	machine_issue_id,
	network_event_id,
	sensor_event_id,
	temperature_state_id,
)

def is_critical_overheating_state(identifier: str | None) -> TypeGuard[str]:
	return identifier is not None and parse_identifier(identifier).is_critical_overheating_state


def critical_overheating_machine_ids(evidence: Iterable[tuple[str, str | None]]) -> set[str]:
	"""Machine ids with observable evidence that overheating is critical.

	Physical-state cause ids are simulator truth and are hidden from generated
	monitoring reports. The resolver may infer critical temperature only from
	direct critical physical-state evidence passed by a unit caller or a
	resolver-visible critical-temperature cause id.
	"""
	machine_ids: set[str] = set()
	for identifier, cause_id in evidence:
		for candidate_id in (identifier, cause_id):
			if not is_critical_overheating_state(candidate_id):
				continue
			machine_id = machine_id_from_identifier(candidate_id)
			if machine_id is not None:
				machine_ids.add(machine_id)
	return machine_ids


def is_hidden_physical_state(identifier: str) -> bool:
	"""Physical states are simulator/oracle evidence, not direct monitoring reports.

	Local observations and derived issues are the trusted resolver inputs for
	these deterministic scenarios. This preserves faults.md's distinction
	between hidden physical state and observable evidence.
	"""
	return parse_identifier(identifier).is_hidden_physical_state


def resolver_visible_cause_id(cause_id: str | None) -> str | None:
	if cause_id is None:
		return None
	if is_hidden_physical_state(cause_id):
		return None
	return cause_id


def event_is_observable(event: ReportedEvent) -> bool:
	"""Whether an event becomes a trusted monitoring report for the resolver.

	Physical-state events are simulator/oracle evidence rather than direct
	monitoring reports, so only local observations and derived production issues
	are exposed.
	"""
	if event.kind not in {"observation", "physical_state", "derived_issue"}:
		return False
	if event.kind == "physical_state":
		return False
	if is_hidden_physical_state(event.identifier):
		return False
	return True


def diagnosis_label_for_catalog_id(identifier: str) -> DiagnosisLabel:
	parsed = parse_identifier(identifier)
	if parsed.kind == "network":
		return cast(DiagnosisLabel, parsed.network_fault) if parsed.network_fault is not None else "unknown"
	if parsed.kind == "sensor":
		return cast(DiagnosisLabel, parsed.observation) if parsed.observation in {"no_signal", "stuck"} else "unknown"
	if parsed.kind in {"actuator", "battery", "temperature", "machine"}:
		return cast(DiagnosisLabel, parsed.state_or_issue) if parsed.state_or_issue is not None else "unknown"
	if parsed.kind == "belt":
		return cast(DiagnosisLabel, parsed.observation) if parsed.observation is not None else "unknown"
	return "unknown"


def component_label_for_identifier(identifier: str) -> ComponentLabel:
	parsed = parse_identifier(identifier)
	if parsed.kind == "sensor":
		if parsed.sensor_type in {"Power", "Temperature", ACTUATOR_SENSOR_TYPE}:
			return cast(ComponentLabel, f"{parsed.sensor_type}Sensor" if parsed.sensor_type in {"Power", "Temperature"} else parsed.sensor_type)
		return "Line"
	if parsed.kind == "battery":
		return "Battery"
	if parsed.kind == "temperature":
		return "Temperature"
	if parsed.kind == "actuator":
		return "Actuator"
	if parsed.kind == "network":
		return "Network"
	if parsed.kind == "machine":
		return "Machine"
	if parsed.kind == "belt":
		return "Belt"
	return "Line"


def diagnosis_and_action_for_identifier(
	identifier: str,
	critical_overheating_ids: Iterable[str] = (),
) -> tuple[str | None, DiagnosisLabel, ActionLabel]:
	parsed = parse_identifier(identifier)
	critical_ids = frozenset(critical_overheating_ids)
	if parsed.is_actuator_status_observation:
		fault = ACTUATOR_OBSERVATION_FAULTS[parsed.observation or ""]
		action: ActionLabel = "fix_slow_response" if fault == "slow_response" else "fix_stuck"
		return actuator_state_id(parsed.parts[1], fault), cast(DiagnosisLabel, fault), action
	if parsed.is_sensor_fault_observation:
		fault = SENSOR_OBSERVATION_FAULTS[parsed.observation or ""]
		action = "fix_no_signal" if fault == "no_signal" else "fix_stuck"
		return sensor_event_id(parsed.parts[1], parsed.parts[2], fault), cast(DiagnosisLabel, fault), action
	if parsed.is_battery_observation:
		state = BATTERY_OBSERVATION_STATES[parsed.observation or ""]
		return battery_state_id(parsed.parts[1], state), cast(DiagnosisLabel, state), "replace_battery"
	if parsed.is_temperature_observation:
		state = TEMPERATURE_OBSERVATION_STATES[parsed.observation or ""]
		if state == "overheating" and parsed.parts[1] in critical_ids:
			state = "critical_overheating"
		action = "start_intense_cooling" if state == "critical_overheating" else "start_cooling"
		return temperature_state_id(parsed.parts[1], state), cast(DiagnosisLabel, state), action
	if parsed.is_network_observation:
		fault = NETWORK_OBSERVATION_FAULTS[parsed.raw]
		action = "fix_packet_loss" if fault == "packet_loss" else "fix_latency"
		return network_event_id(fault), cast(DiagnosisLabel, fault), action
	if parsed.is_machine_fault_observation:
		fault = MACHINE_OBSERVATION_FAULTS[parsed.state_or_issue or ""]
		return machine_issue_id(parsed.parts[1], fault), cast(DiagnosisLabel, fault), cast(ActionLabel, CLEAR_ACTION_BY_FAULT_TYPE[fault])
	if parsed.is_belt_fault_observation:
		fault = BELT_OBSERVATION_FAULTS[parsed.observation or ""]
		return belt_issue_id(parsed.parts[1], parsed.parts[2], fault), cast(DiagnosisLabel, fault), cast(ActionLabel, CLEAR_ACTION_BY_FAULT_TYPE[fault])
	if is_fault_catalog_diagnosis_id(identifier):
		diagnosis = diagnosis_label_for_catalog_id(identifier)
		if parsed.kind == "sensor":
			action = "fix_no_signal" if parsed.observation == "no_signal" else "fix_stuck"
			return identifier, diagnosis, action
		if parsed.kind == "actuator":
			action = "fix_slow_response" if parsed.state_or_issue == "slow_response" else "fix_stuck"
			return identifier, diagnosis, action
		if parsed.is_machine_fault:
			return identifier, diagnosis, cast(ActionLabel, CLEAR_ACTION_BY_FAULT_TYPE[parsed.state_or_issue or ""])
		if parsed.is_belt_fault:
			return identifier, diagnosis, cast(ActionLabel, CLEAR_ACTION_BY_FAULT_TYPE[parsed.observation or ""])
		# Production issues (blocked/slowdown/bottleneck/...) are flow symptoms,
		# not clearable root faults: there is no action that repairs them directly.
		return identifier, diagnosis, "wait_for_more_evidence"
	if parsed.is_production_flow_issue:
		return identifier, diagnosis_label_for_catalog_id(identifier), "wait_for_more_evidence"
	return None, "unknown", "wait_for_more_evidence"


@dataclass(frozen=True)
class EventClassification:
	"""Canonical diagnosis and action derived from an event identifier."""

	diagnosis_id: str | None
	component: ComponentLabel
	diagnosis: DiagnosisLabel
	recommended_action: ActionLabel


def classify_identifier(
	identifier: str,
	*,
	visible_cause_id: str | None = None,
	critical_overheating_ids: Iterable[str] = (),
	prefer_visible_cause: bool = False,
) -> EventClassification:
	"""Classify an event identifier into the canonical report fields.

	With ``prefer_visible_cause`` the diagnosis is derived from a resolver-visible
	fault-catalog cause id when one exists.
	"""
	diagnosis_source = visible_cause_id if prefer_visible_cause and visible_cause_id is not None and is_fault_catalog_diagnosis_id(visible_cause_id) else identifier
	diagnosis_id, diagnosis, action = diagnosis_and_action_for_identifier(diagnosis_source, critical_overheating_ids)
	component = component_label_for_identifier(diagnosis_id or diagnosis_source)
	return EventClassification(component=component, diagnosis_id=diagnosis_id, diagnosis=diagnosis, recommended_action=action)
