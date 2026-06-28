from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, get_args

from cps.agents.fault_catalog import is_fault_catalog_diagnosis_id
from cps.core.reporting import ReportedEvent
from cps.types import ActionOutcome

AgentRole: TypeAlias = Literal["power", "temperature", "actuator", "machine_health", "network", "belt"]
MonitoringAgentKind: TypeAlias = Literal["llm_agent", "deterministic_stub"]
ComponentLabel: TypeAlias = Literal[
	"PowerSensor",
	"TemperatureSensor",
	"ActuatorSensor",
	"Battery",
	"Temperature",
	"Actuator",
	"Network",
	"Machine",
	"Belt",
	"Line",
]

DiagnosisLabel: TypeAlias = Literal[
	"no_signal",
	"stuck",
	"slow_response",
	"low_battery",
	"dead_battery",
	"overheating",
	"critical_overheating",
	"latency",
	"packet_loss",
	"production_blocked",
	"production_slowdown",
	"bearing_wear",
	"jammed_workpiece",
	"handoff_blocked",
	"persistent_queue_pressure",
	"transfer_rate_degraded",
	"belt_slippage",
	"belt_jam",
	"healthy",
	"ambiguous",
	"unknown",
]

ActionLabel: TypeAlias = Literal[
	"fix_stuck",
	"fix_no_signal",
	"fix_slow_response",
	"replace_battery",
	"fix_latency",
	"fix_packet_loss",
	"start_cooling",
	"start_intense_cooling",
	"fix_bearing_wear",
	"fix_jammed_workpiece",
	"fix_belt_slippage",
	"fix_belt_jam",
	"wait_for_more_evidence",
]

ConfidenceLevel: TypeAlias = Literal["very_low", "low", "medium", "high", "very_high"]
ConflictType: TypeAlias = Literal["diagnosis", "action", "confidence"]

# Derived from the Literal aliases above so the runtime tuples stay aligned with
# the static types. ``get_args`` preserves declaration order.
AGENT_ROLES: tuple[AgentRole, ...] = get_args(AgentRole)
COMPONENT_LABELS: tuple[ComponentLabel, ...] = get_args(ComponentLabel)
DIAGNOSIS_LABELS: tuple[DiagnosisLabel, ...] = get_args(DiagnosisLabel)
ACTION_LABELS: tuple[ActionLabel, ...] = get_args(ActionLabel)
CONFIDENCE_LEVELS: tuple[ConfidenceLevel, ...] = get_args(ConfidenceLevel)
CONFLICT_TYPES: tuple[ConflictType, ...] = get_args(ConflictType)
MONITORING_AGENT_KINDS: tuple[MonitoringAgentKind, ...] = get_args(MonitoringAgentKind)


def _validate_enum_value(field_name: str, value: str, allowed_values: tuple[str, ...]) -> None:
	if value not in allowed_values:
		allowed = ", ".join(allowed_values)
		raise ValueError(f"{field_name} must be one of: {allowed}. Got {value!r}.")


def _validate_enum_values(field_name: str, values: tuple[str, ...], allowed_values: tuple[str, ...]) -> None:
	for value in values:
		_validate_enum_value(field_name, value, allowed_values)


def _validate_fault_catalog_diagnosis_id(field_name: str, value: str | None) -> None:
	if value is not None and not is_fault_catalog_diagnosis_id(value):
		raise ValueError(f"{field_name} must match a faults.md diagnosis identifier. Got {value!r}.")


@dataclass(frozen=True)
class MonitoringReport:
	report_id: str
	agent_role: AgentRole
	machine_id: str | None
	time: float
	diagnosis: DiagnosisLabel
	recommended_action: ActionLabel
	confidence: ConfidenceLevel
	evidence: tuple[str, ...] = ()
	rationale: str = ""
	diagnosis_id: str | None = None
	component: ComponentLabel = "Line"
	agent_name: str = ""
	agent_kind: MonitoringAgentKind = "deterministic_stub"
	agent_model: str = "deterministic-llm-agent-stub"
	metadata: Mapping[str, object] = field(default_factory=dict)

	def __post_init__(self) -> None:
		object.__setattr__(self, "evidence", tuple(self.evidence))
		_validate_enum_value("agent_role", self.agent_role, AGENT_ROLES)
		_validate_enum_value("component", self.component, COMPONENT_LABELS)
		_validate_enum_value("agent_kind", self.agent_kind, MONITORING_AGENT_KINDS)
		_validate_enum_value("diagnosis", self.diagnosis, DIAGNOSIS_LABELS)
		_validate_enum_value("recommended_action", self.recommended_action, ACTION_LABELS)
		_validate_enum_value("confidence", self.confidence, CONFIDENCE_LEVELS)
		_validate_fault_catalog_diagnosis_id("diagnosis_id", self.diagnosis_id)


@dataclass(frozen=True)
class EvidenceWindow:
	start_time: float
	end_time: float
	events: tuple[ReportedEvent, ...] = ()

	def __post_init__(self) -> None:
		object.__setattr__(self, "events", tuple(self.events))
		if self.end_time < self.start_time:
			raise ValueError("end_time must be greater than or equal to start_time.")


@dataclass(frozen=True)
class Conflict:
	conflict_id: str
	machine_id: str | None
	window: EvidenceWindow
	conflict_types: tuple[ConflictType, ...]
	reports: tuple[MonitoringReport, ...]
	description: str = ""

	def __post_init__(self) -> None:
		object.__setattr__(self, "conflict_types", tuple(self.conflict_types))
		object.__setattr__(self, "reports", tuple(self.reports))
		if not self.conflict_types:
			raise ValueError("conflict_types must contain at least one conflict type.")
		if len(self.reports) < 2:
			raise ValueError("reports must contain at least two monitoring reports.")
		_validate_enum_values("conflict_types", self.conflict_types, CONFLICT_TYPES)


@dataclass(frozen=True)
class ResolutionDecision:
	decision_id: str
	conflict_id: str
	selected_diagnosis: DiagnosisLabel
	selected_action: ActionLabel
	confidence: ConfidenceLevel
	supporting_report_ids: tuple[str, ...] = ()
	explanation: str = ""
	metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
	selected_diagnosis_id: str | None = None

	def __post_init__(self) -> None:
		object.__setattr__(self, "supporting_report_ids", tuple(self.supporting_report_ids))
		_validate_enum_value("selected_diagnosis", self.selected_diagnosis, DIAGNOSIS_LABELS)
		_validate_enum_value("selected_action", self.selected_action, ACTION_LABELS)
		_validate_enum_value("confidence", self.confidence, CONFIDENCE_LEVELS)
		_validate_fault_catalog_diagnosis_id("selected_diagnosis_id", self.selected_diagnosis_id)
		object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class ResolvesConflicts(Protocol):
	"""Anything that can turn a detected conflict into a resolution decision."""

	def resolve(self, conflict: Conflict) -> ResolutionDecision: ...


class DetectsConflicts(Protocol):
	"""Anything that can detect conflicts among monitoring reports."""

	def detect(self, reports: Iterable[MonitoringReport], *, window: EvidenceWindow) -> tuple[Conflict, ...]: ...


def executed_action_to_record(
	report: MonitoringReport,
	outcome: ActionOutcome | None,
	*,
	selected_by_resolver: bool,
) -> dict[str, object]:
	execution_attempted = outcome is not None
	failure_reasons: dict[ActionOutcome | None, str | None] = {
		"succeeded": None,
		"already_resolved": None,
		"failed": "execution_failed",
		None: "no_action_handler",
	}
	return {
		"report_id": report.report_id,
		"agent_name": report.agent_name,
		"agent_role": report.agent_role,
		"component": report.component,
		"machine_id": report.machine_id,
		"time": report.time,
		"diagnosis_id": report.diagnosis_id,
		"recommended_action": report.recommended_action,
		"evidence": list(report.evidence),
		"selected_by_resolver": selected_by_resolver,
		"execution_attempted": execution_attempted,
		"execution_outcome": outcome if outcome is not None else "not_executed",
		"execution_succeeded": outcome == "succeeded",
		"failure_reason": failure_reasons[outcome],
	}
