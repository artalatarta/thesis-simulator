from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from cps.agents.contracts import EvidenceWindow
from cps.agents.diagnosis import critical_overheating_machine_ids, event_is_observable, resolver_visible_cause_id
from cps.core.node.machine import Machine
from cps.core.reporting import ReportedEvent
from cps.types import MachineStatus


@dataclass(frozen=True)
class MonitoringContext:
	"""Evidence snapshot handed to monitoring agents for one evaluation window."""

	window: EvidenceWindow
	machine_status: Mapping[str, MachineStatus] = field(default_factory=lambda: MappingProxyType({}))
	belt_diagnostics: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
	agent_decision_history: Mapping[str, tuple[Mapping[str, object], ...]] = field(default_factory=lambda: MappingProxyType({}))
	_observable_events: tuple[ReportedEvent, ...] = field(init=False, default=(), repr=False, compare=False)

	def __post_init__(self) -> None:
		object.__setattr__(self, "machine_status", MappingProxyType(dict(self.machine_status)))
		object.__setattr__(
			self,
			"belt_diagnostics",
			MappingProxyType({machine_id: tuple(ids) for machine_id, ids in self.belt_diagnostics.items()}),
		)
		object.__setattr__(
			self,
			"agent_decision_history",
			MappingProxyType(
				{
					agent_name: tuple(MappingProxyType(dict(entry)) for entry in entries)
					for agent_name, entries in self.agent_decision_history.items()
				}
			),
		)
		object.__setattr__(
			self,
			"_observable_events",
			tuple(event for event in self.window.events if event_is_observable(event)),
		)

	def observable_events(self) -> tuple[ReportedEvent, ...]:
		return self._observable_events

	def critical_overheating_ids(self) -> frozenset[str]:
		return frozenset(critical_overheating_machine_ids((event.identifier, resolver_visible_cause_id(event.cause_id)) for event in self.observable_events()))


def build_monitoring_context(
	events: Iterable[ReportedEvent],
	*,
	window_start: float = 0.0,
	window_end: float | None = None,
	machine_status: Mapping[str, MachineStatus] | None = None,
	belt_diagnostics: Mapping[str, Sequence[str]] | None = None,
	agent_decision_history: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> MonitoringContext:
	"""Wrap raw evidence in a :class:`MonitoringContext`."""
	event_tuple = tuple(events)
	if window_end is None:
		times = [float(time) for event in event_tuple if isinstance(time := event.context.get("time"), int | float)]
		window_end = max([window_start, *times]) if times else window_start
	window = EvidenceWindow(start_time=window_start, end_time=window_end, events=event_tuple)
	return MonitoringContext(
		window=window,
		machine_status=machine_status or {},
		belt_diagnostics={machine_id: tuple(ids) for machine_id, ids in (belt_diagnostics or {}).items()},
		agent_decision_history={agent_name: tuple(entries) for agent_name, entries in (agent_decision_history or {}).items()},
	)


def monitoring_status_for_machine(machine: Machine) -> MachineStatus:
	return {
		"is_processing": machine.is_processing,
		"battery_level": machine.battery.level,
		"temperature": machine.temperature.value,
		"temperature_state": machine.temperature.state_id,
		"parts_produced": machine.parts_produced,
		"current_product": machine.production_state.active_work.product_id if machine.is_processing and machine.production_state.active_work is not None else None,
	}


def monitoring_belt_diagnostics_for_machine(machine: Machine) -> tuple[str, ...]:
	if machine.outgoing_belt is None:
		return ()
	return tuple(machine.outgoing_belt.active_diagnostic_ids())


def monitoring_context_from_machines(
	machines: Iterable[Machine],
	events: Iterable[ReportedEvent],
	*,
	window_start: float = 0.0,
	window_end: float | None = None,
	agent_decision_history: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> MonitoringContext:
	"""Build a context from machine-owned monitoring snapshots."""
	machine_tuple = tuple(machines)
	machine_status = {machine.id: monitoring_status_for_machine(machine) for machine in machine_tuple}
	belt_diagnostics = {machine.id: monitoring_belt_diagnostics_for_machine(machine) for machine in machine_tuple}
	return build_monitoring_context(
		events,
		window_start=window_start,
		window_end=window_end,
		machine_status=machine_status,
		belt_diagnostics=belt_diagnostics,
		agent_decision_history=agent_decision_history,
	)
