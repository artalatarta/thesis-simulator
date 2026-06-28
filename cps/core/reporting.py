import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal


ReportKind = Literal["root_fault", "fault_resolved", "observation", "physical_state", "derived_issue"]

_LOG_LEVELS: dict[ReportKind, int] = {
	"root_fault": logging.ERROR,
	"fault_resolved": logging.INFO,
	"observation": logging.WARNING,
	"physical_state": logging.WARNING,
	"derived_issue": logging.WARNING,
}


@dataclass(frozen=True)
class ReportedEvent:
	identifier: str
	kind: ReportKind
	component: str
	cause_id: str | None = None
	context: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


class EventReporter:
	"""Collect the event stream for one simulation run.

	The same instance is injected into every component participating in that run.
	"""

	def __init__(self, now: Callable[[], float] | None = None) -> None:
		self.events: list[ReportedEvent] = []
		# When set, every reported event's context is stamped with the current time.
		self.now = now

	def clear(self) -> None:
		self.events.clear()

	def report(
		self,
		kind: ReportKind,
		identifier: str,
		*,
		component: str,
		cause_id: str | None = None,
		message: str | None = None,
		context: dict[str, object] | None = None,
	) -> ReportedEvent:
		event_context = dict(context or {})
		if self.now is not None:
			event_context.setdefault("time", float(self.now()))
		event = ReportedEvent(
			identifier=identifier,
			kind=kind,
			component=component,
			cause_id=cause_id,
			context=MappingProxyType(event_context),
		)
		self.events.append(event)
		log_extra: dict[str, object] = {
			"component": component,
			"event_id": identifier,
			"event_kind": kind,
		}
		if cause_id is not None:
			log_extra["cause_id"] = cause_id
		for key, value in event.context.items():
			log_extra[f"event_{key}"] = value
		logging.log(_LOG_LEVELS[kind], message or identifier, extra=log_extra)
		return event

	def root_fault(
		self,
		identifier: str,
		*,
		component: str = "FaultInjector",
		message: str | None = None,
		context: dict[str, object] | None = None,
	) -> ReportedEvent:
		return self.report("root_fault", identifier, component=component, message=message, context=context)

	def fault_resolved(
		self,
		identifier: str,
		*,
		component: str = "System",
		message: str | None = None,
		context: dict[str, object] | None = None,
	) -> ReportedEvent:
		return self.report("fault_resolved", identifier, component=component, message=message, context=context)

	def observation(
		self,
		identifier: str,
		*,
		component: str,
		message: str | None = None,
		context: dict[str, object] | None = None,
	) -> ReportedEvent:
		return self.report("observation", identifier, component=component, message=message, context=context)

	def physical_state(
		self,
		identifier: str,
		*,
		component: str,
		cause_id: str | None = None,
		message: str | None = None,
		context: dict[str, object] | None = None,
	) -> ReportedEvent:
		return self.report(
			"physical_state",
			identifier,
			component=component,
			cause_id=cause_id,
			message=message,
			context=context,
		)

	def derived_issue(
		self,
		identifier: str,
		*,
		component: str,
		cause_id: str | None = None,
		message: str | None = None,
		context: dict[str, object] | None = None,
	) -> ReportedEvent:
		if message is None and cause_id is not None:
			message = f"{identifier} caused by {cause_id}."
		return self.report(
			"derived_issue",
			identifier,
			component=component,
			cause_id=cause_id,
			message=message,
			context=context,
		)


def sensor_event_id(machine_id: str, sensor_type: str, event: str) -> str:
	return f"sensor:{machine_id}:{sensor_type}:{event}"


def network_event_id(event: str) -> str:
	return f"network:{event}"


def battery_state_id(machine_id: str, state: str) -> str:
	return f"battery:{machine_id}:{state}"


def temperature_state_id(machine_id: str, state: str) -> str:
	return f"temperature:{machine_id}:{state}"


def actuator_state_id(machine_id: str, state: str) -> str:
	return f"actuator:{machine_id}:{state}"


def machine_issue_id(machine_id: str, issue: str) -> str:
	return f"machine:{machine_id}:{issue}"


def belt_issue_id(from_node_id: str, to_node_id: str, issue: str) -> str:
	return f"belt:{from_node_id}:{to_node_id}:{issue}"
