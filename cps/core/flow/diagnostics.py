from typing import TYPE_CHECKING

from cps.core.flow.reasons import (
	ACTUATOR_SLOW_RESPONSE,
	BELT_CAPACITY,
	BELT_QUEUE_PRESSURE,
	DOWNSTREAM_ACTUATOR_SLOW_RESPONSE,
	DOWNSTREAM_CAPACITY_PRESSURE,
	DOWNSTREAM_INPUT_CAPACITY,
	INPUT_CAPACITY,
	NETWORK_LATENCY,
	NETWORK_OR_BELT_CAPACITY,
	PACKET_LOSS_RETRY,
	PARTIAL_CONGESTION,
)
from cps.core.flow.symptoms import BeltSymptomTracker, BottleneckDetector, SymptomRegistry
from cps.core.network import PACKET_LOSS
from cps.core.reporting import belt_issue_id, network_event_id

if TYPE_CHECKING:
	from cps.core.flow import BeltSegment
	from cps.core.node import DownstreamStatus


DIAGNOSTIC_BELT_ISSUES = frozenset({"handoff_blocked", "transfer_rate_degraded", "persistent_queue_pressure"})
BOTTLENECK_SYMPTOM_ORDER = ("queue_growth", "queue_waiting", "handoff_blocked", "transfer_rate_degraded")


class BeltSegmentDiagnostics:
	def __init__(self, belt: "BeltSegment") -> None:
		self.belt = belt
		self._symptoms = SymptomRegistry(
			lambda issue: self.belt_issue_id(issue) if issue in DIAGNOSTIC_BELT_ISSUES else issue
		)
		self._bottleneck = BottleneckDetector(self._symptoms)
		self._last_queue_depth = 0

	def belt_issue_id(self, issue: str) -> str:
		return belt_issue_id(self.belt.from_node.id, self.belt.to_node.id, issue)

	def active_diagnostic_ids(self) -> list[str]:
		return self._symptoms.active_ids(DIAGNOSTIC_BELT_ISSUES)

	def congestion_delay(self) -> float:
		if self.belt.capacity <= 1:
			belt_pressure = 0.0
		else:
			belt_pressure = len(self.belt.queue) / self.belt.capacity
		status = self.belt.to_node.downstream_status()
		downstream_pressure = status.capacity_pressure if status.can_accept else 0.0
		return max(belt_pressure, downstream_pressure) * self.belt.congestion_delay_per_part

	def pressure_diagnostic(self) -> tuple[str, str | None, dict[str, object]]:
		status = self.belt.to_node.downstream_status()
		capacity_cause, cause_id = self._pressure_cause(status)
		context: dict[str, object] = {
			"belt_occupancy": len(self.belt.queue),
			"belt_capacity": self.belt.capacity,
			"downstream_pressure": status.capacity_pressure,
			"capacity_cause": capacity_cause,
		}
		return capacity_cause, cause_id, context

	def _pressure_cause(self, status: "DownstreamStatus") -> tuple[str, str | None]:
		if self.belt.capacity > 1 and self.belt.queue:
			return BELT_QUEUE_PRESSURE, self.belt_issue_id("transfer_rate_degraded")
		if status.capacity_pressure_cause == ACTUATOR_SLOW_RESPONSE:
			cause_id = status.production_slowdown_issue_id
			if cause_id is not None:
				self._emit_downstream_machine_issue(status, cause_id)
			return DOWNSTREAM_ACTUATOR_SLOW_RESPONSE, cause_id
		if status.production_slowdown_issue_id is not None:
			return DOWNSTREAM_CAPACITY_PRESSURE, status.production_slowdown_issue_id
		return DOWNSTREAM_CAPACITY_PRESSURE, self.belt_issue_id("transfer_rate_degraded")

	def network_cause_id(self, reason: str | None) -> str | None:
		if reason == NETWORK_LATENCY:
			return network_event_id("network_latency_detected")
		if reason in {PACKET_LOSS, PACKET_LOSS_RETRY}:
			return network_event_id("network_packet_loss_detected")
		return None

	def symptom(self, issue: str) -> BeltSymptomTracker:
		return self._symptoms.get(issue)

	def observe_symptom(self, issue: str, *, cause_id: str | None = None) -> BeltSymptomTracker:
		return self._symptoms.observe(issue, self.belt.env.now, cause_id=cause_id)

	def observe_queue_growth_or_waiting(self, *, cause_id: str | None = None) -> None:
		queue_depth = len(self.belt.queue)
		if queue_depth > self._last_queue_depth:
			self.observe_symptom("queue_growth", cause_id=cause_id)
		elif queue_depth > 0:
			self.observe_symptom("queue_waiting", cause_id=cause_id)
		self._last_queue_depth = queue_depth

	def report_handoff_blocked(self, product_id: str, reason: str, *, cause_id: str | None = None) -> None:
		handoff_blocked_issue_id = self.belt_issue_id("handoff_blocked")
		status = self.belt.to_node.downstream_status()
		cause_id = cause_id or status.blockage_issue_id(reason)
		if cause_id is not None:
			self._emit_downstream_machine_issue(status, cause_id)
		self.observe_symptom("handoff_blocked", cause_id=cause_id)
		self.observe_queue_growth_or_waiting(cause_id=cause_id or handoff_blocked_issue_id)
		self.belt.event_reporter.derived_issue(
			handoff_blocked_issue_id,
			component=self.belt.from_node.id,
			cause_id=cause_id,
			context={
				"from_node": self.belt.from_node.id,
				"to_node": self.belt.to_node.id,
				"product_id": product_id,
				"reason": reason,
				"capacity_cause": self.capacity_cause(reason),
			},
		)
		self._emit_upstream_machine_blocked(handoff_blocked_issue_id)
		self.report_bottleneck_if_persistent(product_id, "handoff_blocked")

	def report_transfer_rate_degraded(
		self,
		product_id: str,
		delay: float,
		*,
		reason: str = PARTIAL_CONGESTION,
		cause_id: str | None = None,
		extra_context: dict[str, object] | None = None,
	) -> None:
		transfer_rate_degraded_issue_id = self.belt_issue_id("transfer_rate_degraded")
		self.observe_symptom("transfer_rate_degraded", cause_id=cause_id)
		self.observe_queue_growth_or_waiting(cause_id=cause_id or transfer_rate_degraded_issue_id)
		context: dict[str, object] = {
			"from_node": self.belt.from_node.id,
			"to_node": self.belt.to_node.id,
			"product_id": product_id,
			"reason": reason,
			"delay": delay,
			"occupancy": len(self.belt.queue),
			"capacity": self.belt.capacity,
		}
		context.update(extra_context or {})
		self.belt.event_reporter.derived_issue(
			transfer_rate_degraded_issue_id,
			component=self.belt.from_node.id,
			cause_id=cause_id,
			context=context,
		)
		self.report_bottleneck_if_persistent(product_id, "transfer_rate_degraded")

	def report_machine_slowdown_if_persistent(self, cause_id: str, symptom: BeltSymptomTracker) -> None:
		transfer_rate_degraded_issue_id = self.belt_issue_id("transfer_rate_degraded")
		from_status = self.belt.from_node.downstream_status()
		if (
			from_status.production_slowdown_issue_id is not None
			and symptom.identifier == transfer_rate_degraded_issue_id
			and symptom.duration(self.belt.env.now) >= self.belt.bottleneck_detection_delay
		):
			self.belt.event_reporter.derived_issue(
				from_status.production_slowdown_issue_id,
				component=self.belt.from_node.id,
				cause_id=cause_id,
			)

	def report_bottleneck_if_persistent(self, product_id: str, cause_issue: str) -> None:
		cause = self.symptom(cause_issue)
		self.report_machine_slowdown_if_persistent(cause.cause_id or cause.identifier, cause)
		if self._bottleneck.reported:
			return
		diagnostic_symptom = self.persistent_bottleneck_symptom()
		if diagnostic_symptom is None:
			return
		self._bottleneck.mark_reported(diagnostic_symptom)
		self.observe_symptom("persistent_queue_pressure")
		bottleneck_cause_id = diagnostic_symptom.cause_id or diagnostic_symptom.identifier
		self.belt.event_reporter.derived_issue(
			self.belt_issue_id("persistent_queue_pressure"),
			component=self.belt.from_node.id,
			cause_id=bottleneck_cause_id,
			context={
				"from_node": self.belt.from_node.id,
				"to_node": self.belt.to_node.id,
				"product_id": product_id,
				"queue_depth": len(self.belt.queue),
				"capacity": self.belt.capacity,
				"symptom": diagnostic_symptom.identifier,
				"symptom_cause": bottleneck_cause_id,
				"symptom_duration": diagnostic_symptom.duration(self.belt.env.now),
				"symptom_occurrences": diagnostic_symptom.occurrences,
				"diagnosis_duration": self.belt.env.now
				- (self._bottleneck.active_since if self._bottleneck.active_since is not None else self.belt.env.now),
			},
		)

	def capacity_cause(self, reason: str) -> str:
		if reason == BELT_CAPACITY:
			return BELT_CAPACITY
		if self.belt.to_node.id == "FinalStorage":
			return NETWORK_OR_BELT_CAPACITY
		if reason == INPUT_CAPACITY:
			return DOWNSTREAM_INPUT_CAPACITY
		return reason

	def persistent_bottleneck_symptom(self) -> BeltSymptomTracker | None:
		return self._bottleneck.persistent_symptom(
			self.belt.env.now,
			BOTTLENECK_SYMPTOM_ORDER,
			detection_delay=self.belt.bottleneck_detection_delay,
			repeated_failure_threshold=self.belt.repeated_handoff_failure_threshold,
		)

	def clear_drain_symptoms(self, *, clear_blockage_symptoms: bool) -> None:
		self._last_queue_depth = len(self.belt.queue)
		if self.belt.queue:
			return
		self._symptoms.clear("queue_growth", "queue_waiting")
		if clear_blockage_symptoms:
			self._symptoms.clear("handoff_blocked")
		if self.belt.to_node.downstream_status().capacity_pressure > 0:
			return
		self._symptoms.clear("transfer_rate_degraded")
		if clear_blockage_symptoms:
			self._symptoms.clear("persistent_queue_pressure")
			self._bottleneck.reset()

	def _emit_downstream_machine_issue(self, status: "DownstreamStatus", cause_id: str) -> None:
		if cause_id not in {status.production_blocked_issue_id, status.production_slowdown_issue_id}:
			return
		self.belt.event_reporter.derived_issue(
			cause_id,
			component=status.node_id,
			cause_id=status.blocking_state_cause_id,
		)

	def _emit_upstream_machine_blocked(self, cause_id: str) -> None:
		from_status = self.belt.from_node.downstream_status()
		if from_status.production_blocked_issue_id is None:
			return
		self.belt.event_reporter.derived_issue(
			from_status.production_blocked_issue_id,
			component=self.belt.from_node.id,
			cause_id=cause_id,
		)
