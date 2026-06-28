import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, cast

import simpy

from cps.core.flow.symptoms import BeltSymptomTracker as BeltSymptomTracker
from cps.core.flow.diagnostics import BeltSegmentDiagnostics
from cps.core.flow.reasons import BELT_CAPACITY, DOWNSTREAM_REJECTED, HANDOFF_UNCONFIRMED, NETWORK_DELAY_REASONS
from cps.core.kpi import KPITracker
from cps.core.node import Node
from cps.core.network import HandoffResult, Network
from cps.core.reporting import EventReporter, belt_issue_id
from cps.types import ProcessGenerator

BeltFaultType = Literal["belt_slippage", "belt_jam"]
BELT_FAULT_TYPES: tuple[BeltFaultType, ...] = ("belt_slippage", "belt_jam")
BELT_TRANSFER_RATE_DEGRADED_MIN_DELAY = 0.5
BELT_TRANSFER_RATE_DEGRADED_MAX_DELAY = 1.5
BELT_REPAIR_MIN_TIME = 4.0
BELT_REPAIR_MAX_TIME = 6.0


@dataclass
class BeltSegment:
	env: simpy.Environment
	from_node: Node
	to_node: Node
	network: Network
	capacity: int = 1
	congestion_delay_per_part: float = 0.25
	blocked_retry_interval: float = 1.0
	bottleneck_detection_delay: float = 3.0
	repeated_handoff_failure_threshold: int = 3
	queue: deque[str] = field(default_factory=deque)
	fault_type: BeltFaultType | None = None
	fault_param: float = 0.0
	pending_repair: BeltFaultType | None = None
	_delivered_parts: dict[str, int] = field(default_factory=dict)
	diagnostics: BeltSegmentDiagnostics = field(init=False)
	event_reporter: EventReporter = field(init=False)

	def __post_init__(self) -> None:
		self.event_reporter = self.network.event_reporter
		self.diagnostics = BeltSegmentDiagnostics(self)

	def has_capacity(self) -> bool:
		return len(self.queue) < self.capacity

	def active_diagnostic_ids(self) -> list[str]:
		ids = self.diagnostics.active_diagnostic_ids()
		if self.fault_type is not None:
			# Surface a fault-naming detection alongside the flow symptoms so the
			# belt agent can cite it and dispatch the matching repair, mirroring how
			# sensor/network monitors expose a ``*_detected`` observation. Without it
			# the agent only ever sees handoff_blocked/transfer_rate_degraded, which it is
			# told carry no direct repair.
			ids.append(belt_issue_id(self.from_node.id, self.to_node.id, f"{self.fault_type}_detected"))
		return ids

	def inject_fault(self, fault_type: BeltFaultType) -> None:
		self.fault_type = fault_type
		self.fault_param = random.uniform(BELT_TRANSFER_RATE_DEGRADED_MIN_DELAY, BELT_TRANSFER_RATE_DEGRADED_MAX_DELAY) if fault_type == "belt_slippage" else 0.0
		fault_id = belt_issue_id(self.from_node.id, self.to_node.id, fault_type)
		issue = "transfer_rate_degraded" if fault_type == "belt_slippage" else "handoff_blocked"
		issue_id = belt_issue_id(self.from_node.id, self.to_node.id, issue)
		self.event_reporter.root_fault(fault_id, message=f"FAULT INJECTED {fault_id}")
		self.event_reporter.derived_issue(
			issue_id,
			component=self.from_node.id,
			cause_id=fault_id,
			context={"from_node": self.from_node.id, "to_node": self.to_node.id, "fault_delay": self.fault_param},
		)

	def inject_random_fault(self) -> tuple[str, str]:
		self.inject_fault(random.choice(BELT_FAULT_TYPES))
		return f"{self.from_node.id}->{self.to_node.id}", "Belt"

	def clear_fault(self, kpi_tracker: KPITracker, fault_type: str) -> bool:
		if self.fault_type != fault_type:
			return False
		active_fault_type = self.fault_type
		assert active_fault_type is not None
		logging.info(f"Corrective action: Clearing fault on belt:{self.from_node.id}->{self.to_node.id}", extra={"component": "System"})
		self.event_reporter.fault_resolved(belt_issue_id(self.from_node.id, self.to_node.id, active_fault_type), component=self.from_node.id)
		self.fault_type = None
		self.fault_param = 0.0
		kpi_tracker.track_fault_end(f"{self.from_node.id}->{self.to_node.id}", "Belt")
		return True

	def dispatch_repair(self, kpi_tracker: KPITracker, fault_type: BeltFaultType) -> bool:
		if self.fault_type != fault_type:
			return False
		logging.info(f"AGENT ACTION: Dispatching belt repair for {self.from_node.id}->{self.to_node.id} ({fault_type})", extra={"component": self.from_node.id})
		self.clear_fault(kpi_tracker, fault_type)
		return True

	def _sample_repair_time(self) -> float:
		if BELT_REPAIR_MIN_TIME > BELT_REPAIR_MAX_TIME:
			raise ValueError("BELT_REPAIR_MIN_TIME must be less than or equal to BELT_REPAIR_MAX_TIME")
		return random.uniform(BELT_REPAIR_MIN_TIME, BELT_REPAIR_MAX_TIME)

	def has_queued_part(self, product_id: str) -> bool:
		return product_id in self.queue

	def consume_delivered_part(self, product_id: str) -> bool:
		delivered_count = self._delivered_parts.get(product_id, 0)
		if delivered_count <= 0:
			return False
		if delivered_count == 1:
			del self._delivered_parts[product_id]
		else:
			self._delivered_parts[product_id] = delivered_count - 1
		return True

	def handoff(self, product_id: str) -> ProcessGenerator:
		if self.fault_type == "belt_jam":
			fault_id = belt_issue_id(self.from_node.id, self.to_node.id, "belt_jam")
			self.diagnostics.report_handoff_blocked(product_id, "belt_fault", cause_id=fault_id)
			return False
		if not self.has_capacity():
			self.diagnostics.report_handoff_blocked(product_id, BELT_CAPACITY)
			return False

		congestion_delay = self.diagnostics.congestion_delay()
		if self.fault_type == "belt_slippage":
			congestion_delay += self.fault_param
		self.queue.append(product_id)
		if congestion_delay > 0:
			if self.fault_type == "belt_slippage":
				reason = "belt_fault"
				cause_id = belt_issue_id(self.from_node.id, self.to_node.id, "belt_slippage")
				context: dict[str, object] = {"fault_delay": self.fault_param}
			else:
				reason, cause_id, context = self.diagnostics.pressure_diagnostic()
			self.diagnostics.report_transfer_rate_degraded(
				product_id,
				congestion_delay,
				reason=reason,
				cause_id=cause_id,
				extra_context=context,
			)
			yield self.env.timeout(congestion_delay)
			self.diagnostics.report_bottleneck_if_persistent(product_id, "transfer_rate_degraded")

		handoff_result = cast(
			HandoffResult,
			(
				yield self.env.process(
					self.network.coordinate_handoff(
						self.from_node.id,
						self.to_node.id,
						product_id,
					)
				)
			),
		)
		if handoff_result.success and handoff_result.reason in NETWORK_DELAY_REASONS:
			cause_id = self.diagnostics.network_cause_id(handoff_result.reason)
			self.diagnostics.report_transfer_rate_degraded(
				product_id,
				handoff_result.delay,
				reason=handoff_result.reason,
				cause_id=cause_id,
			)
		if not handoff_result.success:
			self.diagnostics.report_handoff_blocked(
				product_id,
				handoff_result.reason or HANDOFF_UNCONFIRMED,
				cause_id=self.diagnostics.network_cause_id(handoff_result.reason),
			)
			self._remove_queued_part(product_id)
			self.diagnostics.clear_drain_symptoms(clear_blockage_symptoms=False)
			return False

		delivered = yield self.env.process(self.deliver_queued_part(product_id))
		return delivered

	def deliver_queued_part(self, product_id: str) -> ProcessGenerator:
		while True:
			if not self.queue or self.queue[0] != product_id:
				self.diagnostics.observe_queue_growth_or_waiting(cause_id=belt_issue_id(self.from_node.id, self.to_node.id, "transfer_rate_degraded"))
				self.diagnostics.report_bottleneck_if_persistent(product_id, "queue_waiting")
				yield self.env.timeout(self.blocked_retry_interval)
				continue
			status = self.to_node.downstream_status()
			if status.can_accept:
				accepted = self.to_node.receive_part(product_id)
				if accepted:
					self.queue.popleft()
					self._delivered_parts[product_id] = self._delivered_parts.get(product_id, 0) + 1
					self.diagnostics.clear_drain_symptoms(clear_blockage_symptoms=True)
					return True
			self.diagnostics.report_handoff_blocked(product_id, status.rejection_reason or DOWNSTREAM_REJECTED)
			yield self.env.timeout(self.blocked_retry_interval)

	def _remove_queued_part(self, product_id: str) -> None:
		try:
			self.queue.remove(product_id)
		except ValueError:
			return
