import logging
import random
from dataclasses import dataclass

import simpy

from cps.config import OBSERVATION_MONITOR_INTERVAL
from cps.core.kpi import KPITracker
from cps.core.processes import process_is_alive
from cps.core.reporting import EventReporter, network_event_id
from cps.types import NetworkFaultType, ProcessGenerator

NETWORK_REPAIR_TIME = 5.0
NETWORK_FAULT_TYPES: tuple[NetworkFaultType, ...] = ("latency", "packet_loss")
# Canonical handoff-failure reasons emitted by coordinate_handoff and consumed
# by the flow layer (re-exported from cps.core.flow.reasons).
NETWORK_LATENCY = "network_latency"
PACKET_LOSS = "packet_loss"
PACKET_LOSS_RETRY = "packet_loss_retry"
DEFAULT_NETWORK_LATENCY_DELAY = 0.5


@dataclass(frozen=True)
class HandoffResult:
	success: bool
	delay: float
	reason: str | None = None


class Network:
	def __init__(
		self,
		env: simpy.Environment,
		event_reporter: EventReporter | None = None,
		*,
		handoff_timeout: float = 0.75,
		latency_base_mean: float = 0.05,
		latency_base_stddev: float = 0.01,
		latency_fault_delay: float | None = None,
		latency_fault_mean: float | None = None,
		latency_fault_stddev: float = 0.0,
		packet_loss_probability: float = 0.5,
	) -> None:
		self.env = env
		self.event_reporter = event_reporter or EventReporter()
		self.handoff_timeout = handoff_timeout
		self.latency_base_mean = latency_base_mean
		self.latency_base_stddev = latency_base_stddev
		if latency_fault_delay is None:
			latency_fault_delay = latency_fault_mean if latency_fault_mean is not None else DEFAULT_NETWORK_LATENCY_DELAY
		self.latency_fault_delay = latency_fault_delay
		self.latency_fault_stddev = latency_fault_stddev
		self.packet_loss_probability = packet_loss_probability
		self.fault_type: NetworkFaultType | None = None
		self.monitor_process: simpy.Process | None = None
		self.pending_repairs: set[str] = set()

	def inject_fault(
		self,
		fault_type: NetworkFaultType,
		*,
		latency_delay: float | None = None,
		packet_loss_percent: float | None = None,
	) -> None:
		self.fault_type = fault_type
		if fault_type == "latency":
			if latency_delay is not None:
				self.latency_fault_delay = float(latency_delay)
			context: dict[str, object] = {"latency_delay": self.latency_fault_delay}
		elif fault_type == "packet_loss":
			if packet_loss_percent is not None:
				self.packet_loss_probability = max(0.0, min(float(packet_loss_percent), 100.0)) / 100.0
			context = {
				"packet_loss_percent": self.packet_loss_probability * 100.0,
				"packet_loss_probability": self.packet_loss_probability,
			}
		else:
			context = {}
		fault_id = network_event_id(self.fault_type)
		self.event_reporter.root_fault(fault_id, message=f"FAULT INJECTED {fault_id}", component="Network", context=context)

	def inject_random_fault(self) -> tuple[str, str]:
		fault_type = random.choice(NETWORK_FAULT_TYPES)
		self.inject_fault(fault_type)
		return "network", fault_type

	def clear_fault(self) -> None:
		if self.fault_type is not None:
			self.event_reporter.fault_resolved(network_event_id(self.fault_type), component="Network")
		self.fault_type = None

	def observe_fault(self) -> str | None:
		if self.fault_type == "latency":
			return network_event_id("network_latency_detected")
		if self.fault_type == "packet_loss":
			return network_event_id("network_packet_loss_detected")
		return None

	def start_observation_monitor(self) -> None:
		if not process_is_alive(self.monitor_process):
			self.monitor_process = self.env.process(self.monitor_faults())

	def dispatch_repair(self, kpi_tracker: KPITracker) -> bool:
		if self.fault_type is None:
			return False
		fault_type = self.fault_type
		logging.info(f"AGENT ACTION: Scheduling network repair for {fault_type}", extra={"component": "Network"})
		logging.info(f"Corrective action: Network fault {fault_type} has been repaired.", extra={"component": "System"})
		if self.fault_type == fault_type:
			self.clear_fault()
			kpi_tracker.track_fault_end("network", fault_type)
		return True

	def monitor_faults(self) -> ProcessGenerator:
		while True:
			yield self.env.timeout(OBSERVATION_MONITOR_INTERVAL)
			observation_id = self.observe_fault()
			if observation_id is not None:
				self.event_reporter.observation(observation_id, component="Network")

	def coordinate_handoff(
		self,
		from_node: str,
		to_node: str,
		product_id: str,
	) -> ProcessGenerator:
		latency = abs(random.normalvariate(self.latency_base_mean, self.latency_base_stddev))
		if self.fault_type == "latency":
			latency += self.latency_fault_delay
			if self.latency_fault_stddev > 0:
				latency += abs(random.normalvariate(0.0, self.latency_fault_stddev))
		yield self.env.timeout(latency)
		if self.fault_type == "latency":
			delivered = latency <= self.handoff_timeout
			return HandoffResult(success=delivered, delay=latency, reason=NETWORK_LATENCY)
		if self.fault_type == "packet_loss":
			if random.random() < self.packet_loss_probability:
				return HandoffResult(success=False, delay=latency, reason=PACKET_LOSS)
			return HandoffResult(success=True, delay=latency, reason=PACKET_LOSS_RETRY)
		logging.info(
			f"Network: Handoff from {from_node} to {to_node} confirmed for {product_id}.",
			extra={"component": "Network"},
		)
		return HandoffResult(success=True, delay=latency)
