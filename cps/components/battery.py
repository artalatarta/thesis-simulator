import logging
from collections.abc import Callable

import simpy

from cps.core.kpi import KPITracker
from cps.core.reporting import battery_state_id

BATTERY_REPLACEMENT_MIN_TIME = 4.0
BATTERY_REPLACEMENT_MAX_TIME = 6.0


class Battery:
	def __init__(
		self,
		machine_id: str,
		level: float = 100.0,
		drain_rate: float = 1.2,
		idle_drain_divisor: float = 5.0,
		low_threshold: float = 20.0,
		depleted_threshold: float = 0.0,
		replacement_time_min: float = BATTERY_REPLACEMENT_MIN_TIME,
		replacement_time_max: float = BATTERY_REPLACEMENT_MAX_TIME,
	) -> None:
		self.machine_id = machine_id
		self.level = level
		self.drain_rate = drain_rate
		self.idle_drain_divisor = idle_drain_divisor
		self.low_threshold = low_threshold
		self.depleted_threshold = depleted_threshold
		self.replacement_time_min = replacement_time_min
		self.replacement_time_max = replacement_time_max
		self.pending_replacement = False

	@property
	def is_low(self) -> bool:
		return self.depleted_threshold < self.level < self.low_threshold

	@property
	def is_dead(self) -> bool:
		return self.level <= self.depleted_threshold

	@property
	def state_id(self) -> str | None:
		if self.is_dead:
			return battery_state_id(self.machine_id, "dead_battery")
		if self.is_low:
			return battery_state_id(self.machine_id, "low_battery")
		return None

	def drain(self, is_processing: bool) -> None:
		rate = self.drain_rate if is_processing else self.drain_rate / self.idle_drain_divisor
		self.level = max(self.depleted_threshold, self.level - rate)

	def replace(self) -> None:
		self.level = 100.0

	def dispatch_replacement(
		self,
		env: simpy.Environment,
		kpi_tracker: KPITracker,
		*,
		after_replace: Callable[[], object] | None = None,
	) -> bool:
		if self.pending_replacement:
			return True
		logging.info(f"AGENT ACTION: Dispatching battery replacement for {self.machine_id}", extra={"component": self.machine_id})
		logging.info(f"Corrective action: Battery for {self.machine_id} has been replaced.", extra={"component": "System"})
		self.replace()
		kpi_tracker.track_fault_end(self.machine_id, "Battery")
		if after_replace is not None:
			after_replace()
		return True
