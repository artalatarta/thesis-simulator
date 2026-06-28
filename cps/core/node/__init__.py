import logging
from dataclasses import dataclass, field

import simpy

from cps.types import ProcessGenerator


@dataclass
class DownstreamStatus:
	node_id: str
	can_accept: bool
	rejection_reason: str | None = None
	capacity_pressure: float = 0.0
	capacity_pressure_cause: str | None = None
	production_blocked_issue_id: str | None = None
	production_slowdown_issue_id: str | None = None
	blocking_state_cause_id: str | None = None

	def blockage_issue_id(self, reason: str) -> str | None:
		from cps.core.flow.reasons import BLOCKING_MACHINE_REASONS

		if reason in BLOCKING_MACHINE_REASONS:
			return self.production_blocked_issue_id
		return None


@dataclass
class Node:
	id: str

	def can_accept_part(self) -> bool:
		return True

	def capacity_pressure(self) -> float:
		return 0.0

	def rejection_reason(self) -> str | None:
		if self.can_accept_part():
			return None
		return "downstream_rejected"

	def receive_part(self, product_id: str) -> bool:
		_ = product_id
		return self.can_accept_part()

	def blocking_state_cause(self, reason: str) -> str | None:
		"""Internal state cause id behind a downstream-blocking reason."""
		_ = reason
		return None

	def capacity_pressure_cause(self) -> str | None:
		"""Machine-internal reason this node currently applies capacity pressure, if any."""
		return None

	def downstream_status(self) -> DownstreamStatus:
		return DownstreamStatus(
			node_id=self.id,
			can_accept=self.can_accept_part(),
			rejection_reason=self.rejection_reason(),
			capacity_pressure=self.capacity_pressure(),
			capacity_pressure_cause=self.capacity_pressure_cause(),
		)


@dataclass
class FinalStorage(Node):
	id: str = "FinalStorage"
	stored_parts: list[str] = field(default_factory=list)
	env: simpy.Environment | None = None

	def receive_part(self, product_id: str) -> bool:
		self.stored_parts.append(product_id)
		at_time = f" at T={self.env.now:.2f}" if self.env is not None else ""
		logging.info(f"{self.id}: Stored completed product {product_id}{at_time}.", extra={"component": self.id})
		return True


@dataclass
class RawMaterialSource(Node):
	"""Raw-material boundary node at the start of the production line."""

	id: str = "RawMaterialSource"

	def start_feeding(self, env: simpy.Environment, target: Node, product_ids: list[str]) -> simpy.Process:
		return env.process(self.feed(env, target, product_ids))

	def feed(self, env: simpy.Environment, target: Node, product_ids: list[str]) -> ProcessGenerator:
		"""Push raw stock straight into the first machine's input buffer.

		There is no infeed belt: the source connects to the first machine directly
		(see FactoryLineBuilder._link_machines for why), so feeding is a plain
		retry against the machine's input capacity rather than a belt handoff.
		"""
		for product_id in product_ids:
			while not target.receive_part(product_id):
				yield env.timeout(1)


__all__ = [
	"DownstreamStatus",
	"FinalStorage",
	"Node",
	"RawMaterialSource",
]
