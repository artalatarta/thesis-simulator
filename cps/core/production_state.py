from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from cps.core.flow import BeltSegment
from cps.types import ScheduleEntry


class WorkLocation(Enum):
	INPUT = "input"
	WORK_IN_PROGRESS = "work_in_progress"
	HANDOFF = "handoff"


@dataclass
class ActiveWorkItem:
	product_id: str
	schedule_entry: ScheduleEntry | None
	location: WorkLocation


@dataclass(frozen=True)
class PendingWorkItem:
	product_id: str
	process_time: float
	schedule_entry: ScheduleEntry
	from_input: bool


class ProductionState:
	def __init__(self, production_schedule: Sequence[ScheduleEntry], inbound_parts: list[str]) -> None:
		self.production_schedule: list[ScheduleEntry] = list(production_schedule)
		self.inbound_parts = inbound_parts
		self.active_work: ActiveWorkItem | None = None

	def handoff_product_id(self) -> str | None:
		if self.active_work is None or self.active_work.location is not WorkLocation.HANDOFF:
			return None
		return self.active_work.product_id

	def has_pending_work(self) -> bool:
		if self.active_work is not None:
			return True
		return bool(self.production_schedule and self.inbound_parts)

	def next_work(self) -> PendingWorkItem | None:
		if self.active_work is not None and self.active_work.location is WorkLocation.WORK_IN_PROGRESS:
			if not self.production_schedule:
				return None
			product_id = self.active_work.product_id
			schedule_entry = self.production_schedule.pop(0)
			return PendingWorkItem(product_id, schedule_entry[1], schedule_entry, from_input=False)
		if not self.production_schedule or not self.inbound_parts:
			return None
		product_id = self.inbound_parts[0]
		schedule_entry = self.production_schedule.pop(0)
		return PendingWorkItem(product_id, schedule_entry[1], schedule_entry, from_input=True)

	def start_work(self, work: PendingWorkItem) -> None:
		self.active_work = ActiveWorkItem(
			product_id=work.product_id,
			schedule_entry=work.schedule_entry,
			location=WorkLocation.INPUT if work.from_input else WorkLocation.WORK_IN_PROGRESS,
		)

	def mark_active_input_as_work_in_progress(self) -> None:
		if self.active_work is None or self.active_work.location is not WorkLocation.INPUT:
			return
		try:
			self.inbound_parts.remove(self.active_work.product_id)
		except ValueError:
			pass
		self.active_work.location = WorkLocation.WORK_IN_PROGRESS

	def prepare_handoff(self, product_id: str | None = None) -> None:
		if self.active_work is None:
			if product_id is None:
				return
			self.active_work = ActiveWorkItem(product_id=product_id, schedule_entry=None, location=WorkLocation.HANDOFF)
			return
		self.active_work.schedule_entry = None
		self.active_work.location = WorkLocation.HANDOFF

	def clear_active_work(self) -> None:
		self.active_work = None

	def restore_work_item(self) -> None:
		work = self.active_work
		if work is None or work.schedule_entry is None:
			return
		self.production_schedule.insert(0, work.schedule_entry)
		if work.location is WorkLocation.INPUT:
			self.active_work = None
			return
		if work.location is WorkLocation.WORK_IN_PROGRESS:
			work.schedule_entry = None
			return
		self.active_work = None

	def active_output_is_complete(self, outgoing_belt: BeltSegment | None) -> bool:
		if self.active_work is None:
			return False
		return self.active_work.location is WorkLocation.HANDOFF or (outgoing_belt is not None and outgoing_belt.has_queued_part(self.active_work.product_id))
