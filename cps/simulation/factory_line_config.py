from dataclasses import dataclass

from cps.types import ScheduleEntry

type Station = tuple[str, float]


@dataclass(frozen=True)
class FactoryLineConfig:
	product: str
	quantity: int
	stations: tuple[Station, ...]
	source_id: str = "RawMaterialSource"
	storage_id: str = "FinalStorage"

	def schedule_for(self, station: Station) -> tuple[ScheduleEntry, ...]:
		_, process_time = station
		return tuple((f"{self.product}-{index:03d}", process_time) for index in range(1, self.quantity + 1))
