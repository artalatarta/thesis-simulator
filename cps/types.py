from collections.abc import Generator
from enum import Enum
from typing import Any
from typing import Literal, TypeAlias, TypedDict

from simpy.events import Event


class CoolingState(Enum):
	"""Cooling response currently applied to a machine.

	``LIGHT`` cooling slows production; ``INTENSE`` cooling blocks it. Production
	is also blocked by the latched safety cutoff at ``Temperature.shutdown_threshold``;
	``Machine.thermal_blocked`` is derived from both.
	"""

	NONE = "none"
	LIGHT = "light"
	INTENSE = "intense"


SensorFaultType: TypeAlias = Literal["stuck", "no_signal"]
# "already_resolved" means the action was justified by its report but the
# condition it addresses resolved before the action needed to mutate state.
ActionOutcome: TypeAlias = Literal["succeeded", "failed", "already_resolved"]
ActuatorFaultType: TypeAlias = Literal["slow_response", "stuck"]
NetworkFaultType: TypeAlias = Literal["latency", "packet_loss"]
ScheduleEntry: TypeAlias = tuple[str, float]
ProcessGenerator: TypeAlias = Generator[Event, object, Any]


class ProcessingTimeState(TypedDict):
	total_processing_time: float
	last_change_time: float
	is_processing: bool


class MachineStatus(TypedDict):
	is_processing: bool
	battery_level: float
	temperature: float
	temperature_state: str | None
	parts_produced: int
	current_product: str | None
