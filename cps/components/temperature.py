from cps.core.reporting import temperature_state_id
from cps.types import CoolingState


class Temperature:
	def __init__(
		self,
		machine_id: str,
		value: float = 25.0,
		heating_rate: float = 2.0,
		idle_cooling_rate: float = 0.5,
		light_cooling_rate: float = 0.3,
		intense_cooling_rate: float = 4.0,
		warning_threshold: float = 90.0,
		critical_threshold: float = 100.0,
		shutdown_threshold: float = 110.0,
		safe_threshold: float = 75.0,
		ambient_temperature: float = 25.0,
	) -> None:
		self.machine_id = machine_id
		self.value = value
		self.heating_rate = heating_rate
		self.idle_cooling_rate = idle_cooling_rate
		self.light_cooling_rate = light_cooling_rate
		self.intense_cooling_rate = intense_cooling_rate
		self.warning_threshold = warning_threshold
		self.critical_threshold = critical_threshold
		self.shutdown_threshold = shutdown_threshold
		self.safe_threshold = safe_threshold
		self.ambient_temperature = ambient_temperature
		self.cooling_state: CoolingState = CoolingState.NONE
		self._shutdown_latched = False

	@property
	def is_overheating(self) -> bool:
		return self.warning_threshold <= self.value < self.critical_threshold

	@property
	def is_critical(self) -> bool:
		return self.value >= self.critical_threshold

	@property
	def is_shutdown(self) -> bool:
		"""Latched safety cutoff: once the temperature reaches ``shutdown_threshold``
		the machine stops producing and stays stopped until it cools back to
		``safe_threshold``, independently of any monitoring agent."""
		return self._shutdown_latched

	@property
	def is_safe(self) -> bool:
		return self.value <= self.safe_threshold

	@property
	def state_id(self) -> str | None:
		if self.is_critical or self.is_shutdown:
			return temperature_state_id(self.machine_id, "critical_overheating")
		if self.is_overheating:
			return temperature_state_id(self.machine_id, "overheating")
		return None

	@property
	def is_thermal_blocked(self) -> bool:
		return self._shutdown_latched or self.cooling_state is CoolingState.INTENSE

	@property
	def process_time_factor(self) -> float:
		if self.cooling_state is CoolingState.LIGHT:
			return 1.5
		return 1.0

	def start_light_cooling(self) -> bool:
		if self.cooling_state is not CoolingState.NONE:
			return True
		if not (self.is_overheating or self.is_critical):
			return False
		self.cooling_state = CoolingState.LIGHT
		return True

	def start_intense_cooling(self) -> bool:
		if self.cooling_state is CoolingState.INTENSE:
			return True
		self.cooling_state = CoolingState.INTENSE
		return True

	def stop_cooling(self) -> None:
		self.cooling_state = CoolingState.NONE

	def update(self, *, is_processing: bool) -> None:
		if self.cooling_state is CoolingState.INTENSE:
			delta = -self.intense_cooling_rate
		elif self.cooling_state is CoolingState.LIGHT and is_processing:
			delta = -self.light_cooling_rate
		elif is_processing:
			delta = self.heating_rate
		else:
			# An idle machine sheds heat at the idle rate even under light cooling,
			# which would otherwise be slower than not cooling at all.
			delta = -self.idle_cooling_rate
		self.value = max(self.ambient_temperature, self.value + delta)
		if self.value >= self.shutdown_threshold:
			self._shutdown_latched = True
		elif self.value <= self.safe_threshold:
			self._shutdown_latched = False
