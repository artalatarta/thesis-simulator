from dataclasses import dataclass
from functools import lru_cache

from cps.agents.fault_catalog import (
	ACTUATOR_SENSOR_TYPE,
	BELT_FAULT_IDS,
	BELT_OBSERVATION_FAULTS,
	MACHINE_FAULT_IDS,
	MACHINE_OBSERVATION_FAULTS,
	MEASUREMENT_SENSOR_TYPES,
	SENSOR_OBSERVATION_FAULTS,
)

ACTUATOR_STATUS_OBSERVATIONS = frozenset({"actuator_stuck_detected", "actuator_slow_response_detected"})
BATTERY_OBSERVATIONS = frozenset({"dead_battery_detected", "low_battery_detected"})
TEMPERATURE_OBSERVATIONS = frozenset({"critical_overheating_detected", "overheating_detected"})
NETWORK_OBSERVATIONS = frozenset({"network:network_latency_detected", "network:network_packet_loss_detected"})
MACHINE_FAULT_OBSERVATIONS = frozenset(MACHINE_OBSERVATION_FAULTS)
BELT_FAULT_OBSERVATIONS = frozenset(BELT_OBSERVATION_FAULTS)
MACHINE_PRODUCTION_ISSUES = frozenset({"production_blocked", "production_slowdown"})
BELT_PRODUCTION_ISSUES = frozenset({"handoff_blocked", "transfer_rate_degraded", "persistent_queue_pressure"})
HIDDEN_PHYSICAL_STATE_KINDS = frozenset({"battery", "temperature"})


@dataclass(frozen=True)
class ParsedIdentifier:
	raw: str
	kind: str
	parts: tuple[str, ...]

	@property
	def machine_id(self) -> str | None:
		if self.kind in {"sensor", "actuator", "battery", "temperature", "machine"} and len(self.parts) >= 2:
			return self.parts[1]
		if self.kind == "belt" and len(self.parts) >= 3:
			return f"{self.parts[1]}->{self.parts[2]}"
		return None

	@property
	def sensor_type(self) -> str | None:
		return self.parts[2] if self.kind == "sensor" and len(self.parts) >= 3 else None

	@property
	def observation(self) -> str | None:
		return self.parts[3] if len(self.parts) >= 4 else None

	@property
	def state_or_issue(self) -> str | None:
		return self.parts[2] if len(self.parts) >= 3 else None

	@property
	def from_node_id(self) -> str | None:
		return self.parts[1] if self.kind == "belt" and len(self.parts) >= 3 else None

	@property
	def to_node_id(self) -> str | None:
		return self.parts[2] if self.kind == "belt" and len(self.parts) >= 3 else None

	@property
	def network_fault(self) -> str | None:
		return self.parts[1] if self.kind == "network" and len(self.parts) == 2 else None

	@property
	def is_measurement_sensor(self) -> bool:
		return self.kind == "sensor" and self.sensor_type in MEASUREMENT_SENSOR_TYPES

	@property
	def is_actuator_sensor(self) -> bool:
		return self.kind == "sensor" and self.sensor_type == ACTUATOR_SENSOR_TYPE

	@property
	def is_sensor_fault_observation(self) -> bool:
		if len(self.parts) != 4 or self.kind != "sensor":
			return False
		if self.is_actuator_sensor:
			return self.observation == "sensor_no_signal_detected"
		return self.is_measurement_sensor and self.observation in SENSOR_OBSERVATION_FAULTS

	@property
	def is_actuator_status_observation(self) -> bool:
		return self.kind == "sensor" and len(self.parts) == 4 and self.observation in ACTUATOR_STATUS_OBSERVATIONS

	@property
	def is_battery_observation(self) -> bool:
		return self.kind == "sensor" and len(self.parts) == 4 and self.observation in BATTERY_OBSERVATIONS

	@property
	def is_temperature_observation(self) -> bool:
		return self.kind == "sensor" and len(self.parts) == 4 and self.observation in TEMPERATURE_OBSERVATIONS

	@property
	def is_network_observation(self) -> bool:
		return self.raw in NETWORK_OBSERVATIONS

	@property
	def is_critical_overheating_state(self) -> bool:
		return len(self.parts) == 3 and self.kind == "temperature" and self.state_or_issue == "critical_overheating"

	@property
	def is_production_flow_issue(self) -> bool:
		if self.is_machine_production_issue:
			return self.state_or_issue in MACHINE_PRODUCTION_ISSUES
		if self.is_belt_production_issue:
			return self.observation in BELT_PRODUCTION_ISSUES
		return False

	@property
	def is_machine_fault(self) -> bool:
		return len(self.parts) == 3 and self.kind == "machine" and self.state_or_issue in MACHINE_FAULT_IDS

	@property
	def is_machine_fault_observation(self) -> bool:
		return len(self.parts) == 3 and self.kind == "machine" and self.state_or_issue in MACHINE_FAULT_OBSERVATIONS

	@property
	def is_belt_fault(self) -> bool:
		return len(self.parts) == 4 and self.kind == "belt" and self.observation in BELT_FAULT_IDS

	@property
	def is_belt_fault_observation(self) -> bool:
		return len(self.parts) == 4 and self.kind == "belt" and self.observation in BELT_FAULT_OBSERVATIONS

	@property
	def is_machine_production_issue(self) -> bool:
		return len(self.parts) == 3 and self.kind == "machine" and self.state_or_issue in MACHINE_PRODUCTION_ISSUES

	@property
	def is_belt_production_issue(self) -> bool:
		return len(self.parts) == 4 and self.kind == "belt" and self.observation in BELT_PRODUCTION_ISSUES

	@property
	def is_hidden_physical_state(self) -> bool:
		return len(self.parts) == 3 and self.kind in HIDDEN_PHYSICAL_STATE_KINDS


@lru_cache(maxsize=4096)
def parse_identifier(identifier: str) -> ParsedIdentifier:
	parts = tuple(identifier.split(":"))
	return ParsedIdentifier(raw=identifier, kind=parts[0] if parts else "", parts=parts)


def machine_id_from_identifier(identifier: str) -> str | None:
	return parse_identifier(identifier).machine_id
