"""Fault catalog identifier rules and deterministic defaults."""

from collections.abc import Iterable

from cps.components.actuators import ACTUATOR_FAULT_TYPES as ACTUATOR_COMPONENT_FAULT_TYPES
from cps.components.sensors import SENSOR_FAULT_TYPES
from cps.core.network import NETWORK_FAULT_TYPES as NETWORK_CORE_FAULT_TYPES

MEASUREMENT_SENSOR_FAULT_TYPES = set(SENSOR_FAULT_TYPES)
ACTUATOR_SENSOR_FAULT_TYPES = {"no_signal"}
ACTUATOR_FAULT_TYPES = set(ACTUATOR_COMPONENT_FAULT_TYPES)
NETWORK_FAULT_TYPES = set(NETWORK_CORE_FAULT_TYPES)
MEASUREMENT_SENSOR_TYPES = {"Power", "Temperature"}
ACTUATOR_SENSOR_TYPE = "ActuatorSensor"
BATTERY_STATE_IDS = {"low_battery", "dead_battery"}
TEMPERATURE_STATE_IDS = {"overheating", "critical_overheating"}
MACHINE_ISSUE_IDS = {"production_slowdown", "production_blocked"}
MACHINE_FAULT_IDS = {"bearing_wear", "jammed_workpiece"}
BELT_ISSUE_IDS = {"handoff_blocked", "persistent_queue_pressure", "transfer_rate_degraded"}
BELT_FAULT_IDS = {"belt_slippage", "belt_jam"}

# Each clearable root fault has exactly one action named after it; the handler
# clears only that fault type, so a mismatched recommendation is a no-op.
CLEAR_ACTION_BY_FAULT_TYPE = {
	"bearing_wear": "fix_bearing_wear",
	"jammed_workpiece": "fix_jammed_workpiece",
	"belt_slippage": "fix_belt_slippage",
	"belt_jam": "fix_belt_jam",
}


SENSOR_OBSERVATION_FAULTS = {
	"sensor_no_signal_detected": "no_signal",
	"sensor_stuck_detected": "stuck",
}
ACTUATOR_OBSERVATION_FAULTS = {
	"actuator_stuck_detected": "stuck",
	"actuator_slow_response_detected": "slow_response",
}
BATTERY_OBSERVATION_STATES = {
	"dead_battery_detected": "dead_battery",
	"low_battery_detected": "low_battery",
}
TEMPERATURE_OBSERVATION_STATES = {
	"critical_overheating_detected": "critical_overheating",
	"overheating_detected": "overheating",
}
NETWORK_OBSERVATION_FAULTS = {
	"network:network_latency_detected": "latency",
	"network:network_packet_loss_detected": "packet_loss",
}
MACHINE_OBSERVATION_FAULTS = {
	"bearing_wear_detected": "bearing_wear",
	"jammed_workpiece_detected": "jammed_workpiece",
}
BELT_OBSERVATION_FAULTS = {
	"belt_slippage_detected": "belt_slippage",
	"belt_jam_detected": "belt_jam",
}


def _id_options(values: Iterable[str]) -> str:
	return "{" + "|".join(sorted(values)) + "}"


def measurement_sensor_diagnosis_id_template(sensor_type: str) -> str:
	return f"sensor:<machine_id>:{sensor_type}:{_id_options(MEASUREMENT_SENSOR_FAULT_TYPES)}"


def actuator_sensor_diagnosis_id_template() -> str:
	return f"sensor:<machine_id>:{ACTUATOR_SENSOR_TYPE}:{_id_options(ACTUATOR_SENSOR_FAULT_TYPES)}"


def actuator_diagnosis_id_template() -> str:
	return f"actuator:<machine_id>:{_id_options(ACTUATOR_FAULT_TYPES)}"


def battery_diagnosis_id_template() -> str:
	return f"battery:<machine_id>:{_id_options(BATTERY_STATE_IDS)}"


def temperature_diagnosis_id_template() -> str:
	return f"temperature:<machine_id>:{_id_options(TEMPERATURE_STATE_IDS)}"


def machine_diagnosis_id_template() -> str:
	return f"machine:<machine_id>:{_id_options(MACHINE_ISSUE_IDS | MACHINE_FAULT_IDS)}"


def belt_diagnosis_id_template() -> str:
	return f"belt:<from_node>:<to_node>:{_id_options(BELT_ISSUE_IDS | BELT_FAULT_IDS)}"


def network_diagnosis_id_template() -> str:
	return f"network:{_id_options(NETWORK_FAULT_TYPES)}"


def is_fault_catalog_diagnosis_id(identifier: str) -> bool:
	parts = identifier.split(":")
	if len(parts) == 2:
		return parts[0] == "network" and parts[1] in NETWORK_FAULT_TYPES
	if len(parts) == 3:
		return _valid_three_part_catalog_id(parts)
	if len(parts) == 4:
		return _valid_four_part_catalog_id(parts)
	return False


def _valid_three_part_catalog_id(parts: list[str]) -> bool:
	kind = parts[0]
	value = parts[2]
	allowed_by_kind = {
		"actuator": ACTUATOR_FAULT_TYPES,
		"battery": BATTERY_STATE_IDS,
		"temperature": TEMPERATURE_STATE_IDS,
		"machine": MACHINE_ISSUE_IDS | MACHINE_FAULT_IDS,
	}
	allowed = allowed_by_kind.get(kind)
	return allowed is not None and value in allowed


def _valid_four_part_catalog_id(parts: list[str]) -> bool:
	kind = parts[0]
	if kind == "sensor":
		sensor_type = parts[2]
		fault = parts[3]
		if sensor_type == ACTUATOR_SENSOR_TYPE:
			return fault in ACTUATOR_SENSOR_FAULT_TYPES
		return sensor_type in MEASUREMENT_SENSOR_TYPES and fault in MEASUREMENT_SENSOR_FAULT_TYPES
	if kind == "belt":
		return parts[3] in BELT_ISSUE_IDS | BELT_FAULT_IDS
	return False
