"""Direct unit tests for the deterministic classification source of truth."""

import pytest

from cps.agents.diagnosis import diagnosis_and_action_for_identifier
from cps.evaluation.event_records import agent_role_for_identifier


@pytest.mark.parametrize(
	("identifier", "diagnosis_id", "diagnosis", "action"),
	[
		(
			"sensor:M1:Power:sensor_stuck_detected",
			"sensor:M1:Power:stuck",
			"stuck",
			"fix_stuck",
		),
		(
			"sensor:M1:Power:low_battery_detected",
			"battery:M1:low_battery",
			"low_battery",
			"replace_battery",
		),
		(
			"sensor:M1:Power:dead_battery_detected",
			"battery:M1:dead_battery",
			"dead_battery",
			"replace_battery",
		),
		(
			"sensor:M2:Temperature:overheating_detected",
			"temperature:M2:overheating",
			"overheating",
			"start_cooling",
		),
		(
			"sensor:M2:Temperature:critical_overheating_detected",
			"temperature:M2:critical_overheating",
			"critical_overheating",
			"start_intense_cooling",
		),
		(
			"sensor:M2:Temperature:sensor_stuck_detected",
			"sensor:M2:Temperature:stuck",
			"stuck",
			"fix_stuck",
		),
		(
			"sensor:M3:ActuatorSensor:actuator_stuck_detected",
			"actuator:M3:stuck",
			"stuck",
			"fix_stuck",
		),
		(
			"sensor:M3:ActuatorSensor:actuator_slow_response_detected",
			"actuator:M3:slow_response",
			"slow_response",
			"fix_slow_response",
		),
		(
			"sensor:M3:ActuatorSensor:sensor_no_signal_detected",
			"sensor:M3:ActuatorSensor:no_signal",
			"no_signal",
			"fix_no_signal",
		),
		(
			"sensor:M2:Temperature:stuck",
			"sensor:M2:Temperature:stuck",
			"stuck",
			"fix_stuck",
		),
		(
			"actuator:M3:slow_response",
			"actuator:M3:slow_response",
			"slow_response",
			"fix_slow_response",
		),
		(
			"network:network_latency_detected",
			"network:latency",
			"latency",
			"fix_latency",
		),
		(
			"network:network_packet_loss_detected",
			"network:packet_loss",
			"packet_loss",
			"fix_packet_loss",
		),
		(
			"machine:M1:bearing_wear_detected",
			"machine:M1:bearing_wear",
			"bearing_wear",
			"fix_bearing_wear",
		),
		(
			"machine:M1:jammed_workpiece_detected",
			"machine:M1:jammed_workpiece",
			"jammed_workpiece",
			"fix_jammed_workpiece",
		),
		(
			"belt:M1:M2:belt_slippage_detected",
			"belt:M1:M2:belt_slippage",
			"belt_slippage",
			"fix_belt_slippage",
		),
		(
			"belt:M1:M2:belt_jam_detected",
			"belt:M1:M2:belt_jam",
			"belt_jam",
			"fix_belt_jam",
		),
	],
)
def test_diagnosis_and_action_for_identifier(identifier: str, diagnosis_id: str, diagnosis: str, action: str) -> None:
	assert diagnosis_and_action_for_identifier(identifier) == (diagnosis_id, diagnosis, action)


def test_overheating_escalates_to_critical_for_flagged_machines() -> None:
	assert diagnosis_and_action_for_identifier(
		"sensor:M1:Temperature:overheating_detected",
		critical_overheating_ids={"M1"},
	) == ("temperature:M1:critical_overheating", "critical_overheating", "start_intense_cooling")


def test_unmapped_identifier_is_unknown_and_waits() -> None:
	assert diagnosis_and_action_for_identifier("mystery:event") == (None, "unknown", "wait_for_more_evidence")


@pytest.mark.parametrize(
	("identifier", "expected_role"),
	[
		("sensor:M1:Power:sensor_stuck_detected", "power"),
		("sensor:M1:Power:low_battery_detected", "power"),
		("battery:M1:low_battery", "power"),
		("sensor:M2:Temperature:overheating_detected", "temperature"),
		("temperature:M2:overheating", "temperature"),
		("sensor:M3:ActuatorSensor:actuator_stuck_detected", "actuator"),
	],
)
def test_agent_role_for_identifier_uses_specific_measurement_roles(identifier: str, expected_role: str) -> None:
	assert agent_role_for_identifier(identifier) == expected_role
