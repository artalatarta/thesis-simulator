from cps.core.network import (
	NETWORK_LATENCY as NETWORK_LATENCY,
	PACKET_LOSS_RETRY as PACKET_LOSS_RETRY,
)

ACTUATOR_SLOW_RESPONSE = "actuator_slow_response"
ACTUATOR_STUCK = "actuator_stuck"
BELT_CAPACITY = "belt_capacity"
BELT_QUEUE_PRESSURE = "belt_queue_pressure"
DEAD_BATTERY = "dead_battery"
DOWNSTREAM_ACTUATOR_SLOW_RESPONSE = "downstream_actuator_slow_response"
DOWNSTREAM_CAPACITY_PRESSURE = "downstream_capacity_pressure"
DOWNSTREAM_INPUT_CAPACITY = "downstream_input_capacity"
DOWNSTREAM_REJECTED = "downstream_rejected"
HANDOFF_UNCONFIRMED = "handoff_unconfirmed"
INPUT_CAPACITY = "input_capacity"
NETWORK_OR_BELT_CAPACITY = "network_or_belt_capacity"
OUTPUT_CAPACITY = "output_capacity"
PARTIAL_CONGESTION = "partial_congestion"
THERMAL_BLOCKED = "thermal_blocked"

BLOCKING_MACHINE_REASONS = {DEAD_BATTERY, THERMAL_BLOCKED, ACTUATOR_STUCK, INPUT_CAPACITY}
NETWORK_DELAY_REASONS = {NETWORK_LATENCY, PACKET_LOSS_RETRY}
