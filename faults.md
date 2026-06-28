# Faults, Observations, and Derived Issues

Fault model for the CPS simulation. It separates injected root faults from local observations, physical states, and derived production effects.

## Vocabulary

| Term              | Meaning                                                                                                                                                                                         | Example                                                                                                                                                                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Root fault        | Underlying failure injected into the simulation.                                                                                                                                                | `sensor:<machine_id>:Power:stuck`, `actuator:<machine_id>:stuck`, `network:packet_loss`, `machine:<machine_id>:bearing_wear`, `belt:<from_node_id>:<to_node_id>:belt_jam`                                                                                    |
| Physical state    | Battery, temperature.                                                                                                                                                                           | `battery:<machine_id>:low_battery`, `battery:<machine_id>:dead_battery`, `temperature:<machine_id>:overheating`, `temperature:<machine_id>:critical_overheating`                                                                                             |
| Derived issue     | Machine or belt issue inferred from root faults or, physical states.                                                                                                                            | `machine:<machine_id>:production_blocked`, `belt:<from_node_id>:<to_node_id>:handoff_blocked`, `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure`                                                                                                        |
| Local observation | Direct observation produced by a component. Some observations report sensor or network faults; others report physical battery/temperature state or actuator condition through a working sensor. | `sensor:<machine_id>:Power:sensor_stuck_detected`, `sensor:<machine_id>:Power:low_battery_detected`, `sensor:<machine_id>:Power:dead_battery_detected`, `sensor:<machine_id>:ActuatorSensor:actuator_stuck_detected`, `network:network_packet_loss_detected` |

## Identifier forms

| Kind                              | Identifier form                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| Sensor root fault                 | `sensor:<machine_id>:<sensor_type>:<fault>` or `sensor:<machine_id>:ActuatorSensor:no_signal`   |
| Actuator root fault               | `actuator:<machine_id>:<fault>`                                                                 |
| Network root fault                | `network:<fault>`                                                                               |
| Machine root fault                | `machine:<machine_id>:bearing_wear` or `machine:<machine_id>:jammed_workpiece`                  |
| Belt root fault                   | `belt:<from_node_id>:<to_node_id>:belt_slippage` or `belt:<from_node_id>:<to_node_id>:belt_jam` |
| Battery state                     | `battery:<machine_id>:<state>`                                                                  |
| Temperature state                 | `temperature:<machine_id>:<state>`                                                              |
| Sensor observation                | `sensor:<machine_id>:<sensor_type>:<observation>`                                               |
| Actuator sensor observation       | `sensor:<machine_id>:ActuatorSensor:<observation>`                                              |
| Network observation               | `network:<observation>`                                                                         |
| Machine derived issue observation | `machine:<machine_id>:<issue>`                                                                  |
| Belt derived issue observation    | `belt:<from_node_id>:<to_node_id>:<issue>`                                                      |

`bottleneck` is not a root fault. `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` is a derived diagnosis from  persistent production flow symptoms, not the underlying physical state itself. Trace each bottleneck diagnosis to concrete upstream, downstream, network, or capacity causes.

Parts wait on an incoming belt, are moved into the machine by the machine actuator, are processed by the machine, and are then handed off to an outgoing belt. For a three-machine line, the material path is `machine1Actuator -> machine1 -> belt:machine1:machine2 -> machine2Actuator -> machine2 -> belt:machine2:machine3 -> machine3Actuator -> machine3 -> belt:machine3:FinalStorage -> FinalStorage`. This model assumes one actuator per machine, so actuator identifiers do not include an actuator type segment.

Machines can hand off completed output to the outgoing belt when that belt has available capacity. A blocked or saturated outgoing belt prevents handoff and can block upstream production.

Once a machine has completed processing a part, that part is completed output waiting for outgoing handoff. A blocked handoff, maintenance reboot, or interrupted production process must not put that part back into the machine's input side or production schedule. Recovery resumes the outgoing handoff for the completed output already at the machine output or on the outgoing belt.

Production recovery must treat machine processing and handoff subprocesses as owned work. If a maintenance reboot interrupts a machine, the active subprocess is also interrupted so it cannot later finish in the background after its schedule entry has been restored. The runtime may restart an idle-but-live production process when pending inbound work is stranded behind a stale wake event; this is a recovery action, not a new product source.

Use `belt:<from_node_id>:<to_node_id>:handoff_blocked` when material or completion flow cannot move from one node to the next node. This can mean a physical downstream blockage or that control/coordination did not confirm the handoff. A machine is one type of node. `FinalStorage` is the final storage node.

For simplicity, `FinalStorage` has infinite capacity and no local issues. It is not a machine, so it should not emit `machine:<machine_id>:...` issues. The final handoff to `FinalStorage` can still be affected by shared network faults.

The belt segment leading to `FinalStorage` may still have finite conveyor capacity. If that segment is full, diagnostics should attribute blockage to belt capacity, not to storage capacity or a `FinalStorage` machine issue.

There is one shared network. Network faults are global root faults, so network latency or packet loss can affect handoff coordination and throughput across the whole belt, including the final handoff to `FinalStorage`. Derived belt issues are still emitted per affected segment where symptoms are observed.

Partial downstream capacity constraints outside `FinalStorage` can slow production while some handoffs still succeed. Full downstream capacity constraints outside `FinalStorage` block handoff and can eventually block the upstream machine. We emit `transfer_rate_degraded` for measured reduced flow, `handoff_blocked` for failed movement or unconfirmed transfer, and `persistent_queue_pressure` only after symptoms persist long enough to distinguish a durable capacity constraint from a transient delay.

## Sensor Model

All sensors share a base `Sensor` abstraction for identity, active fault state, and fault clearing. Specific sensor types then define what they can measure and which fault modes are meaningful.

| Sensor class | Responsibility | Applicable sensor faults |
|---|---|---|
| `Sensor` | Base abstraction for all sensors. Owns common identity, `no_signal` state, and fault lifecycle. | `no_signal` |
| `MeasurementSensor` | Common base for sensors that read a scalar machine state value. | `stuck`, `no_signal` |
| `PowerSensor` | Extends `MeasurementSensor`; reads `Battery.level` and observes low or dead battery state. | `stuck`, `no_signal` |
| `TemperatureSensor` | Extends `MeasurementSensor`; reads `Temperature.value` and observes overheating state. | `stuck`, `no_signal` |
| `ActuatorSensor` | Extends `Sensor`; observes actuator execution status rather than a scalar measurement. | `no_signal` |

`no_signal` is the common sensor fault because any sensor can lose telemetry. `stuck` also applies to `MeasurementSensor` subclasses because those sensors read numeric state values. It does not apply to `ActuatorSensor` in this model because actuator detection is based on execution status, not a continuous actuator measurement.

The current simulator detects measurement sensor faults by comparing measured values with simulator ground truth. Treat that as a simulation diagnostic oracle, not as a claim that a deployed local sensor can directly know the true physical value.

## Battery Model

Battery level is not an injected root fault. Batteries drain while machines operate. A low or dead battery becomes relevant when it is detected or missed because Power sensor readings are ignored during an active sensor fault.

Separate battery state from machine state:

| Component | Responsibility | Fault or issue relation |
|---|---|---|
| Battery | Owns charge level, drain rate, low threshold, depleted threshold, and battery states such as `battery:<machine_id>:low_battery` and `battery:<machine_id>:dead_battery`. | Not directly fault-injected in this plan; changes through operation and maintenance. |
| Machine | Consumes battery while operating and stops when the battery state is `battery:<machine_id>:dead_battery`. | Dead battery derives `machine:<machine_id>:production_blocked`. |
| Power sensor | Observes battery state and reports battery-related readings or observations. | Can report `sensor:<machine_id>:Power:low_battery_detected` or `sensor:<machine_id>:Power:dead_battery_detected` only when it has no active fault; readings are ignored while a Power sensor fault is active. |
| Maintenance | Replaces battery after a low-battery or dead-battery observation from a working Power sensor, or after a scheduled maintenance decision. | Ignored Power sensor readings can delay battery replacement until the Power sensor fault is cleared. |

## Temperature Model

Temperature is machine-local physical state, similar to battery charge. It is influenced by continuous operation rather than being injected as a root fault. The Temperature sensor only observes temperature state.

Separate temperature state from machine state:

| Component                   | Responsibility                                                                                                                                                                                           | Fault or issue relation                                                                                                                                           |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Temperature                 | Owns current temperature, heating rate, warning threshold, critical threshold, and thermal states such as `temperature:<machine_id>:overheating` and `temperature:<machine_id>:critical_overheating`. | Not directly fault-injected in this plan; rises during long continuous operation.                                                                                  |
| Machine                     | Operates while temperature is acceptable, slows down for light cooling when temperature exceeds the normal operating range, and stops for intense cooling when temperature reaches the critical state.                            | Out-of-range temperature derives `machine:<machine_id>:production_slowdown`; critical overheating derives `machine:<machine_id>:production_blocked`.                    |
| Temperature sensor          | Observes temperature state and reports temperature-related readings or observations.                                                                                                                     | Can report `sensor:<machine_id>:Temperature:overheating_detected` or `sensor:<machine_id>:Temperature:critical_overheating_detected` only when it has no active fault; readings are ignored while a Temperature sensor fault is active. |
| Cooling action              | Reduces temperature by lowering throughput for light cooling or stopping production for intense cooling.                                                                                                 | Missed overheating information can allow thermal state to reach `temperature:<machine_id>:critical_overheating`.                                                  |

## Actuator Model

Actuator condition is modeled as an injected mechanical root fault, while a dedicated `ActuatorSensor` observes that condition. The agent should not treat an actuator's internal condition as a direct observation unless it comes through a working `ActuatorSensor`.

Separate actuator faults from machine state:

| Component | Responsibility | Fault or issue relation |
|---|---|---|
| Actuator | Owns mechanical execution state and action timing. Injected actuator faults use ids such as `actuator:<machine_id>:slow_response` and `actuator:<machine_id>:stuck`. | Mechanical actuator faults are root faults, not physical-state ground-truth rows. |
| Machine | Uses the actuator to accept input from the incoming belt. Slow actuator movement reduces intake rate; stuck actuator movement prevents intake. | Slow response derives `machine:<machine_id>:production_slowdown`; stuck faults derive `machine:<machine_id>:production_blocked`. |
| ActuatorSensor | Observes actuator execution status and reports actuator-related observations. | Can report `sensor:<machine_id>:ActuatorSensor:actuator_slow_response_detected` or `sensor:<machine_id>:ActuatorSensor:actuator_stuck_detected` only when it has no active `no_signal` fault; readings are ignored while the `ActuatorSensor` has no signal. |
| Maintenance or recalibration action | Clears a mechanical actuator fault after a valid actuator observation or scheduled maintenance decision. | Ignored `ActuatorSensor` readings can delay recalibration, allowing slowdown or blockage symptoms to persist. |

## Root Fault Catalog

| Domain   | Fault           | Local observations                                                      | Derived production issues                                                                              | Notes                                                                                                                                                                                                                                                                                              |
| -------- | --------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sensor | `no_signal` | `sensor:<machine_id>:<sensor_type>:sensor_no_signal_detected` or `sensor:<machine_id>:ActuatorSensor:sensor_no_signal_detected` | None directly | Missing telemetry. Can cause missed battery replacement, missed temperature cooling, or delayed actuator recalibration if unresolved. |
| MeasurementSensor | `stuck`     | `sensor:<machine_id>:<sensor_type>:sensor_stuck_detected`               | None directly                                                                                          | Numeric reading freezes at one value until resolved or overwritten. Applies to `PowerSensor` and `TemperatureSensor`; can cause missed battery replacement or temperature cooling if unresolved. |
| Actuator | `slow_response` | `sensor:<machine_id>:ActuatorSensor:actuator_slow_response_detected` | `machine:<machine_id>:production_slowdown`                                                             | Action completes late while moving products from the incoming belt into the machine. The incoming belt segment can first show reduced flow as `belt:<from_node_id>:<machine_id>:transfer_rate_degraded`. This observation is not emitted from a faulty `ActuatorSensor`.                                                                                                  |
| Actuator | `stuck`         | `sensor:<machine_id>:ActuatorSensor:actuator_stuck_detected`         | `machine:<machine_id>:production_blocked`                                                              | Hard block while moving products from the incoming belt into the machine. If the incoming belt fills, the preceding node can later be blocked from handing off to that belt. Recovery may require recalibration and process reboot. This observation is not emitted from a faulty `ActuatorSensor`.                                                                |
| Network  | `latency`       | `network:network_latency_detected`                                      | `belt:<from_node_id>:<to_node_id>:transfer_rate_degraded`, `belt:<from_node_id>:<to_node_id>:handoff_blocked` | Delayed status reads, coordination messages, and handoff notifications across the shared network. Mild latency reduces throughput; severe or sustained latency can block handoff. If the upstream node is a machine, blocked handoff can later derive `machine:<from_node_id>:production_blocked`. |
| Network  | `packet_loss`   | `network:network_packet_loss_detected`                                  | `belt:<from_node_id>:<to_node_id>:handoff_blocked`, `belt:<from_node_id>:<to_node_id>:transfer_rate_degraded` | Dropped reads or handoff notifications across the shared network can require retries, reduce throughput, or block machines from releasing completed output. If the upstream node is a machine, blocked handoff can later derive `machine:<from_node_id>:production_blocked`.                       |
| Machine | `bearing_wear` | `machine:<machine_id>:bearing_wear_detected` | `machine:<machine_id>:production_slowdown`, `belt:<from_node_id>:<machine_id>:transfer_rate_degraded` | Mechanical wear increases processing time without fully stopping the station. The machine-health monitor detects the worn component directly; clearing the machine fault represents maintenance replacing or servicing it. |
| Machine | `jammed_workpiece` | `machine:<machine_id>:jammed_workpiece_detected` | `machine:<machine_id>:production_blocked`, `belt:<from_node_id>:<machine_id>:handoff_blocked` | A part is physically jammed at the station, blocking intake or processing until maintenance clears the obstruction. The machine-health monitor detects the jam directly. |
| Belt | `belt_slippage` | `belt:<from_node_id>:<to_node_id>:belt_slippage_detected` | `belt:<from_node_id>:<to_node_id>:transfer_rate_degraded`, `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` | Belt traction loss slows transfer across the segment without completely blocking movement. The belt monitor detects the slippage directly. |
| Belt | `belt_jam` | `belt:<from_node_id>:<to_node_id>:belt_jam_detected` | `belt:<from_node_id>:<to_node_id>:handoff_blocked`, `machine:<from_node_id>:production_blocked` | A conveyor jam prevents transfer across the segment until maintenance clears it. The belt monitor detects the jam directly. |

## Local Observations

Local observations are reports produced by sensors, the shared network monitor, or the machine-health and belt monitors. Fault observations report sensor, network, machine, or belt faults. State observations report physical battery/temperature state or actuator condition observed through a working sensor.

This plan accepts battery/temperature state observations and actuator-condition observations only when the relevant sensor has no active fault. It does not model coincidental true positives from faulty sensors. If a sensor has an active fault, its readings are ignored for maintenance decisions. For `ActuatorSensor`, the only modeled sensor fault is `no_signal`.

| Observation | Meaning | Related state or fault |
|---|---|---|
| `sensor:<machine_id>:Power:sensor_stuck_detected` or `sensor:<machine_id>:Temperature:sensor_stuck_detected` | MeasurementSensor reading stops changing when it should vary. | MeasurementSensor `stuck` |
| `sensor:<machine_id>:<sensor_type>:sensor_no_signal_detected` or `sensor:<machine_id>:ActuatorSensor:sensor_no_signal_detected` | Sensor telemetry is missing. | Sensor `no_signal` |
| `sensor:<machine_id>:Power:low_battery_detected` | Power sensor observes `battery:<machine_id>:low_battery`. | Power sensor has no active fault and reports low charge. This observation is not emitted from a faulty Power sensor. |
| `sensor:<machine_id>:Power:dead_battery_detected` | Power sensor observes `battery:<machine_id>:dead_battery`. | Power sensor has no active fault and reports depleted charge. This observation is not emitted from a faulty Power sensor. |
| `sensor:<machine_id>:Temperature:overheating_detected` | Temperature sensor observes `temperature:<machine_id>:overheating`. | Temperature sensor has no active fault and reports overheating caused by long continuous operation. This observation is not emitted from a faulty Temperature sensor. |
| `sensor:<machine_id>:Temperature:critical_overheating_detected` | Temperature sensor observes `temperature:<machine_id>:critical_overheating`. | Temperature sensor has no active fault and reports critical overheating caused by long continuous operation or unresolved overheating. This observation is not emitted from a faulty Temperature sensor. |
| `sensor:<machine_id>:ActuatorSensor:actuator_slow_response_detected` | ActuatorSensor observes `actuator:<machine_id>:slow_response`. | ActuatorSensor has no active fault and reports delayed actuator movement. This observation is not emitted from a faulty ActuatorSensor. |
| `sensor:<machine_id>:ActuatorSensor:actuator_stuck_detected` | ActuatorSensor observes `actuator:<machine_id>:stuck`. | ActuatorSensor has no active fault and reports failed actuator movement. This observation is not emitted from a faulty ActuatorSensor. |
| `network:network_latency_detected` | Shared network latency exceeds the configured tolerance. | network `latency` |
| `network:network_packet_loss_detected` | Shared network drops messages or acknowledgements. | network `packet_loss` |
| `machine:<machine_id>:bearing_wear_detected` | Machine-health monitor detects worn mechanical components at the station. | machine `bearing_wear` |
| `machine:<machine_id>:jammed_workpiece_detected` | Machine-health monitor detects a workpiece jammed at the station. | machine `jammed_workpiece` |
| `belt:<from_node_id>:<to_node_id>:belt_slippage_detected` | Belt monitor detects traction loss on the segment. | belt `belt_slippage` |
| `belt:<from_node_id>:<to_node_id>:belt_jam_detected` | Belt monitor detects a conveyor jam on the segment. | belt `belt_jam` |

## System States and Derived Issues

| State or issue | Meaning | Possible causes |
|---|---|---|
| `battery:<machine_id>:low_battery` | Battery warning threshold is crossed; production can continue. | normal battery drain during operation |
| `battery:<machine_id>:dead_battery` | Battery is depleted. | missed battery replacement, ignored Power sensor readings |
| `temperature:<machine_id>:overheating` | Temperature is outside the safe range; production can continue at reduced rate for light cooling. | long continuous operation |
| `temperature:<machine_id>:critical_overheating` | Temperature has reached the critical threshold and the machine must stop for intense cooling. | unresolved overheating after long continuous operation, ignored Temperature sensor readings |
| `actuator:<machine_id>:slow_response` | Actuator completes movement but more slowly than expected. | injected mechanical actuator root fault |
| `actuator:<machine_id>:stuck` | Actuator cannot complete the expected movement. | injected mechanical actuator root fault |
| `machine:<machine_id>:production_slowdown` | Machine continues production but below expected rate. | `actuator:<machine_id>:slow_response`, `temperature:<machine_id>:overheating`, partial downstream capacity constraint, network retries that reduce effective handoff rate |
| `machine:<machine_id>:production_blocked` | Machine cannot continue production, cannot accept input, or cannot release completed output. Completed output remains at the machine output or on the outgoing belt until handoff succeeds. | `battery:<machine_id>:dead_battery`, `actuator:<machine_id>:stuck`, `temperature:<machine_id>:critical_overheating`, blocked downstream handoff, full input or output capacity constraint |
| `belt:<from_node_id>:<to_node_id>:handoff_blocked` | Material or completion flow cannot move from one node to the next node, either because the path is physically blocked or because handoff coordination is not confirmed. Nodes can include machines and `FinalStorage`. | downstream production block, packet loss, severe latency, downstream machine capacity reached, downstream queue saturation |
| `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` | A belt segment has persistent queue buildup, waiting, blocked handoff, or degraded throughput. Nodes can include machines and `FinalStorage`; for segments ending at `FinalStorage`, the cause is not storage capacity. | persistent downstream block, persistent machine slowdown, belt or downstream-machine capacity reached, queue saturation, repeated handoff blockage, repeated network handoff blockage |
| `belt:<from_node_id>:<to_node_id>:transfer_rate_degraded` | Completion rate across a specific belt segment falls below expectation. Nodes can include machines and `FinalStorage`; for segments ending at `FinalStorage`, the cause is not storage capacity. | slow upstream actuator, latency, packet-loss retries, overheating slowdown, partial downstream machine capacity constraint |

## Important Causal Chains

| Scenario | Fault relation |
|---|---|
| Power sensor fails and battery dies | `sensor:<machine_id>:Power:stuck` or `sensor:<machine_id>:Power:no_signal` -> Power sensor readings are ignored -> `battery:<machine_id>:low_battery` is missed -> missed battery replacement -> `battery:<machine_id>:dead_battery` -> `machine:<machine_id>:production_blocked` -> sensor is repaired -> `sensor:<machine_id>:Power:dead_battery_detected` -> maintenance replaces battery |
| Low battery is detected in time | normal battery drain during operation -> `battery:<machine_id>:low_battery` -> `sensor:<machine_id>:Power:low_battery_detected` -> maintenance can replace battery before `battery:<machine_id>:dead_battery` |
| Dead battery is detected after missed low battery | normal battery drain during operation -> `battery:<machine_id>:low_battery` is missed -> `battery:<machine_id>:dead_battery` -> `sensor:<machine_id>:Power:dead_battery_detected` -> maintenance replaces battery and production can restart |
| Actuator gets stuck | `actuator:<machine_id>:stuck`; if the ActuatorSensor has no active fault -> `sensor:<machine_id>:ActuatorSensor:actuator_stuck_detected` -> recalibration can be dispatched; regardless of detection, the machine cannot accept input -> `machine:<machine_id>:production_blocked`; if the incoming belt fills, the preceding node can no longer hand off into it -> `belt:<from_node_id>:<machine_id>:handoff_blocked`; if the blockage persists and causes queue buildup, waiting, or degraded throughput -> `belt:<from_node_id>:<machine_id>:persistent_queue_pressure` |
| Actuator sensor loses signal while a separate actuator fault persists | `sensor:<machine_id>:ActuatorSensor:no_signal` and a separate `actuator:<machine_id>:stuck` root fault -> ActuatorSensor status readings are ignored -> no recalibration is dispatched from sensor observation -> `machine:<machine_id>:production_blocked` persists until the sensor fault is cleared or scheduled maintenance intervenes |
| Actuator responds slowly | `actuator:<machine_id>:slow_response`; if the ActuatorSensor has no active fault -> `sensor:<machine_id>:ActuatorSensor:actuator_slow_response_detected` -> recalibration can be dispatched; regardless of detection, machine input rate drops -> `machine:<machine_id>:production_slowdown` -> `belt:<from_node_id>:<machine_id>:transfer_rate_degraded`; if reduced flow persists and causes queue buildup or waiting -> `belt:<from_node_id>:<machine_id>:persistent_queue_pressure` |
| Network latency reduces throughput | `network:latency` -> `network:network_latency_detected` -> delayed status reads or handoff coordination -> `belt:<from_node_id>:<to_node_id>:transfer_rate_degraded`; if reduced flow persists and causes queue buildup or waiting -> `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` |
| Severe or sustained network latency disrupts handoff | `network:latency` -> `network:network_latency_detected` -> delayed handoff coordination persists beyond the handoff timeout or coordination tolerance -> `belt:<from_node_id>:<to_node_id>:handoff_blocked`; if the upstream node is a machine, this can derive `machine:<from_node_id>:production_blocked`; if the blockage persists and causes queue buildup, waiting, or degraded throughput -> `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` |
| Network packet loss reduces throughput | `network:packet_loss` -> `network:network_packet_loss_detected` -> retries or dropped status reads reduce effective coordination rate -> `belt:<from_node_id>:<to_node_id>:transfer_rate_degraded`; if reduced flow persists and causes queue buildup or waiting -> `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` |
| Network packet dropped during handoff | `network:packet_loss` -> `network:network_packet_loss_detected` -> handoff notification or acknowledgement is lost -> `belt:<from_node_id>:<to_node_id>:handoff_blocked`; if the upstream node is a machine, this can derive `machine:<from_node_id>:production_blocked` while completed output waits for outgoing handoff; if the blockage persists and causes queue buildup, waiting, or degraded throughput -> `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` |
| Downstream machine blocked | `machine:<to_node_id>:production_blocked` -> `belt:<from_node_id>:<to_node_id>:handoff_blocked`; if the upstream node is a machine, this can derive `machine:<from_node_id>:production_blocked` while completed output waits at the upstream output or on the belt; if the blockage persists and causes queue buildup, waiting, or degraded throughput -> `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` |
| Machine bearing wear | `machine:<machine_id>:bearing_wear` -> slower processing -> `machine:<machine_id>:production_slowdown` -> `belt:<from_node_id>:<machine_id>:transfer_rate_degraded`; if reduced flow persists and causes queue buildup or waiting -> `belt:<from_node_id>:<machine_id>:persistent_queue_pressure` |
| Workpiece jams inside a machine | `machine:<machine_id>:jammed_workpiece` -> `machine:<machine_id>:production_blocked` -> `belt:<from_node_id>:<machine_id>:handoff_blocked`; if the upstream node is a machine, this can derive `machine:<from_node_id>:production_blocked` |
| Conveyor belt slips | `belt:<from_node_id>:<to_node_id>:belt_slippage` -> `belt:<from_node_id>:<to_node_id>:transfer_rate_degraded`; if reduced flow persists and causes queue buildup or waiting -> `belt:<from_node_id>:<to_node_id>:persistent_queue_pressure` |
| Conveyor belt jams | `belt:<from_node_id>:<to_node_id>:belt_jam` -> `belt:<from_node_id>:<to_node_id>:handoff_blocked`; if the upstream node is a machine, this can derive `machine:<from_node_id>:production_blocked` |
| Temperature rises too high | long continuous operation -> `temperature:<machine_id>:overheating`; if the Temperature sensor has no active fault -> `sensor:<machine_id>:Temperature:overheating_detected` -> machine lowers throughput for light cooling -> `machine:<machine_id>:production_slowdown`; if temperature crosses the critical threshold or overheating remains unresolved -> `temperature:<machine_id>:critical_overheating`; if the Temperature sensor has no active fault -> `sensor:<machine_id>:Temperature:critical_overheating_detected` -> machine stops production for intense cooling -> `machine:<machine_id>:production_blocked` |
| Temperature sensor fails and overheating escalates | `sensor:<machine_id>:Temperature:stuck` or `sensor:<machine_id>:Temperature:no_signal` -> Temperature sensor readings are ignored -> `temperature:<machine_id>:overheating` is missed -> no light cooling -> `temperature:<machine_id>:critical_overheating` is missed by the Temperature sensor -> machine stops production for intense cooling only after other handling intervenes -> `machine:<machine_id>:production_blocked` |
