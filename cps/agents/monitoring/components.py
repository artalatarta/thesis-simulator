"""Network, belt, and machine-health monitoring agents."""

from collections.abc import Callable

import simpy

from cps.agents.contracts import MonitoringReport
from cps.agents.fault_catalog import belt_diagnosis_id_template, machine_diagnosis_id_template, network_diagnosis_id_template
from cps.agents.identifiers import machine_id_from_identifier, parse_identifier
from cps.agents.monitoring.base import MonitoringAgent
from cps.agents.monitoring.context import MonitoringContext, monitoring_status_for_machine
from cps.agents.monitoring.state_observers import MachineFaultSymptomObserver
from cps.agents.resolution import LLMClient
from cps.core.flow import BeltSegment
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.network import Network
from cps.core.reporting import ReportedEvent
from cps.types import ActionOutcome

class NetworkAgent(MonitoringAgent):
	"""Reports shared-network latency and packet-loss observations."""

	role = "network"
	name = "Network"
	system_prompt_diagnosis_ids = (network_diagnosis_id_template(),)

	def __init__(self, network: Network, kpi_tracker: KPITracker, llm_client: LLMClient) -> None:
		super().__init__(llm_client)
		self.network: Network = network
		self.kpi_tracker: KPITracker = kpi_tracker

	@property
	def identity_name(self) -> str:
		return "Network@line"

	def owns_event(self, event: ReportedEvent) -> bool:
		return event.identifier.startswith("network:")

	def start(self, env: simpy.Environment) -> tuple[simpy.Process, ...]:
		_ = env
		self.network.start_observation_monitor()
		return (self.network.monitor_process,) if self.network.monitor_process is not None else ()

	def _dispatch_network_repair(self, report: MonitoringReport, fault_type: str) -> bool:
		parsed = parse_identifier(report.diagnosis_id or "")
		if parsed.kind != "network" or parsed.network_fault != fault_type:
			return False
		if self.network.fault_type != fault_type:
			return False
		return self.network.dispatch_repair(self.kpi_tracker)

	def _fix_latency(self, report: MonitoringReport, require_sensor_operational: bool) -> bool:
		_ = require_sensor_operational
		return self._dispatch_network_repair(report, "latency")

	def _fix_packet_loss(self, report: MonitoringReport, require_sensor_operational: bool) -> bool:
		_ = require_sensor_operational
		return self._dispatch_network_repair(report, "packet_loss")

	def _action_handlers(self) -> dict[str, Callable[[MonitoringReport, bool], ActionOutcome | bool | None]]:
		return {"fix_latency": self._fix_latency, "fix_packet_loss": self._fix_packet_loss}


class BeltSegmentAgent(MonitoringAgent):
	"""Reports belt-segment production-flow issues.

	Consumes belt derived issues from the event window and folds in any active
	belt diagnostics reported by :class:`~cps.core.flow.BeltSegment` that have
	not yet surfaced as events.
	"""

	role = "belt"
	name = "BeltSegment"
	system_prompt_diagnosis_ids = (belt_diagnosis_id_template(),)
	system_prompt_action_guidance = (
		"A belt fault detection reports a mechanical fault of the segment itself, such as slippage or a "
		"jam, and can also appear among active diagnostics. This is the clearable root fault and warrants a "
		"corrective action, even when handoff_blocked, persistent_queue_pressure, or transfer_rate_degraded also appear in "
		"the same window. Those flow symptoms are downstream effects rather than the fault itself, so choose a "
		"passive action for them only when no belt fault detection is present."
	)

	def __init__(self, belt: BeltSegment, llm_client: LLMClient) -> None:
		super().__init__(llm_client)
		self.belt: BeltSegment = belt

	@property
	def identity_name(self) -> str:
		return f"{self.name}@{self.belt.from_node.id}->{self.belt.to_node.id}"

	def owns_event(self, event: ReportedEvent) -> bool:
		parsed = parse_identifier(event.identifier)
		if parsed.from_node_id is not None and parsed.to_node_id is not None and (parsed.from_node_id, parsed.to_node_id) != (self.belt.from_node.id, self.belt.to_node.id):
			return False
		return parsed.kind == "belt"

	def _llm_supplementary_context(self, context: MonitoringContext) -> dict[str, object]:
		return {
			"from_node": self.belt.from_node.id,
			"to_node": self.belt.to_node.id,
			"active_diagnostics": list(self.belt.active_diagnostic_ids()),
		}

	def _llm_should_use_supplementary_without_events(self, context: MonitoringContext) -> bool:
		return bool(self.belt.active_diagnostic_ids())

	def execute_action(self, report: MonitoringReport, *, require_sensor_operational: bool = False) -> ActionOutcome | None:
		# Every belt agent shares the name "BeltSegment", so reports are matched
		# to the concrete belt by the node pair in their identifiers.
		if not self._report_targets_this_belt(report):
			return None
		return super().execute_action(report, require_sensor_operational=require_sensor_operational)

	def _report_targets_this_belt(self, report: MonitoringReport) -> bool:
		for identifier in (report.diagnosis_id, *report.evidence):
			if identifier is None:
				continue
			parsed = parse_identifier(identifier)
			if parsed.from_node_id is not None and parsed.to_node_id is not None:
				return (parsed.from_node_id, parsed.to_node_id) == (self.belt.from_node.id, self.belt.to_node.id)
		return False

	def _fix_belt_slippage(self, report: MonitoringReport, require_sensor_operational: bool) -> ActionOutcome:
		_ = require_sensor_operational
		return self._dispatch_belt_repair(report, "belt_slippage")

	def _fix_belt_jam(self, report: MonitoringReport, require_sensor_operational: bool) -> ActionOutcome:
		_ = require_sensor_operational
		return self._dispatch_belt_repair(report, "belt_jam")

	def _dispatch_belt_repair(self, report: MonitoringReport, fault_type: str) -> ActionOutcome:
		parsed = parse_identifier(report.diagnosis_id or "")
		if (
			parsed.kind != "belt"
			or (parsed.from_node_id, parsed.to_node_id) != (self.belt.from_node.id, self.belt.to_node.id)
			or parsed.observation != fault_type
		):
			return "failed"
		from_node = self.belt.from_node
		kpi_tracker = getattr(from_node, "kpi_tracker", None)
		if kpi_tracker is None:
			to_node = self.belt.to_node
			kpi_tracker = getattr(to_node, "kpi_tracker", None)
		if kpi_tracker is None:
			return "failed"
		if self.belt.fault_type is None:
			# The named fault was already cleared by the time the action executed.
			return "already_resolved"
		return "succeeded" if self.belt.dispatch_repair(kpi_tracker, fault_type) else "failed"  # type: ignore[arg-type]

	def _action_handlers(self) -> dict[str, Callable[[MonitoringReport, bool], ActionOutcome | bool | None]]:
		return {"fix_belt_slippage": self._fix_belt_slippage, "fix_belt_jam": self._fix_belt_jam}


class MachineHealthAgent(MonitoringAgent):
	"""Reports machine-level production blockage and slowdown.

	Provides the machine's current status as background context while requiring
	the model to ground every report in machine events from the current window.
	"""

	role = "machine_health"
	name = "MachineHealth"
	system_prompt_diagnosis_ids = (machine_diagnosis_id_template(),)
	system_prompt_action_guidance = (
		"A machine fault detection reports a mechanical fault of the station itself, such as bearing wear "
		"or a jammed workpiece. This is the clearable root fault and warrants a corrective action, even when "
		"production_slowdown or production_blocked also appear in the same window. Those production symptoms "
		"are downstream effects rather than the fault itself, so choose a passive action for them only when no "
		"machine fault detection is present."
	)

	def __init__(self, machine: Machine, llm_client: LLMClient) -> None:
		super().__init__(llm_client)
		self.machine: Machine = machine
		self.state_observer = MachineFaultSymptomObserver(machine)

	@property
	def identity_name(self) -> str:
		return f"{self.name}@{self.machine.id}"

	def start(self, env: simpy.Environment) -> tuple[simpy.Process, ...]:
		return (env.process(self.state_observer.monitor()),)

	def owns_event(self, event: ReportedEvent) -> bool:
		if machine_id_from_identifier(event.identifier) != self.machine.id:
			return False
		return parse_identifier(event.identifier).kind == "machine"

	def _report_machine_id(self, identifier: str) -> str:
		return self.machine.id

	def execute_action(self, report: MonitoringReport, *, require_sensor_operational: bool = False) -> ActionOutcome | None:
		# Every machine-health agent shares the name "MachineHealth", so only the
		# agent bound to the report's machine may execute its action.
		if report.machine_id != self.machine.id:
			return None
		return super().execute_action(report, require_sensor_operational=require_sensor_operational)

	def _llm_supplementary_context(self, context: MonitoringContext) -> dict[str, object]:
		status = context.machine_status.get(self.machine.id)
		return {
			"machine_id": self.machine.id,
			"status": dict(status) if status is not None else monitoring_status_for_machine(self.machine),
		}

	def _fix_bearing_wear(self, report: MonitoringReport, require_sensor_operational: bool) -> ActionOutcome:
		_ = require_sensor_operational
		return self._dispatch_repair(report, "bearing_wear")

	def _fix_jammed_workpiece(self, report: MonitoringReport, require_sensor_operational: bool) -> ActionOutcome:
		_ = require_sensor_operational
		return self._dispatch_repair(report, "jammed_workpiece")

	def _dispatch_repair(self, report: MonitoringReport, fault_type: str) -> ActionOutcome:
		parsed = parse_identifier(report.diagnosis_id or "")
		if parsed.kind != "machine" or parsed.parts[1] != self.machine.id or parsed.state_or_issue != fault_type:
			return "failed"
		if self.machine.fault_type is None:
			# The named fault was already cleared by the time the action executed.
			return "already_resolved"
		return "succeeded" if self.machine.dispatch_repair(fault_type) else "failed"  # type: ignore[arg-type]

	def _action_handlers(self) -> dict[str, Callable[[MonitoringReport, bool], ActionOutcome | bool | None]]:
		return {"fix_bearing_wear": self._fix_bearing_wear, "fix_jammed_workpiece": self._fix_jammed_workpiece}
