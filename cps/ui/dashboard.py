import os
import subprocess
from collections.abc import Mapping

import simpy

from cps.core.flow import BeltSegment
from cps.core.node import FinalStorage, RawMaterialSource
from cps.core.kpi import KPITracker
from cps.core.node.machine import Machine
from cps.core.reporting import actuator_state_id, temperature_state_id
from cps.types import CoolingState


class LiveDashboard:
	def __init__(
		self,
		env: simpy.Environment,
		kpi_tracker: KPITracker,
		machines: Mapping[str, Machine],
		raw_material_source: RawMaterialSource | None = None,
		final_storage: FinalStorage | None = None,
		belt_segments: list[BeltSegment] | None = None,
	) -> None:
		self.env = env
		self.kpi_tracker = kpi_tracker
		self.machines = machines
		self.raw_material_source = raw_material_source
		self.final_storage = final_storage
		self.belt_segments = belt_segments or []
		self.last_machine_statuses: dict[str, str] = {m_id: "" for m_id in machines}

	def get_status_and_details(self, machine: Machine) -> tuple[str, str]:
		if machine.thermal_blocked:
			state_id = machine.temperature.state_id or temperature_state_id(machine.id, "critical_overheating")
			return "BLOCKED", f"Thermal block: {state_id}"
		if machine.temperature.cooling_state is CoolingState.INTENSE:
			return "COOLING", "Intense cooling"
		if machine.temperature.cooling_state is CoolingState.LIGHT:
			return "SLOWDOWN", "Light cooling"
		if machine.actuator.fault_type == "stuck":
			return "STUCK", actuator_state_id(machine.id, "stuck")
		if machine.actuator.fault_type == "slow_response":
			return "SLOWDOWN", actuator_state_id(machine.id, "slow_response")
		for component in (machine.temperature_sensor, machine.power_sensor, machine.actuator_sensor):
			if component.fault_type:
				return "FAULT", f"FAULT: Sensor {component.sensor_type} {component.fault_type}"
		if machine.is_processing:
			active_work = machine.production_state.active_work
			product_id = active_work.product_id if active_work is not None else "--"
			return "PROCESSING", f"Product: {product_id}"
		return "IDLE", "--"

	def display(self) -> None:
		subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"], check=False)
		print("--- CPS ---")
		print(f"Simulation Time: {self.env.now:.2f}s\n")
		active_faults = self.kpi_tracker.active_fault_count
		avg_mttr = self.kpi_tracker.average_repair_time
		print("[ KPIs ]")
		print(f"> Total Throughput: {self.kpi_tracker.throughput} parts")
		print(f"> Mean Time To Repair (MTTR): {avg_mttr:.2f}s")
		print(f"> Active Faults: {active_faults}\n")
		print("[ MACHINE STATUS ]")
		print(f"{'ID':<25} {'STATUS':<15} {'BATTERY':<10} {'TEMP':<8} {'COMPLETED':<8} {'QUEUE':<8} {'DETAILS'}")
		print("-" * 95)
		current_statuses: dict[str, str] = {}
		for machine_id, machine in self.machines.items():
			status, details = self.get_status_and_details(machine)
			current_statuses[machine_id] = status
			highlight_char = ">>" if self.last_machine_statuses.get(machine_id) != status else "  "
			battery_str = f"{machine.battery.level:.1f}%"
			temperature_str = f"{machine.temperature.value:.1f}C"
			parts_str = str(machine.parts_produced)
			queue_str = str(len(machine.production_state.production_schedule))
			print(f"{highlight_char} {machine_id:<22} {status:<15} {battery_str:<10} {temperature_str:<8} {parts_str:<8} {queue_str:<8} {details}")
		self.last_machine_statuses = current_statuses
		self._display_material_flow()
		print("\n--- Press Ctrl+C to stop the simulation ---")

	def _display_material_flow(self) -> None:
		print("\n[ MATERIAL FLOW ]")
		if self.raw_material_source is not None:
			print(f"> Raw Material Source: {self.raw_material_source.id}")
		if self.final_storage is not None:
			print(f"> Final Storage: {self.final_storage.id} ({len(self.final_storage.stored_parts)} stored parts)")
		if not self.belt_segments:
			return

		print("\n[ BELT SEGMENTS ]")
		rows = []
		for belt in self.belt_segments:
			path = f"{belt.from_node.id}->{belt.to_node.id}"
			status = "FAULT" if belt.fault_type else "OK"
			queue = ",".join(belt.queue) if belt.queue else "--"
			diagnostics = ", ".join(belt.active_diagnostic_ids()) or "--"
			rows.append((path, status, queue, str(belt.capacity), diagnostics))

		path_width = max(len("PATH"), *(len(row[0]) for row in rows))
		status_width = max(len("STATUS"), *(len(row[1]) for row in rows))
		queue_width = max(len("QUEUE"), *(len(row[2]) for row in rows))
		capacity_width = max(len("CAPACITY"), *(len(row[3]) for row in rows))
		header = (
			f"{'PATH':<{path_width}}  "
			f"{'STATUS':<{status_width}}  "
			f"{'QUEUE':<{queue_width}}  "
			f"{'CAPACITY':<{capacity_width}}  "
			"DIAGNOSTICS"
		)
		print(header)
		print("-" * len(header))
		for path, status, queue, capacity, diagnostics in rows:
			print(
				f"{path:<{path_width}}  "
				f"{status:<{status_width}}  "
				f"{queue:<{queue_width}}  "
				f"{capacity:<{capacity_width}}  "
				f"{diagnostics}"
			)
