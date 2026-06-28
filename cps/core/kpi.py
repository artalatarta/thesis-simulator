import logging
from collections.abc import Iterable

import simpy

from cps.types import ActionOutcome, ProcessingTimeState


class KPITracker:
	def __init__(self, env: simpy.Environment) -> None:
		self.env = env
		self.throughput = 0
		self.machine_states: dict[str, ProcessingTimeState] = {}
		self.open_faults: dict[str, float] = {}
		self.repair_times: list[float] = []
		self.conflicts_detected = 0
		self.resolver_attempts = 0
		self.resolver_successes = 0
		self.resolver_failures = 0
		self.agent_actions_attempted = 0
		self.agent_actions_succeeded = 0
		self.agent_actions_already_resolved = 0
		self.agent_actions_failed = 0

	@property
	def active_fault_count(self) -> int:
		return len(self.open_faults)

	@property
	def average_repair_time(self) -> float:
		if not self.repair_times:
			return 0.0
		return sum(self.repair_times) / len(self.repair_times)

	def initialize_machine_states(self, machines: Iterable[str]) -> None:
		for machine_id in machines:
			self.machine_states[machine_id] = {"total_processing_time": 0.0, "last_change_time": 0.0, "is_processing": False}

	def track_production(self) -> None:
		self.throughput += 1

	def track_machine_state_change(self, machine_id: str, is_processing: bool) -> None:
		now = self.env.now
		state_data = self.machine_states.get(machine_id)
		if not state_data:
			return
		if state_data["is_processing"]:
			state_data["total_processing_time"] += now - state_data["last_change_time"]
		state_data["last_change_time"] = now
		state_data["is_processing"] = is_processing

	def track_fault_start(self, owner: str, component: str) -> None:
		fault_key = f"{owner}-{component}"
		self.open_faults[fault_key] = self.env.now
		logging.info(f"KPI: Fault started for {fault_key} at T={self.env.now:.2f}", extra={"component": "KPI_Tracker"})

	def track_fault_end(self, owner: str, component: str) -> None:
		fault_key = f"{owner}-{component}"
		if fault_key in self.open_faults:
			start_time = self.open_faults.pop(fault_key)
			duration = self.env.now - start_time
			self.repair_times.append(duration)
			logging.info(f"KPI: Fault ended for {fault_key} at T={self.env.now:.2f}. Repair time: {duration:.2f}", extra={"component": "KPI_Tracker"})

	def track_conflict_detected(self) -> None:
		self.conflicts_detected += 1

	def track_resolver_attempt(self) -> None:
		self.resolver_attempts += 1

	def track_resolver_success(self) -> None:
		self.resolver_successes += 1

	def track_resolver_failure(self) -> None:
		self.resolver_failures += 1

	def track_agent_action(self, outcome: ActionOutcome) -> None:
		self.agent_actions_attempted += 1
		if outcome == "succeeded":
			self.agent_actions_succeeded += 1
		elif outcome == "already_resolved":
			self.agent_actions_already_resolved += 1
		else:
			self.agent_actions_failed += 1

	def generate_report(self) -> None:
		print("\n--- KPI Summary Report ---")
		print(f"Total Throughput: {self.throughput} parts")
		if self.machine_states:
			total_utilization = 0.0
			for machine_id, data in self.machine_states.items():
				effective_processing_time = data["total_processing_time"]
				if data["is_processing"]:
					effective_processing_time += self.env.now - data["last_change_time"]
				utilization_percent = (effective_processing_time / self.env.now) * 100 if self.env.now > 0 else 0
				print(f"  - Machine '{machine_id}' Utilization: {utilization_percent:.2f}%")
				total_utilization += utilization_percent
			avg_utilization = total_utilization / len(self.machine_states)
			print(f"Average Machine Utilization: {avg_utilization:.2f}%")
		if self.repair_times:
			print(f"Mean Time To Repair (MTTR): {self.average_repair_time:.2f} seconds (from {len(self.repair_times)} repairs)")
		else:
			print("Mean Time To Repair (MTTR): N/A (No repairs were completed)")
		if self.open_faults:
			print(f"Warning: {len(self.open_faults)} faults were still unresolved at the end of the simulation.")
		print("Monitoring and Conflict Metrics:")
		print(f"  - Conflicts Detected: {self.conflicts_detected}")
		print(f"  - Resolver Attempts: {self.resolver_attempts}")
		print(f"  - Resolver Successes: {self.resolver_successes}")
		print(f"  - Resolver Failures: {self.resolver_failures}")
		print(f"  - Agent Actions Attempted: {self.agent_actions_attempted}")
		print(f"  - Agent Actions Succeeded: {self.agent_actions_succeeded}")
		print(f"  - Agent Actions Already Resolved: {self.agent_actions_already_resolved}")
		print(f"  - Agent Actions Failed: {self.agent_actions_failed}")
		print("--- End of Report ---\n")
