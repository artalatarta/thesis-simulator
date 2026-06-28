import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import simpy
from dotenv import load_dotenv

from cps.agents.detection import ConflictDetector
from cps.agents.monitoring.setup import configure_monitoring
from cps.agents.resolution import ConflictResolver, openrouter_client_from_env
from cps.config import DEFAULT_FACTORY_CONFIG
from cps.core.kpi import KPITracker
from cps.core.reporting import EventReporter
from cps.evaluation.serialization import conflict_to_record, dataclass_to_record, resolution_decision_to_record
from cps.simulation.runtime import recovery_is_complete, run_simulation_loop
from cps.evaluation.scenarios import build_experiment_record, write_jsonl
from cps.simulation.setup import fault_injector, setup_simulation
from cps.ui.dashboard import LiveDashboard


class ComponentFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		if not hasattr(record, "component"):
			record.component = record.name
		return super().format(record)


OUTPUT_PATH = Path("output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUNS_OUTPUT_PATH = OUTPUT_PATH / "runs.jsonl"
LOG_PATH = OUTPUT_PATH / "simulation.log"
FULL_LOG_PATH = OUTPUT_PATH / "full.log"
CONFLICT_RESOLUTION_LOG_PATH = OUTPUT_PATH / "conflict_resolution.log"
CHECKPOINT_INTERVAL = 50.0


class ConflictResolutionPromptFilter(logging.Filter):
	def filter(self, record: logging.LogRecord) -> bool:
		message = record.getMessage()
		return message.startswith(("RESOLVER_PROMPT ", "RESOLVER_COMPLETION ", "DETECTOR_PROMPT ", "DETECTOR_COMPLETION "))


def configure_logging(
	log_path: Path = LOG_PATH,
	full_log_path: Path | None = None,
	conflict_resolution_log_path: Path | None = None,
) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	log_handler = logging.FileHandler(log_path, mode="w")
	log_handler.setLevel(logging.INFO)
	log_handler.setFormatter(ComponentFormatter("%(asctime)s [%(levelname)s] (%(component)s) - %(message)s"))
	handlers: list[logging.Handler] = [log_handler]
	if full_log_path is not None:
		full_log_path.parent.mkdir(parents=True, exist_ok=True)
		full_handler = logging.FileHandler(full_log_path, mode="w")
		full_handler.setLevel(logging.DEBUG)
		full_handler.setFormatter(ComponentFormatter("%(asctime)s [%(levelname)s] (%(component)s) %(name)s:%(lineno)d - %(message)s"))
		handlers.append(full_handler)
	if conflict_resolution_log_path is not None:
		conflict_resolution_log_path.parent.mkdir(parents=True, exist_ok=True)
		conflict_resolution_handler = logging.FileHandler(conflict_resolution_log_path, mode="w")
		conflict_resolution_handler.setLevel(logging.DEBUG)
		conflict_resolution_handler.addFilter(ConflictResolutionPromptFilter())
		conflict_resolution_handler.setFormatter(ComponentFormatter("%(asctime)s [%(levelname)s] (%(component)s) %(name)s:%(lineno)d - %(message)s"))
		handlers.append(conflict_resolution_handler)
	logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)


def play_completion_sound() -> None:
	if platform.system() == "Darwin":
		subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
		return
	print("\a", end="", flush=True)


def run_simulation() -> None:
	print("--- Starting CPS Simulation ---")

	env = simpy.Environment()

	kpi_tracker = KPITracker(env)
	llm_client = openrouter_client_from_env()
	detector = ConflictDetector(llm_client)
	resolver = ConflictResolver(llm_client)

	# This run's reporter stamps every event with the simulation time so the
	# live run can produce a complete experiment record.
	run_reporter = EventReporter(now=lambda: env.now)
	simulation = setup_simulation(env, kpi_tracker, DEFAULT_FACTORY_CONFIG, run_reporter)
	machines = simulation.machines
	if not machines:
		print("No machines are configured.")
		return

	driver = configure_monitoring(env, machines, kpi_tracker, simulation.network, detector, resolver, llm_client, run_reporter)

	live_dashboard = LiveDashboard(
		env,
		kpi_tracker,
		machines,
		simulation.raw_material_source,
		simulation.final_storage,
		simulation.belt_segments,
	)

	def production_is_complete() -> bool:
		return len(simulation.final_storage.stored_parts) >= DEFAULT_FACTORY_CONFIG.quantity

	last_checkpoint_at = 0.0

	def write_checkpoint() -> None:
		record = build_experiment_record(
			window_start=0.0,
			window_end=float(env.now),
			runtime_llm_decisions=[resolution_decision_to_record(decision) for decision in driver.resolution_decisions],
			runtime_conflicts=[conflict_to_record(conflict) for conflict in driver.conflicts],
			runtime_llm_reports=[dataclass_to_record(report) for report in driver.report_ledger],
			agent_actions=driver.executed_actions,
			events=run_reporter.events,
		)
		write_jsonl([record], RUNS_OUTPUT_PATH)

	def checkpoint_after_step() -> None:
		nonlocal last_checkpoint_at
		if float(env.now) - last_checkpoint_at < CHECKPOINT_INTERVAL:
			return
		write_checkpoint()
		last_checkpoint_at = float(env.now)

	env.process(fault_injector(env, simulation.faultable_components, kpi_tracker, lambda: not production_is_complete()))

	print("--- Simulation starting. Press Ctrl+C to stop. ---")
	try:
		# Fault injection stops with production, but the run keeps stepping
		# until monitoring has repaired every open fault, so late-injected
		# faults are resolved instead of expiring with the run.
		run_simulation_loop(
			env,
			live_dashboard,
			is_complete=lambda: production_is_complete() and recovery_is_complete(machines.values()),
			after_step=checkpoint_after_step,
		)
		play_completion_sound()

	except KeyboardInterrupt:
		print("\n--- Simulation stopped by user (Ctrl+C). ---")
	finally:
		driver.close()

	print("\n--- Simulation Finished ---")
	live_dashboard.display()
	kpi_tracker.generate_report()

	write_checkpoint()
	print(f"Wrote experiment record to {RUNS_OUTPUT_PATH}")


def main() -> None:
	load_dotenv()
	configure_logging(full_log_path=FULL_LOG_PATH, conflict_resolution_log_path=CONFLICT_RESOLUTION_LOG_PATH)
	run_simulation()


if __name__ == "__main__":
	main()
