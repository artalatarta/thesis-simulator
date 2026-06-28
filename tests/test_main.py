import logging
from types import SimpleNamespace

import main


def test_main_runs_interactive_simulation(monkeypatch) -> None:
	calls = []

	def run_simulation() -> None:
		calls.append(True)

	monkeypatch.setattr(main, "configure_logging", lambda *args, **kwargs: None)
	monkeypatch.setattr(main, "run_simulation", run_simulation)

	main.main()

	assert calls == [True]


def test_configure_logging_opens_log_only_when_called(tmp_path) -> None:
	log_path = tmp_path / "output" / "2026-06-10_12-00-00" / "simulation.log"
	full_log_path = tmp_path / "output" / "2026-06-10_12-00-00" / "full.log"
	conflict_resolution_log_path = tmp_path / "output" / "2026-06-10_12-00-00" / "conflict_resolution.log"

	main.configure_logging(log_path, full_log_path, conflict_resolution_log_path)
	logging.info("test log entry")
	logging.debug("debug log entry")
	logging.debug("RESOLVER_PROMPT conflict_id=c1 reports=[] system=system user=user")
	logging.debug("RESOLVER_COMPLETION conflict_id=c1 attempt=1 temperature=0.00 model=mock prompt_tokens=1 completion_tokens=2 latency_ms=3.00 text=response")
	logging.debug("RESOLVER_DECISION conflict_id=c1 selected_action=wait_for_more_evidence")

	assert "test log entry" in log_path.read_text(encoding="utf-8")
	assert "test log entry" in full_log_path.read_text(encoding="utf-8")
	assert "debug log entry" in full_log_path.read_text(encoding="utf-8")
	assert "debug log entry" not in log_path.read_text(encoding="utf-8")
	conflict_resolution_log = conflict_resolution_log_path.read_text(encoding="utf-8")
	assert "RESOLVER_PROMPT conflict_id=c1" in conflict_resolution_log
	assert "RESOLVER_COMPLETION conflict_id=c1" in conflict_resolution_log
	assert "RESOLVER_DECISION conflict_id=c1" not in conflict_resolution_log
	assert "debug log entry" not in conflict_resolution_log


def test_interactive_shutdown_drains_monitoring_driver(monkeypatch) -> None:
	close_calls = []

	def stopped_fault_injector(*_args, **_kwargs):
		yield from ()

	class Driver:
		resolution_decisions = []
		conflicts = []
		reports = []
		report_ledger = []
		executed_actions = []

		def close(self):
			close_calls.append(True)

	machine = SimpleNamespace()
	machines = {"M1": machine}
	kpi_tracker = SimpleNamespace(generate_report=lambda: None)
	simulation = SimpleNamespace(
		machines=machines,
		network=object(),
		faultable_components=[],
		raw_material_source=object(),
		final_storage=SimpleNamespace(stored_parts=[]),
		belt_segments=[],
	)

	monkeypatch.setattr(main, "KPITracker", lambda _env: kpi_tracker)
	monkeypatch.setattr(main, "openrouter_client_from_env", lambda: object())
	monkeypatch.setattr(main, "ConflictResolver", lambda client: object())
	monkeypatch.setattr(main, "setup_simulation", lambda _env, _tracker, _config, _event_reporter: simulation)
	monkeypatch.setattr(main, "configure_monitoring", lambda *args: Driver())
	monkeypatch.setattr(main, "LiveDashboard", lambda *args: SimpleNamespace(display=lambda: None))
	monkeypatch.setattr(main, "fault_injector", stopped_fault_injector)
	monkeypatch.setattr(
		main,
		"run_simulation_loop",
		lambda _env, _dashboard, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
	)
	monkeypatch.setattr(main, "build_experiment_record", lambda **kwargs: {})
	monkeypatch.setattr(main, "write_jsonl", lambda records, path: None)

	main.run_simulation()

	assert close_calls == [True]


def test_interactive_fault_injection_stops_at_production_target_and_run_waits_for_recovery(monkeypatch) -> None:
	captured = {}

	def stopped_fault_injector(_env, _components, _tracker, should_inject, *_args, **_kwargs):
		captured["should_inject"] = should_inject

		def stopped_process():
			yield from ()

		return stopped_process()

	class Driver:
		resolution_decisions = []
		conflicts = []
		reports = []
		report_ledger = []
		executed_actions = []

		def close(self):
			pass

	machine = SimpleNamespace(recovery_is_complete=False)
	machines = {"M1": machine}
	kpi_tracker = SimpleNamespace(throughput=main.DEFAULT_FACTORY_CONFIG.quantity, generate_report=lambda: None)
	final_storage = SimpleNamespace(stored_parts=["part"] * (main.DEFAULT_FACTORY_CONFIG.quantity - 1))
	simulation = SimpleNamespace(
		machines=machines,
		network=object(),
		faultable_components=[],
		raw_material_source=object(),
		final_storage=final_storage,
		belt_segments=[],
	)

	def run_simulation_loop(_env, _dashboard, **kwargs):
		captured["is_complete"] = kwargs["is_complete"]

	monkeypatch.setattr(main, "KPITracker", lambda _env: kpi_tracker)
	monkeypatch.setattr(main, "openrouter_client_from_env", lambda: object())
	monkeypatch.setattr(main, "ConflictResolver", lambda client: object())
	monkeypatch.setattr(main, "setup_simulation", lambda _env, _tracker, _config, _event_reporter: simulation)
	monkeypatch.setattr(main, "configure_monitoring", lambda *args: Driver())
	monkeypatch.setattr(main, "LiveDashboard", lambda *args: SimpleNamespace(display=lambda: None))
	monkeypatch.setattr(main, "fault_injector", stopped_fault_injector)
	monkeypatch.setattr(main, "run_simulation_loop", run_simulation_loop)
	monkeypatch.setattr(main, "play_completion_sound", lambda: None)
	monkeypatch.setattr(main, "build_experiment_record", lambda **kwargs: {})
	monkeypatch.setattr(main, "write_jsonl", lambda records, path: None)

	main.run_simulation()

	assert captured["should_inject"]()
	assert not captured["is_complete"]()

	final_storage.stored_parts.append("part")
	assert not captured["should_inject"]()
	assert not captured["is_complete"]()

	machine.recovery_is_complete = True
	assert captured["is_complete"]()
