# CPS Simulation with Agentic Capabilities: Linear Factory Line

This project is a discrete-event simulation of a factory environment, built using the SimPy library in Python. It models a linear production line: a raw-material source feeds ordered processing stations, belt segments move parts between stations, and final storage collects completed products.
The core of this simulation is a set of role-specific LLM monitoring agents that inspect the health and status of the machines, detect faults, and recommend corrective actions. The thesis focus is resolving conflicts between these agents when their diagnoses or actions disagree.

## Origins and attribution

This project started as an extension of the open-source
[Simulated-Agent-CPS-v1](https://github.com/SabariNathanA/Simulated-Agent-CPS-v1)
discrete-event factory simulator by Sabari Nathan A. The original provided the
SimPy-based linear-line model and the initial single-monitor agent scaffolding.
This thesis fork has since been largely rewritten---adding the six role-specific
LLM monitoring agents, the LLM conflict detector and single-call resolver, the
frozen ground-truth catalogue, and the post-run scoring harness---but the
original simulator is gratefully acknowledged as the starting point.

The editable Excalidraw sources for the factory line and evidence-to-action
flow are in `diagrams/`.

## Getting Started

This repository is managed with Pixi. Use the checked-in lock file so the
Python version and dependencies match the tested environment.

### 1. Prerequisites

Install [Pixi](https://pixi.sh), then run commands from the `simulator/`
directory.

### 2. Configuration

Copy the environment template and add an OpenRouter API key before starting a
live simulation:

```sh
cp .env.example .env
```

`OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`, and `OPENROUTER_PROVIDER_SORT` are
optional and documented in `.env.example`. Unit tests use fake clients and do
not require credentials or make network calls.

The default entry point is `main.py`. The complete ordered line is configured in
`cps/config.py` using `FactoryLineConfig` from
`cps/simulation/factory_line_config.py`: raw-material source, processing
stations, final storage, product name, and production quantity. Machine
schedules are generated from this line configuration. The factory topology is
always linear; stations are declared in processing order as `(machine_id,
process_time)`. Product identifiers are derived from the configured product
name and quantity, for example `car-001`, `car-002`, and so on.

### 3. Execution

Run the simulation from your terminal:

```sh
pixi run start
```

The simulation always runs with the monitoring agents, conflict detector, and conflict resolver.

Run quality checks:

```sh
pixi run test
pixi run lint
pixi run typecheck
```

### 4. Output

- Console output provides high-level status updates. Press Ctrl+C to stop the
  simulation.
- `output/<datetime>/simulation.log` contains the timestamped simulation event
  trace.
- `output/<datetime>/full.log` contains the complete debug log, including LLM
  monitoring and resolver traces.
- `output/<datetime>/conflict_resolution.log` contains only conflict-detector
  and resolver prompts and model responses.
- `output/<datetime>/runs.jsonl` contains the experiment record. All files for
  a run share the same `YYYY-MM-DD_HH-MM-SS` directory.

### 5. LLM monitoring agents and conflict resolver

The monitoring layer (`cps/agents/monitoring/`) models each monitor as a
role-specific LLM-agent persona: power sensor, temperature sensor, actuator
sensor, network, belt segment, and machine health. Real simulation runs call
the configured OpenRouter/OpenAI-compatible model for those monitoring agents;
unit tests inject a scripted fake client and never make live calls. Each emitted
`MonitoringReport` carries the concrete agent identity, kind, model, and
persona metadata. Machine- and belt-bound monitors are named by their concrete
scope, for example `PowerSensor@Sheet-Metal-Press`,
`MachineHealth@Paint-Booth`, or
`BeltSegment@Paint-Booth->Powertrain-Install`. Each named monitor also receives
a bounded history of its own recent reports, including passive
`wait_for_more_evidence` reports: prior evidence, diagnosis, diagnosis id,
recommended action, confidence, resolver selection, and execution outcome. The
history remains background continuity only; past evidence cannot justify a new
report without current-window evidence, and ground-truth fault data is never
included.

When LLM monitoring agents disagree, a *resolver*
(`cps/agents/resolution/`) turns each detected `Conflict` into a single
`ResolutionDecision`. The resolver calls a model through the OpenAI SDK pointed
at OpenRouter (default model `openai/gpt-oss-20b`). The simulator loads `.env`
automatically and fails at startup without `OPENROUTER_API_KEY` for real runs.

Each run writes one JSON record to `runs.jsonl` capturing the whole experiment: the
injected root faults and `ground_truth`, the `generated_reports`, the
`detected_conflicts`, the live monitoring reports (`runtime_llm_reports`) and resolver
decisions (`runtime_llm_decisions`), the executed `agent_actions`, and the scores under
`detection_metrics`, `runtime_detection`, `per_fault_outcomes`,
`resolver_correctness`, and `cascade`. `runtime_correctness` remains in the
artifact as a compatibility field for older analysis code.

## Thesis analysis and archived run

`analysis/analysis.ipynb` reports the thesis results from the fixed production
run `2026-06-26_02-48-14`. It is intentionally pinned rather than selecting the
newest output directory, so later runs cannot silently change the reported
results.
