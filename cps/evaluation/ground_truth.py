"""CSV-backed ground-truth catalog for evaluable faults, states, and issues.

The expected best decision for each injected root fault lives in
``ground_truth.csv`` in the simulator project root rather than in code, so the
thesis evaluation oracle is editable and auditable alongside ``faults.md``. Each row
captures one root-fault type and, crucially, the *causal chain* it can trigger:

- ``required_action``   -- the single action that must be taken to correctly
  handle the fault. A missing required action counts against the agent.
- ``chain_effects``     -- the derived effects the fault can cascade into, taken
  from the "Important Causal Chains" section of ``faults.md``. These are
  documentation/traceability for the record; ``<machine_id>`` is resolved to the
  concrete machine. For belt root faults the id names both endpoints directly;
  for machine-scoped faults, ``<from_node_id>`` / ``<to_node_id>`` are resolved
  from the belt topology observed in the run when that topology is available.

faults.md frames every chain as a *conditional* escalation ("if unresolved ->",
"can cause"). Actions implied by physical-state chain effects are tolerated
rather than mandatory.
"""

import csv
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from cps.agents.identifiers import ParsedIdentifier, parse_identifier

GROUND_TRUTH_CSV = Path(__file__).resolve().parents[2] / "ground_truth.csv"

CatalogKey = tuple[str, str, str, str]
GroundTruthRecord = dict[str, object]


@lru_cache(maxsize=None)
def load_ground_truth_catalog() -> dict[CatalogKey, dict[str, str]]:
	catalog: dict[CatalogKey, dict[str, str]] = {}
	with GROUND_TRUTH_CSV.open(newline="", encoding="utf-8") as handle:
		for row in csv.DictReader(handle):
			key = (row["category"], row["domain"], row["sensor_type"], row["fault"])
			catalog[key] = row
	return catalog


def string_list(value: object) -> list[str]:
	if isinstance(value, (list, tuple)):
		return [item for item in value if isinstance(item, str)]
	if isinstance(value, str):
		return [value]
	return []


def physical_state_response(kind: str, state: str) -> dict[str, str] | None:
	return load_ground_truth_catalog().get(("physical_state", kind, "", state))


def evaluable_ground_truth(ground_truth: Iterable[GroundTruthRecord]) -> list[GroundTruthRecord]:
	return [item for item in ground_truth if item.get("evaluable") is not False]


def evaluable_fault_ground_truth(ground_truth: Iterable[GroundTruthRecord]) -> list[GroundTruthRecord]:
	return [item for item in evaluable_ground_truth(ground_truth) if item.get("source") != "derived_issue"]


def root_fault_ground_truth(ground_truth: Iterable[GroundTruthRecord]) -> list[GroundTruthRecord]:
	return [
		item
		for item in evaluable_ground_truth(ground_truth)
		if item.get("source", "root_fault") == "root_fault" and isinstance(item.get("root_fault"), str)
	]


def duplicate_diagnosis_groups(ground_truth: Iterable[GroundTruthRecord]) -> list[dict[str, object]]:
	"""Return auditable groups where multiple truth rows share one diagnosis."""
	groups: dict[str, list[GroundTruthRecord]] = defaultdict(list)
	for item in ground_truth:
		diagnosis = item.get("diagnosis")
		if isinstance(diagnosis, str):
			groups[diagnosis].append(item)
	return [
		{
			"diagnosis": diagnosis,
			"count": len(items),
			"truth_ids": [str(item.get("truth_id", "")) for item in items],
			"sources": sorted({str(item.get("source", "")) for item in items}),
			"evaluation_roles": sorted({str(item.get("evaluation_role", "")) for item in items}),
		}
		for diagnosis, items in sorted(groups.items())
		if len(items) > 1
	]


def ground_truth_for_root_faults(
	root_fault_ids: Iterable[str],
	neighbors: dict[str, tuple[str | None, str | None]] | None = None,
) -> list[GroundTruthRecord]:
	"""Build expected best decisions from the ``ground_truth.csv`` catalog.

	``neighbors`` maps ``machine_id -> (upstream_node_id, downstream_node_id)`` so
	that machine-scoped faults can resolve the belt/upstream placeholders in their
	``chain_effects``. It defaults to empty, leaving placeholders untouched.
	"""
	catalog = load_ground_truth_catalog()
	neighbors = neighbors or {}
	ground_truth: list[GroundTruthRecord] = []
	for root_fault_id in root_fault_ids:
		expectation = _ground_truth_for_root_fault(parse_identifier(root_fault_id), catalog, neighbors)
		if expectation is not None:
			ground_truth.append(expectation)
	return ground_truth


def physical_state_ground_truth(observed_state_ids: Iterable[str]) -> list[GroundTruthRecord]:
	"""Build expected decisions for physical states that occurred in the run."""
	ground_truth: list[GroundTruthRecord] = []
	seen: set[str] = set()
	for state_id in observed_state_ids:
		if state_id in seen:
			continue
		seen.add(state_id)
		parsed = parse_identifier(state_id)
		response = physical_state_response(parsed.kind, parsed.state_or_issue or "")
		if response is None:
			continue
		ground_truth.append(
			{
				"truth_id": f"physical_state:{state_id}",
				"diagnosis": _substitute(response["diagnosis"], _substitutions_for_root_fault(parsed, {})),
				"required_action": response["required_action"],
				"source": "physical_state",
				"evaluation_role": "physical_state_response",
				"evaluable": True,
			}
		)
	return ground_truth


def derived_issue_ground_truth(observed_issue_ids: Iterable[str]) -> list[GroundTruthRecord]:
	"""Build context rows for derived issues that occurred in the run."""
	ground_truth: list[GroundTruthRecord] = []
	seen: set[str] = set()
	for issue_id in observed_issue_ids:
		if issue_id in seen:
			continue
		seen.add(issue_id)
		parsed = parse_identifier(issue_id)
		issue = parsed.observation if parsed.kind == "belt" else parsed.state_or_issue
		row = load_ground_truth_catalog().get(("derived_issue", parsed.kind, "", issue or ""))
		if row is None:
			continue
		ground_truth.append(
			{
				"truth_id": f"derived_issue:{issue_id}",
				"diagnosis": issue_id,
				"required_action": "",
				"source": "derived_issue",
				"evaluation_role": "derived_issue_context",
				"evaluable": True,
			}
		)
	return ground_truth


def _ground_truth_for_root_fault(
	parsed: ParsedIdentifier,
	catalog: dict[CatalogKey, dict[str, str]],
	neighbors: dict[str, tuple[str | None, str | None]],
) -> GroundTruthRecord | None:
	key = _catalog_key_for_root_fault(parsed)
	if key is None:
		return None
	row = catalog.get(key)
	if row is None:
		return None
	substitutions = _substitutions_for_root_fault(parsed, neighbors)
	return {
		"truth_id": f"root_fault:{parsed.raw}",
		"root_fault": parsed.raw,
		"diagnosis": _substitute(row["diagnosis"], substitutions),
		"required_action": row["required_action"].strip(),
		"chain_effects": [_substitute(effect, substitutions) for effect in _split(row["chain_effects"])],
		"source": "root_fault",
		"evaluation_role": "root_fault",
	}


def _substitutions_for_root_fault(
	parsed: ParsedIdentifier,
	neighbors: dict[str, tuple[str | None, str | None]],
) -> dict[str, str]:
	substitutions: dict[str, str] = {}
	if parsed.machine_id is not None:
		substitutions["<machine_id>"] = parsed.machine_id
	if parsed.from_node_id is not None:
		substitutions["<from_node_id>"] = parsed.from_node_id
	if parsed.to_node_id is not None:
		substitutions["<to_node_id>"] = parsed.to_node_id
	# For a machine-scoped fault the root id names only the machine, not its belt
	# neighbours, so resolve <from_node_id>/<to_node_id> from belt events observed
	# this run. Do not override endpoints the parsed id already provides.
	if parsed.machine_id is not None and parsed.from_node_id is None and parsed.to_node_id is None:
		upstream, downstream = neighbors.get(parsed.machine_id, (None, None))
		if upstream is not None:
			substitutions["<from_node_id>"] = upstream
		if downstream is not None:
			substitutions["<to_node_id>"] = downstream
	return substitutions


def _catalog_key_for_root_fault(parsed: ParsedIdentifier) -> CatalogKey | None:
	if parsed.network_fault is not None:
		return ("root_fault", "network", "", parsed.network_fault)
	if parsed.kind == "actuator" and parsed.state_or_issue is not None:
		return ("root_fault", "actuator", "", parsed.state_or_issue)
	if parsed.kind == "machine" and parsed.state_or_issue is not None:
		return ("root_fault", "machine", "", parsed.state_or_issue)
	if parsed.kind == "belt" and parsed.observation is not None:
		return ("root_fault", "belt", "", parsed.observation)
	if parsed.kind == "sensor" and parsed.sensor_type is not None and parsed.observation is not None:
		return ("root_fault", "sensor", parsed.sensor_type, parsed.observation)
	return None


def _split(value: str) -> list[str]:
	return [item for item in (part.strip() for part in value.split("|")) if item]


def _substitute(template: str, substitutions: dict[str, str]) -> str:
	for placeholder, value in substitutions.items():
		template = template.replace(placeholder, value)
	return template
