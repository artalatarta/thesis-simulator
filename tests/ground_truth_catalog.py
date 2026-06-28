from cps.agents.contracts import ACTION_LABELS
from cps.agents.fault_catalog import (
	ACTUATOR_FAULT_TYPES,
	ACTUATOR_SENSOR_FAULT_TYPES,
	ACTUATOR_SENSOR_TYPE,
	BATTERY_STATE_IDS,
	BELT_FAULT_IDS,
	BELT_ISSUE_IDS,
	MACHINE_FAULT_IDS,
	MACHINE_ISSUE_IDS,
	MEASUREMENT_SENSOR_FAULT_TYPES,
	MEASUREMENT_SENSOR_TYPES,
	NETWORK_FAULT_TYPES,
	TEMPERATURE_STATE_IDS,
	is_fault_catalog_diagnosis_id,
)
from cps.evaluation.ground_truth import CatalogKey, load_ground_truth_catalog


def catalog_rows_by_category(category: str) -> list[dict[str, str]]:
	return [row for key, row in load_ground_truth_catalog().items() if key[0] == category]


def validate_ground_truth_catalog() -> list[str]:
	catalog = load_ground_truth_catalog()
	errors: list[str] = []
	_validate_category_keys(catalog, "root_fault", _expected_root_fault_keys(), errors)
	_validate_category_keys(catalog, "physical_state", _expected_physical_state_keys(), errors)
	_validate_category_keys(catalog, "derived_issue", _expected_derived_issue_keys(), errors)
	for key, row in sorted(catalog.items()):
		_validate_catalog_row(key, row, errors)
	return errors


def _expected_root_fault_keys() -> set[CatalogKey]:
	return (
		{("root_fault", "sensor", sensor_type, fault) for sensor_type in MEASUREMENT_SENSOR_TYPES for fault in MEASUREMENT_SENSOR_FAULT_TYPES}
		| {("root_fault", "sensor", ACTUATOR_SENSOR_TYPE, fault) for fault in ACTUATOR_SENSOR_FAULT_TYPES}
		| {("root_fault", "actuator", "", fault) for fault in ACTUATOR_FAULT_TYPES}
		| {("root_fault", "network", "", fault) for fault in NETWORK_FAULT_TYPES}
		| {("root_fault", "machine", "", fault) for fault in MACHINE_FAULT_IDS}
		| {("root_fault", "belt", "", fault) for fault in BELT_FAULT_IDS}
	)


def _expected_physical_state_keys() -> set[CatalogKey]:
	return (
		{("physical_state", "battery", "", state) for state in BATTERY_STATE_IDS}
		| {("physical_state", "temperature", "", state) for state in TEMPERATURE_STATE_IDS}
	)


def _expected_derived_issue_keys() -> set[CatalogKey]:
	return {("derived_issue", "machine", "", issue) for issue in MACHINE_ISSUE_IDS} | {
		("derived_issue", "belt", "", issue) for issue in BELT_ISSUE_IDS
	}


def _validate_category_keys(
	catalog: dict[CatalogKey, dict[str, str]],
	category: str,
	expected: set[CatalogKey],
	errors: list[str],
) -> None:
	actual = {key for key in catalog if key[0] == category}
	for key in sorted(expected - actual):
		errors.append(f"missing {category} catalog row: {key}")
	for key in sorted(actual - expected):
		errors.append(f"unexpected {category} catalog row: {key}")


def _validate_catalog_row(key: CatalogKey, row: dict[str, str], errors: list[str]) -> None:
	category, domain, _sensor_type, _fault = key
	required_action = row["required_action"].strip()
	if category in {"root_fault", "physical_state"} and not required_action:
		errors.append(f"{key} is missing required_action")
	if category == "derived_issue" and required_action:
		errors.append(f"{key} should not have required_action")
	if category == "root_fault" and "|" in required_action:
		errors.append(f"{key} should have exactly one required_action")
	if required_action and required_action not in ACTION_LABELS:
		errors.append(f"{key} has unknown action {required_action!r}")
	for field in ("diagnosis", "chain_effects"):
		for identifier in _split(row[field]) if field == "chain_effects" else [row[field]]:
			if identifier and not _is_valid_catalog_identifier_template(identifier):
				errors.append(f"{key} has invalid {field} identifier {identifier!r}")
	for identifier in [row["diagnosis"], *_split(row["chain_effects"])]:
		_validate_placeholder_usage(key, identifier, errors)
	if category == "physical_state" and domain not in {"battery", "temperature"}:
		errors.append(f"{key} physical_state domain must be battery or temperature")


def _split(value: str) -> list[str]:
	return [item for item in (part.strip() for part in value.split("|")) if item]


def _is_valid_catalog_identifier_template(identifier: str) -> bool:
	return is_fault_catalog_diagnosis_id(_placeholder_concrete_identifier(identifier))


def _placeholder_concrete_identifier(identifier: str) -> str:
	return (
		identifier.replace("<machine_id>", "M1")
		.replace("<from_node_id>", "M1")
		.replace("<to_node_id>", "M2")
		.replace("<from_node>", "M1")
		.replace("<to_node>", "M2")
	)


def _validate_placeholder_usage(key: CatalogKey, identifier: str, errors: list[str]) -> None:
	placeholders = {part for part in ("<machine_id>", "<from_node_id>", "<to_node_id>") if part in identifier}
	if not placeholders:
		return
	category, domain, _sensor_type, _fault = key
	allowed = {"<machine_id>"} if domain not in {"belt", "network"} else {"<from_node_id>", "<to_node_id>"}
	if domain in {"sensor", "actuator", "machine"} and identifier == "":
		allowed = {"<machine_id>"}
	if category == "root_fault" and domain in {"sensor", "actuator", "machine"} and identifier != _row_diagnosis_marker(key):
		allowed |= {"<from_node_id>", "<to_node_id>"}
	if placeholders - allowed:
		errors.append(f"{key} uses invalid placeholders {sorted(placeholders - allowed)} in {identifier!r}")


def _row_diagnosis_marker(key: CatalogKey) -> str:
	category, domain, sensor_type, fault = key
	if category != "root_fault":
		return ""
	if domain == "network":
		return f"network:{fault}"
	if domain == "sensor":
		return f"sensor:<machine_id>:{sensor_type}:{fault}"
	if domain in {"actuator", "machine"}:
		return f"{domain}:<machine_id>:{fault}"
	if domain == "belt":
		return f"belt:<from_node_id>:<to_node_id>:{fault}"
	return ""
