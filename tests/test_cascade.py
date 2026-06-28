from cps.core.reporting import EventReporter
from cps.evaluation.cascade import score_cascades
from cps.evaluation.ground_truth import ground_truth_for_root_faults


def test_fault_resolved_before_first_chain_effect_is_contained() -> None:
	reporter = EventReporter()
	reporter.root_fault("sensor:M1:Power:stuck", context={"time": 1.0})
	reporter.fault_resolved("sensor:M1:Power:stuck", context={"time": 9.0})
	reporter.physical_state("battery:M1:low_battery", component="Battery", cause_id="sensor:M1:Power:stuck", context={"time": 10.0})

	result = score_cascades(
		reporter.events,
		[
			{
				"root_fault": "sensor:M1:Power:stuck",
				"source": "root_fault",
				"chain_effects": ["battery:M1:low_battery", "machine:M1:production_blocked"],
			}
		],
		window_end=20.0,
		poll_interval=5.0,
	)

	assert result["per_fault"] == [
		{
			"root_fault": "sensor:M1:Power:stuck",
			"injected_at": 1.0,
			"resolved_at": 9.0,
			"first_effect_at": None,
			"manifestation_latency": None,
			"containable": True,
			"polls_to_resolution": 2,
			"depth": 0,
			"contained": True,
			"reached_effects": [],
		}
	]
	assert result["metrics"]["n_contained"] == 1
	assert result["metrics"]["n_cascaded"] == 0
	assert result["metrics"]["n_containable"] == 1
	assert result["metrics"]["n_structurally_cascading"] == 0
	assert result["metrics"]["n_contained_given_containable"] == 1
	assert result["metrics"]["contained_rate_over_containable"] == 1.0


def test_fast_chain_effect_is_structurally_cascading() -> None:
	reporter = EventReporter()
	reporter.root_fault("sensor:M1:Power:stuck", context={"time": 1.0})
	reporter.physical_state("battery:M1:low_battery", component="Battery", cause_id="sensor:M1:Power:stuck", context={"time": 2.0})

	result = score_cascades(
		reporter.events,
		[
			{
				"root_fault": "sensor:M1:Power:stuck",
				"source": "root_fault",
				"chain_effects": ["battery:M1:low_battery", "machine:M1:production_blocked"],
			}
		],
		window_end=20.0,
		poll_interval=5.0,
	)

	row = result["per_fault"][0]
	assert row["first_effect_at"] == 2.0
	assert row["manifestation_latency"] == 1.0
	assert row["containable"] is False
	assert row["contained"] is False
	assert result["metrics"]["n_structurally_cascading"] == 1
	assert result["metrics"]["n_containable"] == 0
	assert result["metrics"]["contained_rate_over_containable"] is None


def test_slow_chain_effect_is_containable_but_cascaded() -> None:
	reporter = EventReporter()
	reporter.root_fault("sensor:M1:Power:stuck", context={"time": 1.0})
	reporter.physical_state("battery:M1:low_battery", component="Battery", cause_id="sensor:M1:Power:stuck", context={"time": 8.0})

	result = score_cascades(
		reporter.events,
		[
			{
				"root_fault": "sensor:M1:Power:stuck",
				"source": "root_fault",
				"chain_effects": ["battery:M1:low_battery", "machine:M1:production_blocked"],
			}
		],
		window_end=20.0,
		poll_interval=5.0,
	)

	row = result["per_fault"][0]
	assert row["first_effect_at"] == 8.0
	assert row["manifestation_latency"] == 7.0
	assert row["containable"] is True
	assert row["contained"] is False
	assert result["metrics"]["n_containable"] == 1
	assert result["metrics"]["n_contained_given_containable"] == 0
	assert result["metrics"]["contained_rate_over_containable"] == 0.0


def test_no_chain_effect_is_contained_and_containable() -> None:
	reporter = EventReporter()
	reporter.root_fault("sensor:M1:Power:stuck", context={"time": 1.0})
	reporter.fault_resolved("sensor:M1:Power:stuck", context={"time": 9.0})

	result = score_cascades(
		reporter.events,
		[
			{
				"root_fault": "sensor:M1:Power:stuck",
				"source": "root_fault",
				"chain_effects": ["battery:M1:low_battery", "machine:M1:production_blocked"],
			}
		],
		window_end=20.0,
		poll_interval=5.0,
	)

	row = result["per_fault"][0]
	assert row["first_effect_at"] is None
	assert row["manifestation_latency"] is None
	assert row["containable"] is True
	assert row["contained"] is True
	assert result["metrics"]["n_containable"] == 1
	assert result["metrics"]["n_contained_given_containable"] == 1
	assert result["metrics"]["contained_rate_over_containable"] == 1.0


	reporter = EventReporter()
	reporter.root_fault("sensor:M1:Power:stuck", context={"time": 1.0})
	reporter.physical_state("battery:M1:low_battery", component="Battery", cause_id="sensor:M1:Power:stuck", context={"time": 3.0})
	reporter.derived_issue(
		"machine:M1:production_blocked",
		component="Battery",
		cause_id="battery:M1:dead_battery",
		context={"time": 7.0},
	)
	reporter.fault_resolved("sensor:M1:Power:stuck", context={"time": 12.0})

	result = score_cascades(
		reporter.events,
		[
			{
				"root_fault": "sensor:M1:Power:stuck",
				"source": "root_fault",
				"chain_effects": ["battery:M1:low_battery", "machine:M1:production_blocked"],
			}
		],
		window_end=20.0,
		poll_interval=5.0,
	)

	assert result["per_fault"][0]["polls_to_resolution"] == 3
	assert result["per_fault"][0]["depth"] == 2
	assert result["per_fault"][0]["contained"] is False
	assert result["per_fault"][0]["reached_effects"] == ["battery:M1:low_battery", "machine:M1:production_blocked"]
	assert result["metrics"]["n_cascaded"] == 1


	reporter = EventReporter()
	reporter.root_fault("sensor:M1:Temperature:stuck", context={"time": 1.0})
	reporter.physical_state(
		"temperature:M1:overheating",
		component="Temperature",
		cause_id="sensor:M1:Temperature:stuck",
		context={"time": 3.0},
	)

	result = score_cascades(
		reporter.events,
		[
			{
				"root_fault": "sensor:M1:Temperature:stuck",
				"source": "root_fault",
				"chain_effects": ["temperature:M1:overheating"],
			}
		],
		window_end=20.0,
		poll_interval=5.0,
	)

def test_belt_neighbor_chain_effect_is_resolved_and_counted() -> None:
	# A sensor fault on Paint-Booth whose chain reaches the incoming belt. The
	# belt rung "belt:<from_node_id>:<machine_id>:handoff_blocked" cannot be
	# resolved from the root id alone, so neighbours are recovered from the belt
	# events in the window. Without the neighbour map the rung stays a literal
	# placeholder and the concrete belt event below never matches.
	root_fault = "sensor:Paint-Booth:Power:stuck"
	reporter = EventReporter()
	reporter.root_fault(root_fault, context={"time": 1.0})
	reporter.derived_issue(
		"belt:Assembly:Paint-Booth:handoff_blocked",
		component="Belt",
		cause_id=root_fault,
		context={"time": 6.0},
	)

	neighbors: dict[str, tuple[str | None, str | None]] = {"Paint-Booth": ("Assembly", "Coating")}
	ground_truth = ground_truth_for_root_faults([root_fault], neighbors)
	chain_effects = ground_truth[0]["chain_effects"]
	assert isinstance(chain_effects, list)
	assert "belt:Assembly:Paint-Booth:handoff_blocked" in chain_effects

	# Sanity check: without neighbours the rung is left unresolved, so the same
	# belt event would not be counted -- this is exactly what the fix repairs.
	unresolved = ground_truth_for_root_faults([root_fault])
	unresolved_chain_effects = unresolved[0]["chain_effects"]
	assert isinstance(unresolved_chain_effects, list)
	assert "belt:Assembly:Paint-Booth:handoff_blocked" not in unresolved_chain_effects
	assert "belt:<from_node_id>:Paint-Booth:handoff_blocked" in unresolved_chain_effects

	result = score_cascades(reporter.events, ground_truth, window_end=20.0, poll_interval=5.0)

	assert result["per_fault"][0]["depth"] >= 1
	assert "belt:Assembly:Paint-Booth:handoff_blocked" in result["per_fault"][0]["reached_effects"]
	assert result["per_fault"][0]["contained"] is False
