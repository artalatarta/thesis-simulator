import math
from collections.abc import Iterable
from typing import Any

from cps.config import AGENT_POLL_INTERVAL
from cps.core.reporting import ReportedEvent
from cps.evaluation.event_records import event_time, first_event_times
from cps.evaluation.ground_truth import root_fault_ground_truth, string_list

def score_cascades(
	events: Iterable[ReportedEvent],
	ground_truth: Iterable[dict[str, object]],
	*,
	window_end: float,
	poll_interval: float = AGENT_POLL_INTERVAL,
) -> dict[str, Any]:
	event_list = sorted(list(events), key=lambda event: event_time(event, default=0.0))
	injection_times = first_event_times(event_list, "root_fault")
	resolution_times = first_event_times(event_list, "fault_resolved")
	per_fault = [
		_score_fault_cascade(
			truth,
			event_list,
			injection_times,
			resolution_times,
			window_end=window_end,
			poll_interval=poll_interval,
		)
		for truth in root_fault_ground_truth(ground_truth)
	]
	n_containable = sum(row["containable"] is True for row in per_fault)
	n_contained_given_containable = sum(row["contained"] is True and row["containable"] is True for row in per_fault)
	return {
		"per_fault": per_fault,
		"metrics": {
			"root_faults": len(per_fault),
			"n_contained": sum(row["contained"] is True for row in per_fault),
			"n_cascaded": sum(row["contained"] is False for row in per_fault),
			"n_containable": n_containable,
			"n_structurally_cascading": sum(row["containable"] is False for row in per_fault),
			"n_contained_given_containable": n_contained_given_containable,
			"contained_rate_over_containable": (
				n_contained_given_containable / n_containable if n_containable > 0 else None
			),
		},
	}


def _score_fault_cascade(
	truth: dict[str, object],
	events: list[ReportedEvent],
	injection_times: dict[str, float],
	resolution_times: dict[str, float],
	*,
	window_end: float,
	poll_interval: float,
) -> dict[str, Any]:
	root_fault = str(truth["root_fault"])
	injected_at = injection_times.get(root_fault)
	resolved_at = resolution_times.get(root_fault)
	chain_effects = string_list(truth.get("chain_effects"))
	matched_events = []
	if injected_at is not None:
		end = resolved_at if resolved_at is not None else window_end
		matched_events = [
			event
			for event in events
			if event.kind in {"physical_state", "derived_issue"}
			if event.identifier in chain_effects
			if injected_at <= event_time(event, default=-math.inf) <= end
		]
	first_effect_at = min((event_time(event, default=math.inf) for event in matched_events), default=None)
	manifestation_latency = (
		first_effect_at - injected_at if first_effect_at is not None and injected_at is not None else None
	)
	containable = first_effect_at is None or (
		manifestation_latency is not None and manifestation_latency >= poll_interval
	)
	depth = max((chain_effects.index(event.identifier) + 1 for event in matched_events), default=0)
	return {
		"root_fault": root_fault,
		"injected_at": injected_at,
		"resolved_at": resolved_at,
		"first_effect_at": first_effect_at,
		"manifestation_latency": manifestation_latency,
		"containable": containable,
		"polls_to_resolution": _polls_to_resolution(injected_at, resolved_at, poll_interval),
		"depth": depth,
		"contained": depth == 0,
		"reached_effects": _unique(event.identifier for event in matched_events),
	}


def _polls_to_resolution(injected_at: float | None, resolved_at: float | None, poll_interval: float) -> int | None:
	if injected_at is None or resolved_at is None:
		return None
	return max(0, math.ceil((resolved_at - injected_at) / poll_interval))


def _unique(values: Iterable[str]) -> list[str]:
	seen: set[str] = set()
	unique: list[str] = []
	for value in values:
		if value in seen:
			continue
		seen.add(value)
		unique.append(value)
	return unique
