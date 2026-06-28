from collections.abc import Iterable

from cps.agents.contracts import ActionLabel, MonitoringReport

PASSIVE_ACTIONS: frozenset[ActionLabel] = frozenset({"wait_for_more_evidence"})


def unique_actionable_reports(
	reports: Iterable[MonitoringReport],
	*,
	excluded_actions: set[str] | frozenset[str] = frozenset(),
) -> list[MonitoringReport]:
	unique: list[MonitoringReport] = []
	seen: set[tuple[str | None, str, str, str]] = set()
	for report in reports:
		if report.diagnosis_id is None:
			continue
		if report.recommended_action in excluded_actions:
			continue
		evidence = report.evidence[0] if report.evidence else report.report_id
		key = (report.machine_id, report.diagnosis_id, report.recommended_action, evidence)
		if key in seen:
			continue
		seen.add(key)
		unique.append(report)
	return unique
