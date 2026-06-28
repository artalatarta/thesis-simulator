"""Compact structured debug logging for the monitoring layer."""

import json
import logging

from cps.agents.contracts import MonitoringReport

logger = logging.getLogger("cps.agents.monitoring")
_EXTRA = {"component": "MonitoringAgents"}


def debug(tag: str, *, exc_info: bool = False, **fields: object) -> None:
	"""Log ``tag key=value ...`` with JSON-formatted values at DEBUG level."""
	if not logger.isEnabledFor(logging.DEBUG):
		return
	formatted = " ".join(f"{key}={_format(value)}" for key, value in fields.items())
	logger.debug("%s %s", tag, formatted, exc_info=exc_info, extra=_EXTRA)


def _format(value: object) -> str:
	if isinstance(value, str):
		return value
	try:
		return json.dumps(value, sort_keys=True, default=str)
	except TypeError:
		return str(value)


def report_summary(report: MonitoringReport) -> dict[str, object]:
	return {
		"report_id": report.report_id,
		"agent": report.agent_name,
		"component": report.component,
		"diagnosis": report.diagnosis,
		"diagnosis_id": report.diagnosis_id,
		"action": report.recommended_action,
		"confidence": report.confidence,
		"evidence": list(report.evidence),
		"time": report.time,
	}
