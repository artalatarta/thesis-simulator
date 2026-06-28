import logging

from cps.core.reporting import EventReporter


def test_reporter_stamps_events_with_its_clock(caplog) -> None:
	reporter = EventReporter(now=lambda: 5.0)
	with caplog.at_level(logging.CRITICAL):
		event = reporter.observation("sensor:M1:Power:low_battery_detected", component="M1")

	assert event.context["time"] == 5.0
	assert reporter.events == [event]


def test_reporter_logs_identifier_metadata(caplog) -> None:
	reporter = EventReporter()
	with caplog.at_level(logging.WARNING):
		event = reporter.derived_issue(
			"machine:M1:production_blocked",
			component="M1",
			cause_id="battery:M1:dead_battery",
		)

	assert event.identifier == "machine:M1:production_blocked"
	assert event.kind == "derived_issue"
	record = caplog.records[-1]
	assert record.message == "machine:M1:production_blocked caused by battery:M1:dead_battery."
	assert record.event_id == "machine:M1:production_blocked"
	assert record.event_kind == "derived_issue"
	assert record.cause_id == "battery:M1:dead_battery"


def test_reporter_emits_fault_resolved_events() -> None:
	reporter = EventReporter()

	event = reporter.fault_resolved("sensor:M1:Power:stuck", component="M1", context={"time": 6.0})

	assert event.kind == "fault_resolved"
	assert event.identifier == "sensor:M1:Power:stuck"
	assert event.context["time"] == 6.0
