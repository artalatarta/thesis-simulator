"""Symptom state and persistence detection for belt flow diagnostics."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass
class BeltSymptomTracker:
	identifier: str
	active_since: float | None = None
	occurrences: int = 0
	cause_id: str | None = None

	def observe(self, now: float, *, cause_id: str | None = None) -> None:
		if self.active_since is None:
			self.active_since = now
			self.occurrences = 0
		self.occurrences += 1
		if cause_id is not None:
			self.cause_id = cause_id

	def clear(self) -> None:
		self.active_since = None
		self.occurrences = 0
		self.cause_id = None

	def duration(self, now: float) -> float:
		return 0.0 if self.active_since is None else now - self.active_since


class SymptomRegistry:
	def __init__(self, identifier_for_issue: Callable[[str], str]) -> None:
		self._identifier_for_issue = identifier_for_issue
		self._symptoms: dict[str, BeltSymptomTracker] = {}

	def get(self, issue: str) -> BeltSymptomTracker:
		if issue not in self._symptoms:
			self._symptoms[issue] = BeltSymptomTracker(self._identifier_for_issue(issue))
		return self._symptoms[issue]

	def find(self, issue: str) -> BeltSymptomTracker | None:
		return self._symptoms.get(issue)

	def observe(self, issue: str, now: float, *, cause_id: str | None = None) -> BeltSymptomTracker:
		symptom = self.get(issue)
		symptom.observe(now, cause_id=cause_id)
		return symptom

	def active_ids(self, issues: Iterable[str]) -> list[str]:
		included = set(issues)
		return [symptom.identifier for issue, symptom in self._symptoms.items() if issue in included and symptom.active_since is not None]

	def clear(self, *issues: str) -> None:
		for issue in issues:
			symptom = self.find(issue)
			if symptom is not None:
				symptom.clear()


class BottleneckDetector:
	def __init__(self, symptoms: SymptomRegistry) -> None:
		self.symptoms = symptoms
		self.reported = False
		self.active_since: float | None = None

	def persistent_symptom(
		self,
		now: float,
		ordered_issues: Iterable[str],
		*,
		detection_delay: float,
		repeated_failure_threshold: int,
	) -> BeltSymptomTracker | None:
		for issue in ordered_issues:
			symptom = self.symptoms.find(issue)
			if symptom is None or symptom.active_since is None:
				continue
			if symptom.duration(now) >= detection_delay:
				return symptom
			if issue == "handoff_blocked" and symptom.occurrences >= repeated_failure_threshold:
				return symptom
		return None

	def mark_reported(self, symptom: BeltSymptomTracker) -> None:
		if self.active_since is None:
			self.active_since = symptom.active_since
		self.reported = True

	def reset(self) -> None:
		self.reported = False
		self.active_since = None
