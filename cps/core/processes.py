"""Helpers for nullable SimPy process references."""

from typing import TypeGuard

import simpy


def process_is_alive(process: simpy.Process | None) -> TypeGuard[simpy.Process]:
	return process is not None and process.is_alive
