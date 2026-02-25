"""Lightweight runtime profiling helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class PerfTracker:
    """Collects named wall-clock durations when profiling is enabled."""

    enabled: bool = False
    _durations: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._durations[name] = self._durations.get(name, 0.0) + elapsed

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in sorted(self._durations.items())}
