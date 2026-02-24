"""Types for spaced-repetition simulator evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True, frozen=True)
class EvalTarget:
    """Selected target card for one simulation episode."""

    user_id: int
    card_id: int
    occurrences: int
    warmup_occurrence: int
    warmup_end_day_offset: float


@dataclass(slots=True)
class ReplayEvalResult:
    """Replay evaluation result with per-target summaries and trajectory."""

    final_metrics: pd.DataFrame
    trajectory: pd.DataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class PolicyEvalResult:
    """Policy-driven simulation evaluation result."""

    per_target_metrics: pd.DataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class CompositeEvalResult:
    """Combined replay/policy result and score summary."""

    policy_result: PolicyEvalResult | None
    replay_result: ReplayEvalResult | None
    summary: dict[str, Any]

