"""Strongly typed runner results and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


def to_builtin(value: Any) -> Any:
    """Convert numpy/dataclass/path values into JSON-serializable builtins."""

    if is_dataclass(value):
        return {f.name: to_builtin(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(slots=True)
class EvalMetrics:
    """Aggregate evaluation metrics."""

    test_reward_mean: float
    test_reward_std: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(self)


@dataclass(slots=True)
class TrainingMetrics:
    """Aggregate training metrics."""

    best_reward: float
    best_reward_std: float
    update_step: int
    train_step: int
    test_step: int
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(self)


@dataclass(slots=True)
class SimEvalArtifacts:
    """Persisted simulator evaluation summaries and artifact paths."""

    summary: dict[str, Any] = field(default_factory=dict)
    composite_summary_path: str | None = None
    policy_metrics_path: str | None = None
    policy_summary_path: str | None = None
    replay_final_metrics_path: str | None = None
    replay_trajectory_path: str | None = None
    replay_summary_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(self)


@dataclass(slots=True)
class TrainingResult:
    """Top-level training result payload."""

    mode: Literal["train", "watch"]
    log_path: str
    run_name: str
    checkpoint_path: str | None
    evaluation: EvalMetrics
    training: TrainingMetrics | None = None
    sim_eval: SimEvalArtifacts | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(self)


@dataclass(slots=True)
class EvaluationResult:
    """Top-level evaluation result payload."""

    mode: Literal["gym", "sim", "replay"]
    checkpoint_path: str | None
    evaluation: EvalMetrics
    sim_eval: SimEvalArtifacts | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(self)
