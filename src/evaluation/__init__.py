"""Spaced-repetition simulator evaluation helpers."""

from src.evaluation.collector import SimulationEvalCollector
from src.evaluation.policy_evaluator import evaluate_policy_with_simulator
from src.evaluation.replay_evaluator import evaluate_replay_with_simulator
from src.evaluation.scoring import compute_score_from_metrics
from src.evaluation.types import (
    CompositeEvalResult,
    EvalTarget,
    PolicyEvalResult,
    ReplayEvalResult,
)

__all__ = [
    "CompositeEvalResult",
    "EvalTarget",
    "PolicyEvalResult",
    "ReplayEvalResult",
    "SimulationEvalCollector",
    "compute_score_from_metrics",
    "evaluate_policy_with_simulator",
    "evaluate_replay_with_simulator",
]
