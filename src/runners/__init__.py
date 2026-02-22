"""Train/eval runners."""

from src.runners.evaluator import run_evaluation
from src.runners.trainer import run_offline_training

__all__ = ["run_offline_training", "run_evaluation"]
