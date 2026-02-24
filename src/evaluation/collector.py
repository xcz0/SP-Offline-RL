"""Custom collector that feeds simulator evaluation into Tianshou trainer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from tianshou.data import CollectStats
from tianshou.data.stats import SequenceSummaryStats

from src.evaluation.pipeline import evaluate_composite_with_simulator
from src.evaluation.types import CompositeEvalResult


@dataclass(slots=True)
class SimulationEvalCache:
    epoch: int
    result: CompositeEvalResult


class SimulationEvalCollector:
    """Collector-like adapter for offline trainer periodic simulator eval."""

    def __init__(
        self,
        *,
        policy: Any,
        sim_eval_cfg: Any,
        action_max: float,
        eval_every_n_epoch: int,
    ) -> None:
        self._policy = policy
        self._sim_eval_cfg = sim_eval_cfg
        self._action_max = float(action_max)
        self._eval_every_n_epoch = max(1, int(eval_every_n_epoch))
        self._current_epoch = 0
        self._cache: SimulationEvalCache | None = None

    @property
    def latest_result(self) -> CompositeEvalResult | None:
        if self._cache is None:
            return None
        return self._cache.result

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def reset(self, reset_buffer: bool = False, reset_stats: bool = False) -> None:
        _ = reset_buffer
        if reset_stats:
            self._cache = None

    def _should_run_eval(self) -> bool:
        if self._cache is None:
            return True
        if self._current_epoch <= 0:
            return False
        return self._current_epoch % self._eval_every_n_epoch == 0

    def collect(self, n_episode: int | None = None, render: float = 0.0) -> CollectStats:
        _ = render
        count = max(1, int(n_episode or 1))
        started = time.time()
        if self._should_run_eval():
            result = evaluate_composite_with_simulator(
                policy=self._policy,
                sim_eval_cfg=self._sim_eval_cfg,
                action_max=self._action_max,
            )
            self._cache = SimulationEvalCache(epoch=self._current_epoch, result=result)

        score = float("nan")
        if self._cache is not None:
            score = float(self._cache.result.summary.get("score_mean", float("nan")))

        returns = np.full(count, score, dtype=np.float32)
        lens = np.ones(count, dtype=np.int64)
        collect_time = max(1e-8, time.time() - started)
        return CollectStats(
            n_collected_episodes=count,
            n_collected_steps=count,
            collect_time=collect_time,
            collect_speed=float(count) / collect_time,
            returns=returns,
            returns_stat=SequenceSummaryStats.from_sequence(returns),
            lens=lens,
            lens_stat=SequenceSummaryStats.from_sequence(lens),
        )
