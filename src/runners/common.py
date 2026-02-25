"""Shared helpers for train/eval pipelines."""

from __future__ import annotations

import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Collector, CollectStats

from src.core.exceptions import ConfigurationError
from src.logging.metrics import collect_stats_to_metrics


def _resolve_eval_workers(value: Any) -> int:
    default_workers = max(1, int(os.cpu_count() or 1))
    if value is None:
        return default_workers

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto"}:
            return default_workers

    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"perf.eval_workers must be an integer or 'auto', got: {value!r}"
        ) from exc
    if workers <= 0:
        return default_workers
    return workers


def build_sim_eval_cfg(
    cfg: DictConfig,
    *,
    obs_norm_mean: list[float] | None = None,
    obs_norm_var: list[float] | None = None,
) -> dict[str, Any]:
    """Resolve simulator-eval config and inject optional runtime fields."""

    sim_eval_cfg = cfg.get("sim_eval")
    if sim_eval_cfg is None:
        raise ConfigurationError("sim_eval config is required for simulator mode.")

    sim_cfg = OmegaConf.to_container(sim_eval_cfg, resolve=True)
    if not isinstance(sim_cfg, dict):
        raise ConfigurationError("sim_eval config must resolve to a dictionary.")

    perf_cfg = cfg.get("perf")
    perf_workers = perf_cfg.get("eval_workers") if perf_cfg is not None else None
    sim_cfg["eval_workers"] = _resolve_eval_workers(perf_workers)

    if obs_norm_mean is not None:
        sim_cfg["obs_norm_mean"] = obs_norm_mean
        sim_cfg["obs_norm_var"] = obs_norm_var
    return sim_cfg


def load_policy_state(algorithm: Any, checkpoint_path: str, device: str) -> None:
    algorithm.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )


def collect_eval_metrics(
    collector: Collector[CollectStats],
    num_episodes: int,
    render: float,
) -> tuple[CollectStats, dict[str, float]]:
    collector.reset()
    stats = collector.collect(
        n_episode=num_episodes,
        render=render,
    )
    return stats, collect_stats_to_metrics(stats)
