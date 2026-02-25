"""Shared helpers and compatibility wrappers for train/eval pipelines."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Collector, CollectStats

from src.core.exceptions import ConfigurationError
from src.logging.metrics import collect_stats_to_metrics
from src.runners import runtime_builder

BCSimComponents = runtime_builder.BCSimComponents
GymComponents = runtime_builder.GymComponents


def make_test_envs(task: str, num_test_envs: int):
    return runtime_builder.make_test_envs(task, num_test_envs)


def build_algorithm(cfg: DictConfig, env: Any, device: str):
    return runtime_builder.build_algorithm(cfg, env, device)


def infer_action_bounds_from_dataset(act):
    return runtime_builder.infer_action_bounds_from_dataset(act)


def build_space_info_from_obs_act(
    obs_act_data,
    *,
    action_low: float | None = None,
    action_high: float | None = None,
):
    return runtime_builder.build_space_info_from_obs_act(
        obs_act_data,
        action_low=action_low,
        action_high=action_high,
    )


def build_algorithm_from_space_info(
    cfg: DictConfig,
    space_info: Any,
    action_space: Any,
    device: str,
):
    return runtime_builder.build_algorithm_from_space_info(
        cfg,
        space_info,
        action_space,
        device,
    )


def prepare_bc_sim_components(
    cfg: DictConfig,
    device: str,
    *,
    seed: int,
    include_train_buffer: bool,
) -> BCSimComponents:
    return runtime_builder.prepare_bc_sim_components(
        cfg,
        device,
        seed=seed,
        include_train_buffer=include_train_buffer,
    )


def prepare_gym_components(
    cfg: DictConfig,
    device: str,
    *,
    seed: int,
    include_train_buffer: bool,
) -> GymComponents:
    return runtime_builder.prepare_gym_components(
        cfg,
        device,
        seed=seed,
        include_train_buffer=include_train_buffer,
    )


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
    if perf_cfg is not None:
        sim_cfg["eval_workers"] = int(perf_cfg.get("eval_workers", 1))
    else:
        sim_cfg.setdefault("eval_workers", 1)

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
