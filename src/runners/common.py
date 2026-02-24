"""Shared helpers for train/eval runner pipelines."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tianshou.data import Collector, CollectStats, ReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.utils.space_info import SpaceInfo

from src.algos.registry import get_algo_factory
from src.data.dataset_adapter import build_dataset_adapter
from src.data.replay_buffer_builder import build_replay_buffer
from src.data.schema import validate_against_env, validate_and_standardize_dataset
from src.data.transforms import normalize_obs_in_replay_buffer
from src.logging.metrics import collect_stats_to_metrics
from src.models.registry import get_model_factory


def make_test_envs(task: str, num_test_envs: int) -> BaseVectorEnv:
    if num_test_envs <= 1:
        return DummyVectorEnv([lambda: gym.make(task)])
    return SubprocVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])


def build_algorithm(cfg: DictConfig, env: Any, device: str):
    space_info = SpaceInfo.from_env(env)
    model_factory = get_model_factory(str(cfg.model.name))
    model_bundle = model_factory.build(cfg.model, space_info, device)
    algo_factory = get_algo_factory(str(cfg.algo.name))
    return algo_factory.build(cfg.algo, env, model_bundle, device)


def build_replay_buffer_from_cfg(
    cfg: DictConfig,
    env: Any,
    buffer_size: int | None,
) -> ReplayBuffer:
    adapter = build_dataset_adapter(cfg.data)
    data = validate_and_standardize_dataset(adapter.load())
    validate_against_env(data, env)
    return build_replay_buffer(data, buffer_size, validate=False)


def apply_obs_norm(
    replay_buffer: ReplayBuffer,
    test_envs: BaseVectorEnv,
) -> tuple[ReplayBuffer, BaseVectorEnv]:
    replay_buffer, obs_rms = normalize_obs_in_replay_buffer(replay_buffer)
    normalized_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    normalized_envs.set_obs_rms(obs_rms)
    return replay_buffer, normalized_envs


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
