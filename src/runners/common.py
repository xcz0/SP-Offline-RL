"""Shared helpers for train/eval runner pipelines."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig
from tianshou.data import Collector, CollectStats, ReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.utils.space_info import ActionSpaceInfo, ObservationSpaceInfo, SpaceInfo

from src.algos.registry import get_algo_factory
from src.core.types import DatasetDict
from src.data.dataset_adapter import build_dataset_adapter
from src.data.obs_act_buffer import ObsActBuffer
from src.data.replay_buffer_builder import build_replay_buffer
from src.data.schema import (
    validate_against_env,
    validate_and_standardize_dataset,
    validate_obs_act_dataset,
)
from src.data.transforms import normalize_obs_array, normalize_obs_in_replay_buffer
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


class _StaticSpaceEnv:
    """Minimal env-like object exposing observation/action spaces."""

    def __init__(self, observation_space: Box, action_space: Box) -> None:
        self.observation_space = observation_space
        self.action_space = action_space


def _shape_to_tuple(shape: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(v) for v in shape)


def infer_action_bounds_from_dataset(act: np.ndarray) -> tuple[float, float]:
    """Infer scalar action bounds from offline action labels."""

    low = float(np.min(act))
    high = float(np.max(act))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Action array contains non-finite values.")
    if low == high:
        delta = abs(low) * 0.05 + 1.0
        low -= delta
        high += delta
    return low, high


def build_synthetic_env_from_obs_act(
    obs_act_data: DatasetDict,
    *,
    action_low: float | None = None,
    action_high: float | None = None,
) -> Any:
    """Create an env-like object from obs/act dataset shapes and bounds."""

    obs = obs_act_data["obs"]
    act = obs_act_data["act"]
    obs_shape = tuple(int(v) for v in obs.shape[1:])
    act_shape = tuple(int(v) for v in act.shape[1:])

    inferred_low, inferred_high = infer_action_bounds_from_dataset(act)
    low = inferred_low if action_low is None else float(action_low)
    high = inferred_high if action_high is None else float(action_high)
    if high <= low:
        raise ValueError(f"Invalid action bounds: low={low}, high={high}.")

    observation_space = Box(
        low=-np.inf,
        high=np.inf,
        shape=obs_shape,
        dtype=np.float32,
    )
    action_space = Box(
        low=np.full(act_shape, low, dtype=np.float32),
        high=np.full(act_shape, high, dtype=np.float32),
        dtype=np.float32,
    )
    return _StaticSpaceEnv(
        observation_space=observation_space,
        action_space=action_space,
    )


def build_space_info_from_obs_act(
    obs_act_data: DatasetDict,
    *,
    action_low: float | None = None,
    action_high: float | None = None,
) -> tuple[SpaceInfo, Box]:
    """Build Tianshou space info and action space from obs/act dataset."""

    obs = obs_act_data["obs"]
    act = obs_act_data["act"]
    obs_shape = tuple(int(v) for v in obs.shape[1:])
    act_shape = tuple(int(v) for v in act.shape[1:])
    inferred_low, inferred_high = infer_action_bounds_from_dataset(act)
    low = inferred_low if action_low is None else float(action_low)
    high = inferred_high if action_high is None else float(action_high)

    space_info = SpaceInfo(
        observation_info=ObservationSpaceInfo(obs_shape=obs_shape),
        action_info=ActionSpaceInfo(
            action_shape=act_shape,
            min_action=low,
            max_action=high,
        ),
    )
    action_space = Box(
        low=np.full(act_shape, low, dtype=np.float32),
        high=np.full(act_shape, high, dtype=np.float32),
        dtype=np.float32,
    )
    return space_info, action_space


def build_algorithm_from_space_info(
    cfg: DictConfig,
    space_info: SpaceInfo,
    action_space: Box,
    device: str,
):
    """Build algorithm from precomputed space info without gym.make()."""

    model_factory = get_model_factory(str(cfg.model.name))
    model_bundle = model_factory.build(cfg.model, space_info, device)
    algo_factory = get_algo_factory(str(cfg.algo.name))
    env_like = _StaticSpaceEnv(
        observation_space=Box(
            low=-np.inf,
            high=np.inf,
            shape=_shape_to_tuple(space_info.observation_info.obs_shape),
            dtype=np.float32,
        ),
        action_space=action_space,
    )
    return algo_factory.build(cfg.algo, env_like, model_bundle, device)


def build_replay_buffer_from_cfg(
    cfg: DictConfig,
    env: Any,
    buffer_size: int | None,
) -> ReplayBuffer:
    adapter = build_dataset_adapter(cfg.data)
    data = validate_and_standardize_dataset(adapter.load())
    validate_against_env(data, env)
    return build_replay_buffer(data, buffer_size, validate=False)


def build_obs_act_dataset_from_cfg(
    cfg: DictConfig,
    env: Any | None,
) -> DatasetDict:
    adapter = build_dataset_adapter(cfg.data)
    data = validate_obs_act_dataset(adapter.load_obs_act())
    if env is not None:
        validate_against_env(data, env)
    return data


def build_obs_act_buffer_from_dataset(
    obs_act_data: DatasetDict,
    seed: int,
) -> ObsActBuffer:
    return ObsActBuffer(
        obs=obs_act_data["obs"],
        act=obs_act_data["act"],
        seed=seed,
    )


def apply_obs_norm(
    replay_buffer: ReplayBuffer,
    test_envs: BaseVectorEnv,
) -> tuple[ReplayBuffer, BaseVectorEnv]:
    replay_buffer, obs_rms = normalize_obs_in_replay_buffer(replay_buffer)
    normalized_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    normalized_envs.set_obs_rms(obs_rms)
    return replay_buffer, normalized_envs


def apply_obs_norm_to_obs_act_data(
    obs_act_data: DatasetDict,
    test_envs: BaseVectorEnv,
) -> tuple[DatasetDict, BaseVectorEnv]:
    normalized_obs, obs_rms = normalize_obs_array(obs_act_data["obs"])
    normalized_data = dict(obs_act_data)
    normalized_data["obs"] = normalized_obs
    normalized_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    normalized_envs.set_obs_rms(obs_rms)
    return normalized_data, normalized_envs


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
