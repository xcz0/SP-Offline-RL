"""Shared helpers for train/eval runner pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Collector, CollectStats, ReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.utils.space_info import ActionSpaceInfo, ObservationSpaceInfo, SpaceInfo

from src.algos.registry import get_algo_factory
from src.core.exceptions import ConfigurationError
from src.core.seed import seed_vector_env
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


@dataclass(slots=True)
class BCSimComponents:
    """Prepared components for bc_il simulator-backed train/eval flow."""

    obs_act_data: DatasetDict
    algorithm: Any
    action_low: float
    action_high: float
    obs_norm_mean: list[float] | None
    obs_norm_var: list[float] | None
    train_buffer: ObsActBuffer | None


@dataclass(slots=True)
class GymComponents:
    """Prepared components for gym-backed train/eval flow."""

    env: Any
    test_envs: BaseVectorEnv
    algorithm: Any
    train_buffer: ReplayBuffer | ObsActBuffer | None


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


def build_sim_eval_cfg(
    cfg: DictConfig,
    *,
    obs_norm_mean: list[float] | None = None,
    obs_norm_var: list[float] | None = None,
) -> dict[str, Any]:
    """Resolve simulator-eval config and inject optional obs normalization stats."""

    sim_eval_cfg = cfg.get("sim_eval")
    if sim_eval_cfg is None:
        raise ConfigurationError("sim_eval config is required for simulator mode.")

    sim_cfg = OmegaConf.to_container(sim_eval_cfg, resolve=True)
    if not isinstance(sim_cfg, dict):
        raise ConfigurationError("sim_eval config must resolve to a dictionary.")

    if obs_norm_mean is not None:
        sim_cfg["obs_norm_mean"] = obs_norm_mean
        sim_cfg["obs_norm_var"] = obs_norm_var
    return sim_cfg


def prepare_bc_sim_components(
    cfg: DictConfig,
    device: str,
    *,
    seed: int,
    include_train_buffer: bool,
) -> BCSimComponents:
    """Prepare bc_il data/model stack for simulator-backed flows."""

    obs_act_data = build_obs_act_dataset_from_cfg(cfg, env=None)
    obs_norm_mean: list[float] | None = None
    obs_norm_var: list[float] | None = None
    if bool(cfg.data.obs_norm):
        normalized_obs, obs_rms = normalize_obs_array(obs_act_data["obs"])
        obs_act_data = dict(obs_act_data)
        obs_act_data["obs"] = normalized_obs
        obs_norm_mean = np.asarray(obs_rms.mean, dtype=np.float32).tolist()
        obs_norm_var = np.asarray(obs_rms.var, dtype=np.float32).tolist()

    action_low, action_high = infer_action_bounds_from_dataset(obs_act_data["act"])
    space_info, action_space = build_space_info_from_obs_act(
        obs_act_data,
        action_low=action_low,
        action_high=action_high,
    )
    algorithm = build_algorithm_from_space_info(
        cfg,
        space_info=space_info,
        action_space=action_space,
        device=device,
    )
    train_buffer = (
        build_obs_act_buffer_from_dataset(obs_act_data, seed=seed)
        if include_train_buffer
        else None
    )
    return BCSimComponents(
        obs_act_data=obs_act_data,
        algorithm=algorithm,
        action_low=action_low,
        action_high=action_high,
        obs_norm_mean=obs_norm_mean,
        obs_norm_var=obs_norm_var,
        train_buffer=train_buffer,
    )


def prepare_gym_components(
    cfg: DictConfig,
    device: str,
    *,
    seed: int,
    include_train_buffer: bool,
) -> GymComponents:
    """Prepare gym env, test envs, algorithm, and optional train buffer."""

    env = gym.make(str(cfg.env.task))
    algo_name = str(cfg.algo.name)
    train_buffer: ReplayBuffer | ObsActBuffer | None = None
    obs_act_data: DatasetDict | None = None

    if include_train_buffer:
        if algo_name == "bc_il":
            obs_act_data = build_obs_act_dataset_from_cfg(cfg, env)
            train_buffer = build_obs_act_buffer_from_dataset(obs_act_data, seed)
        else:
            train_buffer = build_replay_buffer_from_cfg(
                cfg,
                env,
                cfg.get("buffer_size"),
            )

    test_envs = make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
    if bool(cfg.data.obs_norm):
        if algo_name == "bc_il":
            if obs_act_data is None:
                obs_act_data = build_obs_act_dataset_from_cfg(cfg, env)
            obs_act_data, test_envs = apply_obs_norm_to_obs_act_data(obs_act_data, test_envs)
            if include_train_buffer:
                train_buffer = build_obs_act_buffer_from_dataset(obs_act_data, seed)
        else:
            replay_for_norm = train_buffer
            if replay_for_norm is None:
                replay_for_norm = build_replay_buffer_from_cfg(
                    cfg,
                    env,
                    cfg.get("buffer_size"),
                )
            replay_for_norm, test_envs = apply_obs_norm(replay_for_norm, test_envs)
            if include_train_buffer:
                train_buffer = replay_for_norm

    seed_vector_env(test_envs, seed)
    algorithm = build_algorithm(cfg, env, device)
    return GymComponents(
        env=env,
        test_envs=test_envs,
        algorithm=algorithm,
        train_buffer=train_buffer,
    )


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
