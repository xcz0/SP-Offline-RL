"""Runtime component builders for training/evaluation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from omegaconf import DictConfig
from tianshou.data import ReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from tianshou.utils.space_info import ActionSpaceInfo, ObservationSpaceInfo, SpaceInfo

from src.algos.registry import get_algo_factory
from src.core.compile import build_compile_config, compile_model_bundle
from src.core.types import DatasetDict
from src.data.obs_act_buffer import ObsActBuffer
from src.models.registry import get_model_factory
from src.runners.data_builder import (
    build_obs_act_buffer_from_prepared,
    build_replay_buffer_from_prepared,
    load_prepared_dataset,
    obs_norm_stats_as_lists,
)
from src.runners.norm_builder import apply_obs_norm_to_envs


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
    model_bundle = compile_model_bundle(model_bundle, build_compile_config(cfg))
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
    model_bundle = compile_model_bundle(model_bundle, build_compile_config(cfg))
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


def prepare_bc_sim_components(
    cfg: DictConfig,
    device: str,
    *,
    seed: int,
    include_train_buffer: bool,
) -> BCSimComponents:
    """Prepare bc_il data/model stack for simulator-backed flows."""

    prepared = load_prepared_dataset(cfg, env=None, algo_name="bc_il")
    obs_act_data = prepared.to_dict()

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
        build_obs_act_buffer_from_prepared(prepared, seed=seed)
        if include_train_buffer
        else None
    )
    obs_norm_mean, obs_norm_var = obs_norm_stats_as_lists(prepared)
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
    obs_norm_mean: np.ndarray | None = None
    obs_norm_var: np.ndarray | None = None

    needs_data = include_train_buffer or bool(cfg.data.obs_norm)
    if needs_data:
        prepared = load_prepared_dataset(cfg, env=env, algo_name=algo_name)
        obs_norm_mean = prepared.obs_norm_mean
        obs_norm_var = prepared.obs_norm_var
        if include_train_buffer:
            prepared_keys = set(prepared.arrays.keys())
            if prepared_keys == {"obs", "act"}:
                train_buffer = build_obs_act_buffer_from_prepared(prepared, seed=seed)
            else:
                train_buffer = build_replay_buffer_from_prepared(
                    prepared,
                    cfg.get("buffer_size"),
                )

    test_envs = make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
    test_envs = apply_obs_norm_to_envs(test_envs, obs_norm_mean, obs_norm_var)
    test_envs.seed(seed)
    algorithm = build_algorithm(cfg, env, device)
    return GymComponents(
        env=env,
        test_envs=test_envs,
        algorithm=algorithm,
        train_buffer=train_buffer,
    )
