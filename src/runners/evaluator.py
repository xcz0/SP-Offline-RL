"""Standalone policy evaluation from checkpoint."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.utils.space_info import SpaceInfo

from src.algos.registry import get_algo_factory
from src.core.seed import seed_vector_env, set_global_seed
from src.data.dataset_adapter import build_dataset_adapter
from src.data.replay_buffer_builder import build_replay_buffer
from src.data.schema import validate_against_env, validate_and_standardize_dataset
from src.data.transforms import normalize_obs_in_replay_buffer
from src.logging.metrics import collect_stats_to_metrics
from src.models.registry import get_model_factory
from src.utils.hydra import resolve_device


def _make_test_envs(task: str, num_test_envs: int) -> BaseVectorEnv:
    return SubprocVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])


def run_evaluation(cfg: DictConfig, checkpoint_path: str) -> dict[str, Any]:
    """Evaluate a policy checkpoint and return aggregate metrics."""

    device = resolve_device(str(cfg.device))

    env = gym.make(str(cfg.env.task))
    test_envs: BaseVectorEnv | None = None
    try:
        space_info = SpaceInfo.from_env(env)
        set_global_seed(int(cfg.seed))

        test_envs = _make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
        if bool(cfg.data.obs_norm):
            adapter = build_dataset_adapter(cfg.data)
            data = validate_and_standardize_dataset(adapter.load())
            validate_against_env(data, env)
            replay_buffer = build_replay_buffer(data, cfg.get("buffer_size"))
            replay_buffer, obs_rms = normalize_obs_in_replay_buffer(replay_buffer)
            test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
            test_envs.set_obs_rms(obs_rms)

        seed_vector_env(test_envs, int(cfg.seed))

        model_factory = get_model_factory(str(cfg.model.name))
        model_bundle = model_factory.build(cfg.model, space_info, device)

        algo_factory = get_algo_factory(str(cfg.algo.name))
        algorithm = algo_factory.build(cfg.algo, env, model_bundle, device)

        algorithm.load_state_dict(torch.load(checkpoint_path, map_location=device))

        collector = Collector[CollectStats](algorithm, test_envs)
        collector.reset()
        stats = collector.collect(
            n_episode=int(cfg.env.num_test_envs),
            render=float(cfg.env.render),
        )

        metrics = collect_stats_to_metrics(stats)
        return {
            "checkpoint": checkpoint_path,
            "episodes": int(cfg.env.num_test_envs),
            **metrics,
        }
    finally:
        if test_envs is not None:
            test_envs.close()
        env.close()
