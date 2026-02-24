"""Standalone policy evaluation from checkpoint."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from omegaconf import DictConfig
from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv

from src.core.seed import seed_vector_env, set_global_seed
from src.runners.common import (
    apply_obs_norm,
    build_algorithm,
    build_replay_buffer_from_cfg,
    collect_eval_metrics,
    load_policy_state,
    make_test_envs,
)
from src.utils.hydra import resolve_device


def run_evaluation(cfg: DictConfig, checkpoint_path: str) -> dict[str, Any]:
    """Evaluate a policy checkpoint and return aggregate metrics."""

    device = resolve_device(str(cfg.device))

    env = gym.make(str(cfg.env.task))
    test_envs: BaseVectorEnv | None = None
    try:
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))

        test_envs = make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
        if bool(cfg.data.obs_norm):
            replay_buffer = build_replay_buffer_from_cfg(
                cfg,
                env,
                cfg.get("buffer_size"),
            )
            _, test_envs = apply_obs_norm(replay_buffer, test_envs)

        seed_vector_env(test_envs, int(cfg.seed))

        algorithm = build_algorithm(cfg, env, device)
        load_policy_state(algorithm, checkpoint_path, device)

        collector = Collector[CollectStats](algorithm, test_envs)
        _, metrics = collect_eval_metrics(
            collector,
            num_episodes=int(cfg.env.num_test_envs),
            render=float(cfg.env.render),
        )
        return {
            "checkpoint": checkpoint_path,
            "episodes": int(cfg.env.num_test_envs),
            **metrics,
        }
    finally:
        if test_envs is not None:
            test_envs.close()
        env.close()
