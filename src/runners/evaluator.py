"""Standalone policy evaluation from checkpoint."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv

from src.core.exceptions import ConfigurationError
from src.core.seed import seed_vector_env, set_global_seed
from src.data.transforms import normalize_obs_array
from src.evaluation.deps import require_sprwkv
from src.evaluation.pipeline import evaluate_composite_with_simulator
from src.evaluation.replay_evaluator import evaluate_replay_with_simulator
from src.runners.common import (
    apply_obs_norm,
    apply_obs_norm_to_obs_act_data,
    build_algorithm,
    build_algorithm_from_space_info,
    build_obs_act_dataset_from_cfg,
    build_replay_buffer_from_cfg,
    build_space_info_from_obs_act,
    collect_eval_metrics,
    infer_action_bounds_from_dataset,
    load_policy_state,
    make_test_envs,
)
from src.utils.hydra import resolve_device


def _resolve_eval_mode(cfg: DictConfig, algo_name: str) -> str:
    mode = str(cfg.get("eval_mode", "auto")).lower()
    if mode != "auto":
        return mode
    sim_eval_cfg = cfg.get("sim_eval")
    if algo_name == "bc_il" and sim_eval_cfg is not None and bool(
        sim_eval_cfg.get("enabled", False)
    ):
        return "sim"
    return "gym"


def run_evaluation(cfg: DictConfig, checkpoint_path: str) -> dict[str, Any]:
    """Evaluate a policy checkpoint and return aggregate metrics."""

    device = resolve_device(str(cfg.device))
    algo_name = str(cfg.algo.name)
    eval_mode = _resolve_eval_mode(cfg, algo_name)

    if eval_mode == "sim":
        if algo_name != "bc_il":
            raise ConfigurationError("sim eval mode currently supports only bc_il.")
        if not checkpoint_path:
            raise ConfigurationError("checkpoint_path is required for sim evaluation.")
        require_sprwkv()
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))
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
        load_policy_state(algorithm, checkpoint_path, device)

        sim_eval_cfg = cfg.get("sim_eval")
        if sim_eval_cfg is None:
            raise ConfigurationError("sim_eval config is required for sim mode.")
        sim_cfg = OmegaConf.to_container(sim_eval_cfg, resolve=True)
        if not isinstance(sim_cfg, dict):
            raise ConfigurationError("sim_eval config must resolve to a dictionary.")
        if obs_norm_mean is not None:
            sim_cfg["obs_norm_mean"] = obs_norm_mean
            sim_cfg["obs_norm_var"] = obs_norm_var

        composite = evaluate_composite_with_simulator(
            policy=algorithm.policy,
            sim_eval_cfg=sim_cfg,
            action_max=float(action_high),
        )
        return {
            "mode": "sim",
            "checkpoint": checkpoint_path,
            **composite.summary,
        }

    if eval_mode == "replay":
        require_sprwkv()
        sim_eval_cfg = cfg.get("sim_eval")
        if sim_eval_cfg is None:
            raise ConfigurationError("sim_eval config is required for replay mode.")
        sim_cfg = OmegaConf.to_container(sim_eval_cfg, resolve=True)
        if not isinstance(sim_cfg, dict):
            raise ConfigurationError("sim_eval config must resolve to a dictionary.")

        predictor_cfg = dict(sim_cfg.get("predictor", {}))
        replay = evaluate_replay_with_simulator(
            data_dir=str(sim_cfg.get("data_dir", "data")),
            predictor_model_path=str(predictor_cfg.get("model_path", "")),
            predictor_device=str(predictor_cfg.get("device", device)),
            predictor_dtype=str(predictor_cfg.get("dtype", "float32")),
            user_ids=[int(v) for v in sim_cfg.get("user_ids", [])] or None,
            cards_per_user=int(sim_cfg.get("cards_per_user", 20)),
            min_target_occurrences=int(sim_cfg.get("min_target_occurrences", 5)),
            warmup_mode=str(sim_cfg.get("warmup_mode", "fifth")),
            seed=int(sim_cfg.get("seed", 0)),
        )
        return {"mode": "replay", **replay.summary}

    if eval_mode != "gym":
        raise ConfigurationError(
            f"Unsupported eval_mode '{eval_mode}'. Expected one of: auto, gym, sim, replay."
        )

    env = gym.make(str(cfg.env.task))
    test_envs: BaseVectorEnv | None = None
    try:
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))

        test_envs = make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
        if bool(cfg.data.obs_norm):
            if algo_name == "bc_il":
                obs_act_data = build_obs_act_dataset_from_cfg(cfg, env)
                _, test_envs = apply_obs_norm_to_obs_act_data(obs_act_data, test_envs)
            else:
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
