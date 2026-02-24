"""Offline training entrypoint for continuous-control experiments."""

from __future__ import annotations

import datetime
import os
from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv
from tianshou.trainer import InfoStats, OfflineTrainerParams

from src.core.seed import seed_vector_env, set_global_seed
from src.logging.factory import build_logger
from src.runners.common import (
    apply_obs_norm,
    build_algorithm,
    build_replay_buffer_from_cfg,
    collect_eval_metrics,
    load_policy_state,
    make_test_envs,
)
from src.runners.callbacks import build_save_best_fn
from src.utils.hydra import as_yaml, resolve_config, resolve_device
from src.utils.io import ensure_dir, save_json, save_text


def _info_stats_to_dict(stats: InfoStats) -> dict[str, Any]:
    return _to_builtin(asdict(stats))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_offline_training(cfg: DictConfig) -> dict[str, Any]:
    """Run end-to-end offline training and return summary metrics."""

    device = resolve_device(str(cfg.device))
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    run_name = os.path.join(
        str(cfg.env.task), str(cfg.algo.name), str(cfg.seed), timestamp
    )
    log_path = os.path.join(str(cfg.paths.logdir), run_name)
    ensure_dir(log_path)

    resolved_config_path = os.path.join(log_path, "resolved_config.yaml")
    save_text(resolved_config_path, as_yaml(cfg))

    env = gym.make(str(cfg.env.task))
    test_envs: BaseVectorEnv | None = None
    logger_artifacts = None

    try:
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))
        replay_buffer = build_replay_buffer_from_cfg(
            cfg,
            env,
            cfg.get("buffer_size"),
        )

        test_envs = make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
        if bool(cfg.data.obs_norm):
            replay_buffer, test_envs = apply_obs_norm(replay_buffer, test_envs)

        seed_vector_env(test_envs, int(cfg.seed))

        algorithm = build_algorithm(cfg, env, device)

        if cfg.resume_path:
            load_policy_state(algorithm, str(cfg.resume_path), device)

        test_collector = Collector[CollectStats](algorithm, test_envs)
        resolved_cfg = resolve_config(cfg) if str(cfg.logger.type) == "wandb" else None

        logger_artifacts = build_logger(
            logger_cfg=cfg.logger,
            log_path=log_path,
            run_name=run_name,
            resume_id=cfg.resume_id,
            config_dict=resolved_cfg,
            resolved_config_path=resolved_config_path,
        )

        save_best_fn = build_save_best_fn(log_path)

        if bool(cfg.watch):
            checkpoint = cfg.resume_path or os.path.join(log_path, "policy.pth")
            load_policy_state(
                algorithm,
                str(checkpoint),
                device,
            )
            _, metrics = collect_eval_metrics(
                test_collector,
                num_episodes=int(cfg.env.num_test_envs),
                render=float(cfg.env.render),
            )
            output = {
                "mode": "watch",
                "checkpoint": str(checkpoint),
                "log_path": log_path,
                **metrics,
            }
            save_json(
                os.path.join(log_path, str(cfg.paths.save_metrics_filename)), output
            )
            return output

        training_result = algorithm.run_training(
            OfflineTrainerParams(
                buffer=replay_buffer,
                test_collector=test_collector,
                max_epochs=int(cfg.train.epoch),
                epoch_num_steps=int(cfg.train.epoch_num_steps),
                test_step_num_episodes=int(cfg.env.num_test_envs),
                batch_size=int(cfg.train.batch_size),
                save_best_fn=save_best_fn,
                logger=logger_artifacts.logger,
            )
        )

        seed_vector_env(test_envs, int(cfg.seed))
        _, eval_metrics = collect_eval_metrics(
            test_collector,
            num_episodes=int(cfg.env.num_test_envs),
            render=float(cfg.env.render),
        )

        train_dict = _info_stats_to_dict(training_result)
        final_metrics = {
            "log_path": log_path,
            "run_name": run_name,
            "test_reward_mean": eval_metrics["test_reward_mean"],
            "test_reward_std": eval_metrics["test_reward_std"],
            "best_reward": float(training_result.best_reward),
            "best_reward_std": float(training_result.best_reward_std),
            "update_step": int(training_result.update_step),
            "train_step": int(training_result.train_step),
            "test_step": int(training_result.test_step),
        }

        save_json(
            os.path.join(log_path, str(cfg.paths.save_metrics_filename)), final_metrics
        )
        return {
            "train": train_dict,
            "final_metrics": final_metrics,
        }
    finally:
        if logger_artifacts is not None:
            logger_artifacts.writer.close()
        if test_envs is not None:
            test_envs.close()
        env.close()
