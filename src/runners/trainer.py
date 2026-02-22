"""Offline training entrypoint for continuous-control experiments."""

from __future__ import annotations

import datetime
import os
from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv, SubprocVectorEnv, VectorEnvNormObs
from tianshou.trainer import InfoStats, OfflineTrainerParams
from tianshou.utils.space_info import SpaceInfo

from src.algos.registry import get_algo_factory
from src.core.seed import seed_vector_env, set_global_seed
from src.data.dataset_adapter import build_dataset_adapter
from src.data.replay_buffer_builder import build_replay_buffer
from src.data.schema import validate_against_env, validate_and_standardize_dataset
from src.data.transforms import normalize_obs_in_replay_buffer
from src.logging.factory import build_logger
from src.logging.metrics import collect_stats_to_metrics
from src.models.registry import get_model_factory
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


def _make_test_envs(task: str, num_test_envs: int) -> BaseVectorEnv:
    return SubprocVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])


def run_offline_training(cfg: DictConfig) -> dict[str, Any]:
    """Run end-to-end offline training and return summary metrics."""

    resolved_cfg = resolve_config(cfg)
    device = resolve_device(str(cfg.device))
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    run_name = os.path.join(
        str(cfg.env.task), str(cfg.algo.name), str(cfg.seed), timestamp
    )
    log_path = os.path.join(str(cfg.paths.logdir), run_name)
    ensure_dir(log_path)

    save_text(os.path.join(log_path, "resolved_config.yaml"), as_yaml(cfg))

    env = gym.make(str(cfg.env.task))
    test_envs: BaseVectorEnv | None = None
    logger_artifacts = None

    try:
        space_info = SpaceInfo.from_env(env)
        set_global_seed(int(cfg.seed))

        adapter = build_dataset_adapter(cfg.data)
        data = validate_and_standardize_dataset(adapter.load())
        validate_against_env(data, env)

        replay_buffer = build_replay_buffer(data, cfg.get("buffer_size"))

        test_envs = _make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
        if bool(cfg.data.obs_norm):
            replay_buffer, obs_rms = normalize_obs_in_replay_buffer(replay_buffer)
            test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
            test_envs.set_obs_rms(obs_rms)

        seed_vector_env(test_envs, int(cfg.seed))

        model_factory = get_model_factory(str(cfg.model.name))
        model_bundle = model_factory.build(cfg.model, space_info, device)

        algo_factory = get_algo_factory(str(cfg.algo.name))
        algorithm = algo_factory.build(cfg.algo, env, model_bundle, device)

        if cfg.resume_path:
            algorithm.load_state_dict(
                torch.load(str(cfg.resume_path), map_location=device)
            )

        test_collector = Collector[CollectStats](algorithm, test_envs)

        logger_artifacts = build_logger(
            logger_cfg=cfg.logger,
            log_path=log_path,
            run_name=run_name,
            resume_id=cfg.resume_id,
            config_dict=resolved_cfg,
        )

        save_best_fn = build_save_best_fn(log_path)

        if bool(cfg.watch):
            checkpoint = cfg.resume_path or os.path.join(log_path, "policy.pth")
            algorithm.load_state_dict(torch.load(str(checkpoint), map_location=device))
            test_collector.reset()
            collector_stats = test_collector.collect(
                n_episode=int(cfg.env.num_test_envs),
                render=float(cfg.env.render),
            )
            metrics = collect_stats_to_metrics(collector_stats)
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
        test_collector.reset()
        collector_stats = test_collector.collect(
            n_episode=int(cfg.env.num_test_envs),
            render=float(cfg.env.render),
        )

        train_dict = _info_stats_to_dict(training_result)
        eval_metrics = collect_stats_to_metrics(collector_stats)
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
