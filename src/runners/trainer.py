"""Offline training entrypoint for continuous-control experiments."""

from __future__ import annotations

import datetime
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv
from tianshou.trainer import InfoStats, OfflineTrainerParams

from src.core.seed import seed_vector_env, set_global_seed
from src.data.transforms import normalize_obs_array
from src.evaluation.sp_sim.collector import SimulationEvalCollector
from src.evaluation.sp_sim.deps import require_sprwkv
from src.logging.factory import build_logger
from src.runners.common import (
    apply_obs_norm,
    apply_obs_norm_to_obs_act_data,
    build_algorithm,
    build_algorithm_from_space_info,
    build_obs_act_buffer_from_dataset,
    build_obs_act_dataset_from_cfg,
    build_replay_buffer_from_cfg,
    build_space_info_from_obs_act,
    collect_eval_metrics,
    infer_action_bounds_from_dataset,
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


def _get_task_name(cfg: DictConfig) -> str:
    env_cfg = cfg.get("env")
    if env_cfg and env_cfg.get("task"):
        return str(env_cfg.task)
    return "offline"


def _is_bc_sim_eval_enabled(cfg: DictConfig, algo_name: str) -> bool:
    if algo_name != "bc_il":
        return False
    sim_eval_cfg = cfg.get("sim_eval")
    if sim_eval_cfg is None:
        return False
    return bool(sim_eval_cfg.get("enabled", False))


def _persist_sim_eval_artifacts(log_path: str, collector: SimulationEvalCollector) -> dict[str, Any]:
    latest = collector.latest_result
    if latest is None:
        return {}

    sim_dir = Path(log_path) / "sim_eval"
    sim_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}

    policy_result = latest.policy_result
    if policy_result is not None:
        policy_metrics_path = sim_dir / "policy_metrics.parquet"
        policy_summary_path = sim_dir / "policy_summary.json"
        policy_result.per_target_metrics.to_parquet(policy_metrics_path, index=False)
        save_json(str(policy_summary_path), _to_builtin(policy_result.summary))
        output["policy_metrics_path"] = str(policy_metrics_path)
        output["policy_summary_path"] = str(policy_summary_path)

    replay_result = latest.replay_result
    if replay_result is not None:
        replay_final_path = sim_dir / "replay_final_metrics.parquet"
        replay_traj_path = sim_dir / "replay_trajectory.parquet"
        replay_summary_path = sim_dir / "replay_summary.json"
        replay_result.final_metrics.to_parquet(replay_final_path, index=False)
        replay_result.trajectory.to_parquet(replay_traj_path, index=False)
        save_json(str(replay_summary_path), _to_builtin(replay_result.summary))
        output["replay_final_metrics_path"] = str(replay_final_path)
        output["replay_trajectory_path"] = str(replay_traj_path)
        output["replay_summary_path"] = str(replay_summary_path)

    composite_summary_path = sim_dir / "composite_summary.json"
    save_json(str(composite_summary_path), _to_builtin(latest.summary))
    output["composite_summary_path"] = str(composite_summary_path)
    return output


def run_offline_training(cfg: DictConfig) -> dict[str, Any]:
    """Run end-to-end offline training and return summary metrics."""

    device = resolve_device(str(cfg.device))
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    run_name = os.path.join(
        _get_task_name(cfg), str(cfg.algo.name), str(cfg.seed), timestamp
    )
    log_path = os.path.join(str(cfg.paths.logdir), run_name)
    ensure_dir(log_path)

    resolved_config_path = os.path.join(log_path, "resolved_config.yaml")
    save_text(resolved_config_path, as_yaml(cfg))

    env = None
    test_envs: BaseVectorEnv | None = None
    test_collector: Any | None = None
    sim_collector: SimulationEvalCollector | None = None
    logger_artifacts = None

    try:
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))
        algo_name = str(cfg.algo.name)
        use_bc_sim_eval = _is_bc_sim_eval_enabled(cfg, algo_name)
        obs_norm_mean: list[float] | None = None
        obs_norm_var: list[float] | None = None
        trainer_test_fn = None
        trainer_compute_score_fn = None

        if algo_name == "bc_il" and use_bc_sim_eval:
            require_sprwkv()
            obs_act_data = build_obs_act_dataset_from_cfg(cfg, env=None)
            if bool(cfg.data.obs_norm):
                normalized_obs, obs_rms = normalize_obs_array(obs_act_data["obs"])
                obs_act_data = dict(obs_act_data)
                obs_act_data["obs"] = normalized_obs
                obs_norm_mean = np.asarray(obs_rms.mean, dtype=np.float32).tolist()
                obs_norm_var = np.asarray(obs_rms.var, dtype=np.float32).tolist()

            train_buffer = build_obs_act_buffer_from_dataset(obs_act_data, int(cfg.seed))
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

            sim_cfg = OmegaConf.to_container(cfg.sim_eval, resolve=True)  # type: ignore[arg-type]
            assert isinstance(sim_cfg, dict)
            if obs_norm_mean is not None:
                sim_cfg["obs_norm_mean"] = obs_norm_mean
                sim_cfg["obs_norm_var"] = obs_norm_var
            sim_collector = SimulationEvalCollector(
                policy=algorithm.policy,
                sim_eval_cfg=sim_cfg,
                action_max=float(action_high),
                eval_every_n_epoch=int(cfg.sim_eval.eval_every_n_epoch),
            )
            test_collector = sim_collector

            def _sim_test_fn(epoch: int, _env_step: int | None) -> None:
                assert sim_collector is not None
                sim_collector.set_epoch(int(epoch))

            trainer_test_fn = _sim_test_fn

            def _score_fn(stats: CollectStats) -> float:
                if stats.returns_stat is None:
                    return float("nan")
                return float(stats.returns_stat.mean)

            trainer_compute_score_fn = _score_fn
        else:
            env = gym.make(str(cfg.env.task))

            if algo_name == "bc_il":
                obs_act_data = build_obs_act_dataset_from_cfg(cfg, env)
                train_buffer = build_obs_act_buffer_from_dataset(
                    obs_act_data, int(cfg.seed)
                )
            else:
                train_buffer = build_replay_buffer_from_cfg(
                    cfg,
                    env,
                    cfg.get("buffer_size"),
                )

            test_envs = make_test_envs(str(cfg.env.task), int(cfg.env.num_test_envs))
            if bool(cfg.data.obs_norm):
                if algo_name == "bc_il":
                    obs_act_data, test_envs = apply_obs_norm_to_obs_act_data(
                        obs_act_data, test_envs
                    )
                    train_buffer = build_obs_act_buffer_from_dataset(
                        obs_act_data, int(cfg.seed)
                    )
                else:
                    train_buffer, test_envs = apply_obs_norm(train_buffer, test_envs)

            seed_vector_env(test_envs, int(cfg.seed))
            algorithm = build_algorithm(cfg, env, device)
            test_collector = Collector[CollectStats](algorithm, test_envs)

        if cfg.resume_path:
            load_policy_state(algorithm, str(cfg.resume_path), device)
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
            if use_bc_sim_eval and sim_collector is not None:
                sim_collector.set_epoch(int(cfg.train.epoch))
                stats = sim_collector.collect(n_episode=1)
                latest = sim_collector.latest_result
                output = {
                    "mode": "watch",
                    "checkpoint": str(checkpoint),
                    "log_path": log_path,
                    "score_mean": float(stats.returns_stat.mean)
                    if stats.returns_stat is not None
                    else float("nan"),
                    "sim_eval_summary": _to_builtin(latest.summary) if latest else {},
                }
                output.update(_persist_sim_eval_artifacts(log_path, sim_collector))
            else:
                assert test_collector is not None
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

        if test_collector is None:
            raise RuntimeError("test_collector must be initialized before training.")

        training_result = algorithm.run_training(
            OfflineTrainerParams(
                buffer=train_buffer,
                test_collector=test_collector,
                max_epochs=int(cfg.train.epoch),
                epoch_num_steps=int(cfg.train.epoch_num_steps),
                test_step_num_episodes=(
                    1 if use_bc_sim_eval else int(cfg.env.num_test_envs)
                ),
                batch_size=int(cfg.train.batch_size),
                save_best_fn=save_best_fn,
                logger=logger_artifacts.logger,
                test_fn=trainer_test_fn,
                compute_score_fn=trainer_compute_score_fn,
            )
        )

        if use_bc_sim_eval and sim_collector is not None:
            sim_collector.set_epoch(int(cfg.train.epoch))
            final_stats = sim_collector.collect(n_episode=1)
            latest = sim_collector.latest_result
            eval_metrics = {
                "test_reward_mean": float(final_stats.returns_stat.mean)
                if final_stats.returns_stat is not None
                else float("nan"),
                "test_reward_std": float(final_stats.returns_stat.std)
                if final_stats.returns_stat is not None
                else float("nan"),
            }
            sim_artifacts = _persist_sim_eval_artifacts(log_path, sim_collector)
        else:
            assert test_envs is not None
            seed_vector_env(test_envs, int(cfg.seed))
            _, eval_metrics = collect_eval_metrics(
                test_collector,
                num_episodes=int(cfg.env.num_test_envs),
                render=float(cfg.env.render),
            )
            latest = None
            sim_artifacts = {}

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
        if use_bc_sim_eval and latest is not None:
            final_metrics.update(
                {
                    "best_sim_score": float(training_result.best_reward),
                    "last_sim_score": float(latest.summary.get("score_mean", float("nan"))),
                    "sim_eval_targets": float(latest.summary.get("num_targets", 0.0)),
                }
            )
            final_metrics.update(sim_artifacts)

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
        if env is not None:
            env.close()
