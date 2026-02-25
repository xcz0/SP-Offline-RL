"""Unified train/eval pipelines with strongly typed results."""

from __future__ import annotations

import datetime
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from tianshou.data import Collector, CollectStats
from tianshou.trainer import InfoStats, OfflineTrainerParams

from src.core.exceptions import ConfigurationError
from src.core.seed import seed_vector_env, set_global_seed
from src.evaluation.collector import SimulationEvalCollector
from src.evaluation.deps import require_sprwkv
from src.evaluation.pipeline import evaluate_composite_with_simulator
from src.evaluation.replay_evaluator import evaluate_replay_with_simulator
from src.logging.factory import build_logger
from src.runners.callbacks import build_save_best_fn
from src.runners.common import (
    build_sim_eval_cfg,
    collect_eval_metrics,
    load_policy_state,
    prepare_bc_sim_components,
    prepare_gym_components,
)
from src.runners.mode import resolve_eval_mode, resolve_train_mode
from src.runners.types import (
    EvalMetrics,
    EvaluationResult,
    SimEvalArtifacts,
    TrainingMetrics,
    TrainingResult,
    to_builtin,
)
from src.utils.hydra import as_yaml, resolve_config, resolve_device
from src.utils.io import ensure_dir, save_json, save_text


def _info_stats_to_dict(stats: InfoStats) -> dict[str, Any]:
    return to_builtin(asdict(stats))


def _get_task_name(cfg: DictConfig) -> str:
    env_cfg = cfg.get("env")
    if env_cfg and env_cfg.get("task"):
        return str(env_cfg.task)
    return "offline"


def _metrics_from_collect_stats(stats: CollectStats) -> EvalMetrics:
    if stats.returns_stat is None:
        return EvalMetrics(
            test_reward_mean=float("nan"),
            test_reward_std=float("nan"),
        )
    return EvalMetrics(
        test_reward_mean=float(stats.returns_stat.mean),
        test_reward_std=float(stats.returns_stat.std),
    )


def _metrics_from_dict(metrics: dict[str, float]) -> EvalMetrics:
    payload = dict(metrics)
    reward_mean = float(payload.pop("test_reward_mean", float("nan")))
    reward_std = float(payload.pop("test_reward_std", float("nan")))
    return EvalMetrics(
        test_reward_mean=reward_mean,
        test_reward_std=reward_std,
        extra=to_builtin(payload),
    )


def _persist_sim_eval_artifacts(
    log_path: str,
    collector: SimulationEvalCollector,
) -> SimEvalArtifacts | None:
    latest = collector.latest_result
    if latest is None:
        return None

    sim_dir = Path(log_path) / "sim_eval"
    sim_dir.mkdir(parents=True, exist_ok=True)

    artifacts = SimEvalArtifacts(summary=to_builtin(latest.summary))

    policy_result = latest.policy_result
    if policy_result is not None:
        policy_metrics_path = sim_dir / "policy_metrics.parquet"
        policy_summary_path = sim_dir / "policy_summary.json"
        policy_result.per_target_metrics.to_parquet(policy_metrics_path, index=False)
        save_json(policy_summary_path, to_builtin(policy_result.summary))
        artifacts.policy_metrics_path = str(policy_metrics_path)
        artifacts.policy_summary_path = str(policy_summary_path)

    replay_result = latest.replay_result
    if replay_result is not None:
        replay_final_path = sim_dir / "replay_final_metrics.parquet"
        replay_traj_path = sim_dir / "replay_trajectory.parquet"
        replay_summary_path = sim_dir / "replay_summary.json"
        replay_result.final_metrics.to_parquet(replay_final_path, index=False)
        replay_result.trajectory.to_parquet(replay_traj_path, index=False)
        save_json(replay_summary_path, to_builtin(replay_result.summary))
        artifacts.replay_final_metrics_path = str(replay_final_path)
        artifacts.replay_trajectory_path = str(replay_traj_path)
        artifacts.replay_summary_path = str(replay_summary_path)

    composite_summary_path = sim_dir / "composite_summary.json"
    save_json(composite_summary_path, artifacts.summary)
    artifacts.composite_summary_path = str(composite_summary_path)
    return artifacts


def train(cfg: DictConfig) -> TrainingResult:
    """Run end-to-end offline training and return typed summary metrics."""

    device = resolve_device(str(cfg.device))
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_name = os.path.join(_get_task_name(cfg), str(cfg.algo.name), str(cfg.seed), timestamp)
    log_path = os.path.join(str(cfg.paths.logdir), run_name)
    ensure_dir(log_path)

    resolved_config_path = os.path.join(log_path, "resolved_config.yaml")
    save_text(resolved_config_path, as_yaml(cfg))

    gym_components = None
    logger_artifacts = None
    sim_collector: SimulationEvalCollector | None = None

    try:
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))
        algo_name = str(cfg.algo.name)
        run_mode, use_bc_sim_eval = resolve_train_mode(cfg, algo_name)

        trainer_test_fn = None
        trainer_compute_score_fn = None

        if algo_name == "bc_il" and use_bc_sim_eval:
            require_sprwkv()
            sim_components = prepare_bc_sim_components(
                cfg,
                device,
                seed=int(cfg.seed),
                include_train_buffer=True,
            )
            if sim_components.train_buffer is None:
                raise RuntimeError("train_buffer must be initialized for bc_il training.")
            train_buffer = sim_components.train_buffer
            algorithm = sim_components.algorithm
            sim_cfg = build_sim_eval_cfg(
                cfg,
                obs_norm_mean=sim_components.obs_norm_mean,
                obs_norm_var=sim_components.obs_norm_var,
            )
            sim_collector = SimulationEvalCollector(
                policy=algorithm.policy,
                sim_eval_cfg=sim_cfg,
                action_max=float(sim_components.action_high),
                eval_every_n_epoch=int(cfg.sim_eval.eval_every_n_epoch),
            )
            test_collector: Any = sim_collector

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
            gym_components = prepare_gym_components(
                cfg,
                device,
                seed=int(cfg.seed),
                include_train_buffer=True,
            )
            if gym_components.train_buffer is None:
                raise RuntimeError("train_buffer must be initialized for training.")
            train_buffer = gym_components.train_buffer
            algorithm = gym_components.algorithm
            test_collector = Collector[CollectStats](algorithm, gym_components.test_envs)

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
        checkpoint_path = str(cfg.resume_path or (Path(log_path) / "policy.pth"))

        if run_mode == "watch":
            load_policy_state(algorithm, checkpoint_path, device)
            if sim_collector is not None:
                sim_collector.set_epoch(int(cfg.train.epoch))
                stats = sim_collector.collect(n_episode=1)
                eval_metrics = _metrics_from_collect_stats(stats)
                sim_artifacts = _persist_sim_eval_artifacts(log_path, sim_collector)
            else:
                _, metrics = collect_eval_metrics(
                    test_collector,
                    num_episodes=int(cfg.env.num_test_envs),
                    render=float(cfg.env.render),
                )
                eval_metrics = _metrics_from_dict(metrics)
                sim_artifacts = None

            result = TrainingResult(
                mode="watch",
                log_path=log_path,
                run_name=run_name,
                checkpoint_path=checkpoint_path,
                training=None,
                evaluation=eval_metrics,
                sim_eval=sim_artifacts,
            )
            save_json(Path(log_path) / str(cfg.paths.save_metrics_filename), result.to_dict())
            return result

        training_result = algorithm.run_training(
            OfflineTrainerParams(
                buffer=train_buffer,
                test_collector=test_collector,
                max_epochs=int(cfg.train.epoch),
                epoch_num_steps=int(cfg.train.epoch_num_steps),
                test_step_num_episodes=(1 if use_bc_sim_eval else int(cfg.env.num_test_envs)),
                batch_size=int(cfg.train.batch_size),
                save_best_fn=save_best_fn,
                logger=logger_artifacts.logger,
                test_fn=trainer_test_fn,
                compute_score_fn=trainer_compute_score_fn,
            )
        )

        if sim_collector is not None:
            sim_collector.set_epoch(int(cfg.train.epoch))
            stats = sim_collector.collect(n_episode=1)
            eval_metrics = _metrics_from_collect_stats(stats)
            sim_artifacts = _persist_sim_eval_artifacts(log_path, sim_collector)
        else:
            assert gym_components is not None
            seed_vector_env(gym_components.test_envs, int(cfg.seed))
            _, metrics = collect_eval_metrics(
                test_collector,
                num_episodes=int(cfg.env.num_test_envs),
                render=float(cfg.env.render),
            )
            eval_metrics = _metrics_from_dict(metrics)
            sim_artifacts = None

        training_metrics = TrainingMetrics(
            best_reward=float(training_result.best_reward),
            best_reward_std=float(training_result.best_reward_std),
            update_step=int(training_result.update_step),
            train_step=int(training_result.train_step),
            test_step=int(training_result.test_step),
            raw=_info_stats_to_dict(training_result),
        )
        result = TrainingResult(
            mode="train",
            log_path=log_path,
            run_name=run_name,
            checkpoint_path=str(Path(log_path) / "policy.pth"),
            training=training_metrics,
            evaluation=eval_metrics,
            sim_eval=sim_artifacts,
        )
        save_json(Path(log_path) / str(cfg.paths.save_metrics_filename), result.to_dict())
        return result
    finally:
        if logger_artifacts is not None:
            logger_artifacts.writer.close()
        if gym_components is not None:
            gym_components.test_envs.close()
            gym_components.env.close()


def evaluate(cfg: DictConfig, checkpoint_path: str | None = None) -> EvaluationResult:
    """Evaluate a policy checkpoint and return aggregate typed metrics."""

    device = resolve_device(str(cfg.device))
    algo_name = str(cfg.algo.name)
    eval_mode = resolve_eval_mode(cfg, algo_name)
    effective_checkpoint = str(checkpoint_path or cfg.get("checkpoint_path") or "")

    if eval_mode != "replay" and not effective_checkpoint:
        raise ConfigurationError(
            "checkpoint_path is required for eval unless eval_mode=replay. "
            "Example: python scripts/eval.py checkpoint_path=/path/policy.pth"
        )

    if eval_mode == "sim":
        if algo_name != "bc_il":
            raise ConfigurationError("sim eval mode currently supports only bc_il.")
        require_sprwkv()
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))

        sim_components = prepare_bc_sim_components(
            cfg,
            device,
            seed=int(cfg.seed),
            include_train_buffer=False,
        )
        load_policy_state(sim_components.algorithm, effective_checkpoint, device)
        sim_cfg = build_sim_eval_cfg(
            cfg,
            obs_norm_mean=sim_components.obs_norm_mean,
            obs_norm_var=sim_components.obs_norm_var,
        )
        composite = evaluate_composite_with_simulator(
            policy=sim_components.algorithm.policy,
            sim_eval_cfg=sim_cfg,
            action_max=float(sim_components.action_high),
        )
        summary = to_builtin(composite.summary)
        return EvaluationResult(
            mode="sim",
            checkpoint_path=effective_checkpoint,
            evaluation=EvalMetrics(
                test_reward_mean=float(summary.get("score_mean", float("nan"))),
                test_reward_std=float(summary.get("score_std", float("nan"))),
                extra=summary,
            ),
            sim_eval=SimEvalArtifacts(summary=summary),
        )

    if eval_mode == "replay":
        require_sprwkv()
        sim_cfg = build_sim_eval_cfg(cfg)
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
        summary = to_builtin(replay.summary)
        return EvaluationResult(
            mode="replay",
            checkpoint_path=None,
            evaluation=EvalMetrics(
                test_reward_mean=float(summary.get("retention_area_mean", float("nan"))),
                test_reward_std=float(summary.get("retention_area_std", float("nan"))),
                extra=summary,
            ),
            sim_eval=SimEvalArtifacts(summary=summary),
        )

    gym_components = None
    try:
        set_global_seed(int(cfg.seed), seed_cuda=device.startswith("cuda"))
        gym_components = prepare_gym_components(
            cfg,
            device,
            seed=int(cfg.seed),
            include_train_buffer=False,
        )
        load_policy_state(gym_components.algorithm, effective_checkpoint, device)
        collector = Collector[CollectStats](gym_components.algorithm, gym_components.test_envs)
        _, metrics = collect_eval_metrics(
            collector,
            num_episodes=int(cfg.env.num_test_envs),
            render=float(cfg.env.render),
        )
        return EvaluationResult(
            mode="gym",
            checkpoint_path=effective_checkpoint,
            evaluation=_metrics_from_dict(metrics),
            sim_eval=None,
        )
    finally:
        if gym_components is not None:
            gym_components.test_envs.close()
            gym_components.env.close()
