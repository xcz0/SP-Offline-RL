"""Composite simulator evaluation pipeline."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.evaluation.sp_sim.policy_evaluator import evaluate_policy_with_simulator
from src.evaluation.sp_sim.replay_evaluator import evaluate_replay_with_simulator
from src.evaluation.sp_sim.types import CompositeEvalResult


def _to_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _to_int_list(values: Any) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        out = [int(v) for v in values]
        return out if out else None
    return None


def evaluate_composite_with_simulator(
    *,
    policy: Any,
    sim_eval_cfg: DictConfig | dict[str, Any],
    action_max: float,
) -> CompositeEvalResult:
    """Run policy/replay simulator evaluators and combine their summaries."""

    cfg = _to_dict(sim_eval_cfg)
    predictor_cfg = dict(cfg.get("predictor", {}))
    replay_cfg = dict(cfg.get("replay_eval", {}))
    policy_cfg = dict(cfg.get("policy_eval", {}))
    weights = dict(cfg.get("score_weights", {}))

    common_kwargs = {
        "data_dir": str(cfg.get("data_dir", "data")),
        "predictor_model_path": str(predictor_cfg.get("model_path", "")),
        "predictor_device": str(predictor_cfg.get("device", "cpu")),
        "predictor_dtype": str(predictor_cfg.get("dtype", "float32")),
        "user_ids": _to_int_list(cfg.get("user_ids")),
        "cards_per_user": int(cfg.get("cards_per_user", 20)),
        "min_target_occurrences": int(cfg.get("min_target_occurrences", 5)),
        "warmup_mode": str(cfg.get("warmup_mode", "fifth")),
        "seed": int(cfg.get("seed", 0)),
    }

    policy_result = None
    replay_result = None
    if bool(policy_cfg.get("enabled", True)):
        policy_result = evaluate_policy_with_simulator(
            policy=policy,
            action_max=float(action_max),
            score_weights=weights,
            obs_mean=cfg.get("obs_norm_mean"),
            obs_var=cfg.get("obs_norm_var"),
            **common_kwargs,
        )
    if bool(replay_cfg.get("enabled", True)):
        replay_result = evaluate_replay_with_simulator(**common_kwargs)

    summary: dict[str, Any] = {
        "policy_eval_enabled": bool(policy_cfg.get("enabled", True)),
        "replay_eval_enabled": bool(replay_cfg.get("enabled", True)),
    }
    if policy_result is not None:
        summary.update(
            {
                "score_mean": float(policy_result.summary.get("score_mean", float("nan"))),
                "score_std": float(policy_result.summary.get("score_std", float("nan"))),
                "retention_area_mean": float(
                    policy_result.summary.get("retention_area_mean", float("nan"))
                ),
                "final_retention_mean": float(
                    policy_result.summary.get("final_retention_mean", float("nan"))
                ),
                "review_count_mean": float(
                    policy_result.summary.get("review_count_mean", float("nan"))
                ),
                "num_targets": float(policy_result.summary.get("num_targets", 0.0)),
            }
        )
    if replay_result is not None:
        summary.update(
            {
                "replay_retention_area_mean": float(
                    replay_result.summary.get("retention_area_mean", float("nan"))
                ),
                "replay_final_retention_mean": float(
                    replay_result.summary.get("final_retention_mean", float("nan"))
                ),
                "replay_num_targets": float(replay_result.summary.get("num_targets", 0.0)),
            }
        )
    return CompositeEvalResult(
        policy_result=policy_result,
        replay_result=replay_result,
        summary=summary,
    )
