"""Ground-truth replay evaluation on RWKV simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import ConfigurationError
from src.evaluation.sp_sim.dataset import load_user_data_map, sample_eval_targets
from src.evaluation.sp_sim.deps import require_sprwkv
from src.evaluation.sp_sim.types import ReplayEvalResult


def _resolve_torch_dtype(dtype_name: str):
    import torch

    name = dtype_name.strip().lower()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ConfigurationError(
            f"Unsupported predictor dtype: {dtype_name}. "
            "Expected one of: float32, float16, bfloat16."
        )
    return mapping[name]


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # noqa: BLE001
            return float("nan")
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def evaluate_replay_with_simulator(
    *,
    data_dir: str,
    predictor_model_path: str,
    predictor_device: str,
    predictor_dtype: str,
    user_ids: list[int] | None,
    cards_per_user: int,
    min_target_occurrences: int,
    warmup_mode: str,
    seed: int,
) -> ReplayEvalResult:
    """Replay evaluation by following ground-truth review schedule per target card."""

    RWKVSrsPredictor, RWKVSrsRlEnv, _ = require_sprwkv()
    torch_dtype = _resolve_torch_dtype(predictor_dtype)
    data_map = load_user_data_map(data_dir=Path(data_dir), user_ids=user_ids)
    targets = sample_eval_targets(
        user_data=data_map,
        cards_per_user=cards_per_user,
        min_target_occurrences=min_target_occurrences,
        warmup_mode=warmup_mode,
        seed=seed,
    )

    final_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []

    for target in targets:
        user_df = data_map[target.user_id]
        target_df = user_df[user_df["card_id"] == target.card_id].reset_index(drop=True)
        planned_rows = target_df.iloc[target.warmup_occurrence :].reset_index(drop=True)
        if planned_rows.empty:
            continue

        cursor = {"idx": 0}

        def _row_at_cursor() -> pd.Series:
            idx = min(cursor["idx"], len(planned_rows) - 1)
            return planned_rows.iloc[idx]

        def _rating_generator(*_args, **_kwargs):
            return int(_row_at_cursor()["rating"])

        def _duration_fn(*_args, **_kwargs):
            return float(_row_at_cursor()["duration"])

        def _state_fn(*_args, **_kwargs):
            return float(_row_at_cursor()["state"])

        predictor = RWKVSrsPredictor(
            model_path=predictor_model_path or None,
            device=predictor_device,
            dtype=torch_dtype,
        )
        env = RWKVSrsRlEnv(
            predictor=predictor,
            parquet_df=user_df,
            target_card_id=int(target.card_id),
            warmup_end_day_offset=float(target.warmup_end_day_offset),
            rating_generator=_rating_generator,
            duration_fn=_duration_fn,
            state_fn=_state_fn,
        )
        if hasattr(env, "prepare"):
            env.prepare()

        obs = env.reset()
        last_retention = float("nan")
        for step_idx, replay_row in planned_rows.iterrows():
            planned_day_offset = float(replay_row["day_offset"])
            current_day_offset = float(getattr(obs, "current_day_offset"))
            delta_days = max(0.0, planned_day_offset - current_day_offset)
            step_result = env.step(delta_days)

            sim_result = getattr(step_result, "sim_result", None)
            metrics = getattr(step_result.observation, "metrics", None)
            last_retention = _to_float(
                _first_non_none(
                    getattr(sim_result, "query_imm_prob", None),
                    getattr(sim_result, "step_imm_prob", None),
                    getattr(sim_result, "imm_prob", None),
                )
            )

            traj_rows.append(
                {
                    "user_id": target.user_id,
                    "card_id": target.card_id,
                    "step_index": int(step_idx),
                    "warmup_end_day_offset": float(target.warmup_end_day_offset),
                    "planned_day_offset": planned_day_offset,
                    "ground_truth_day_offset": planned_day_offset,
                    "replay_rating": int(replay_row["rating"]),
                    "replay_duration": float(replay_row["duration"]),
                    "replay_state": float(replay_row["state"]),
                    "done": bool(step_result.done),
                    "retention_area": _to_float(getattr(metrics, "retention_area", None)),
                    "target_review_count": _to_float(
                        getattr(metrics, "target_review_count", None)
                    ),
                    "total_review_count": _to_float(
                        getattr(metrics, "total_review_count", None)
                    ),
                    "final_retention": last_retention,
                }
            )

            obs = step_result.observation
            cursor["idx"] += 1
            if step_result.done:
                break

        metrics = getattr(obs, "metrics", None)
        final_rows.append(
            {
                "user_id": target.user_id,
                "card_id": target.card_id,
                "target_occurrences": target.occurrences,
                "warmup_occurrence": target.warmup_occurrence,
                "warmup_end_day_offset": float(target.warmup_end_day_offset),
                "simulation_steps": int(cursor["idx"]),
                "retention_area": _to_float(getattr(metrics, "retention_area", None)),
                "review_count": _to_float(getattr(metrics, "target_review_count", None)),
                "final_retention": last_retention,
            }
        )

    final_metrics = pd.DataFrame(final_rows)
    trajectory = pd.DataFrame(traj_rows)
    summary = {
        "num_targets": int(len(final_metrics)),
        "num_trajectory_rows": int(len(trajectory)),
        "retention_area_mean": float(final_metrics["retention_area"].mean())
        if not final_metrics.empty
        else float("nan"),
        "final_retention_mean": float(final_metrics["final_retention"].mean())
        if not final_metrics.empty
        else float("nan"),
    }
    return ReplayEvalResult(final_metrics=final_metrics, trajectory=trajectory, summary=summary)
