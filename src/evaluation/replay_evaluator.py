"""Ground-truth replay evaluation on RWKV simulator."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import ConfigurationError
from src.evaluation.dataset import UserEvalData, load_user_data_map, sample_eval_targets
from src.evaluation.deps import require_sprwkv, resolve_predictor_device
from src.evaluation.types import EvalTarget, ReplayEvalResult


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


def _evaluate_replay_for_user(
    *,
    user_id: int,
    user_data: UserEvalData,
    user_targets: list[EvalTarget],
    RWKVSrsPredictor: Any,
    RWKVSrsRlEnv: Any,
    predictor_model_path: str,
    predictor_device: str,
    torch_dtype: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    for target in user_targets:
        target_df = user_data.get_card_frame(target.card_id)
        if target_df is None or target_df.empty:
            continue
        planned_rows = target_df.iloc[target.warmup_occurrence :].reset_index(drop=True)
        if planned_rows.empty:
            continue

        planned_day_offsets = planned_rows["day_offset"].to_numpy(dtype=np.float32, copy=False)
        planned_ratings = planned_rows["rating"].to_numpy(dtype=np.int64, copy=False)
        planned_durations = planned_rows["duration"].to_numpy(dtype=np.float32, copy=False)
        planned_states = planned_rows["state"].to_numpy(dtype=np.float32, copy=False)
        planned_len = int(planned_day_offsets.shape[0])
        if planned_len == 0:
            continue

        cursor_idx = 0

        def _active_idx() -> int:
            return min(cursor_idx, planned_len - 1)

        predictor = RWKVSrsPredictor(
            model_path=predictor_model_path or None,
            device=predictor_device,
            dtype=torch_dtype,
        )

        class _ReplayPredictorProxy:
            def __init__(self, base: Any, ratings: np.ndarray) -> None:
                self._base = base
                self._ratings = ratings
                self._cursor = 0

            def reset_state(self) -> None:
                self._cursor = 0
                self._base.reset_state()

            def sample_rating(self, query_row: dict[str, Any], generator: Any = None):
                _ = generator
                idx = min(self._cursor, len(self._ratings) - 1)
                rating = int(self._ratings[idx])
                self._cursor += 1
                query_out = self._base.predict_query(query_row)
                return rating, query_out

            def __getattr__(self, name: str) -> Any:
                return getattr(self._base, name)

        replay_predictor = _ReplayPredictorProxy(predictor, planned_ratings)

        def _duration_fn(*_args, **_kwargs):
            return float(planned_durations[_active_idx()])

        def _state_fn(*_args, **_kwargs):
            return float(planned_states[_active_idx()])

        env = RWKVSrsRlEnv(
            predictor=replay_predictor,
            parquet_df=user_data.frame,
            target_card_id=int(target.card_id),
            warmup_end_day_offset=float(target.warmup_end_day_offset),
            duration_fn=_duration_fn,
            state_fn=_state_fn,
        )
        if hasattr(env, "prepare"):
            env.prepare()

        obs = env.reset()
        last_retention = float("nan")
        for step_idx in range(planned_len):
            planned_day_offset = float(planned_day_offsets[step_idx])
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
                    "user_id": user_id,
                    "card_id": target.card_id,
                    "step_index": int(step_idx),
                    "warmup_end_day_offset": float(target.warmup_end_day_offset),
                    "planned_day_offset": planned_day_offset,
                    "ground_truth_day_offset": planned_day_offset,
                    "replay_rating": int(planned_ratings[step_idx]),
                    "replay_duration": float(planned_durations[step_idx]),
                    "replay_state": float(planned_states[step_idx]),
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
            cursor_idx += 1
            if step_result.done:
                break

        metrics = getattr(obs, "metrics", None)
        final_rows.append(
            {
                "user_id": user_id,
                "card_id": target.card_id,
                "target_occurrences": target.occurrences,
                "warmup_occurrence": target.warmup_occurrence,
                "warmup_end_day_offset": float(target.warmup_end_day_offset),
                "simulation_steps": int(cursor_idx),
                "retention_area": _to_float(getattr(metrics, "retention_area", None)),
                "review_count": _to_float(getattr(metrics, "target_review_count", None)),
                "final_retention": last_retention,
            }
        )
    return final_rows, traj_rows


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
    eval_workers: int = 1,
) -> ReplayEvalResult:
    """Replay evaluation by following ground-truth review schedule per target card."""

    RWKVSrsPredictor, RWKVSrsRlEnv, _ = require_sprwkv()
    resolved_predictor_device = resolve_predictor_device(predictor_device)
    torch_dtype = _resolve_torch_dtype(predictor_dtype)
    data_map = load_user_data_map(data_dir=Path(data_dir), user_ids=user_ids)
    targets = sample_eval_targets(
        user_data=data_map,
        cards_per_user=cards_per_user,
        min_target_occurrences=min_target_occurrences,
        warmup_mode=warmup_mode,
        seed=seed,
    )

    targets_by_user: dict[int, list[EvalTarget]] = defaultdict(list)
    for target in targets:
        targets_by_user[int(target.user_id)].append(target)

    workers = max(1, min(int(eval_workers), len(targets_by_user))) if targets_by_user else 1
    final_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                user_id: executor.submit(
                    _evaluate_replay_for_user,
                    user_id=user_id,
                    user_data=data_map[user_id],
                    user_targets=targets_by_user[user_id],
                    RWKVSrsPredictor=RWKVSrsPredictor,
                    RWKVSrsRlEnv=RWKVSrsRlEnv,
                    predictor_model_path=predictor_model_path,
                    predictor_device=resolved_predictor_device,
                    torch_dtype=torch_dtype,
                )
                for user_id in sorted(targets_by_user)
            }
            for user_id in sorted(future_map):
                user_final, user_traj = future_map[user_id].result()
                final_rows.extend(user_final)
                traj_rows.extend(user_traj)
    else:
        for user_id in sorted(targets_by_user):
            user_final, user_traj = _evaluate_replay_for_user(
                user_id=user_id,
                user_data=data_map[user_id],
                user_targets=targets_by_user[user_id],
                RWKVSrsPredictor=RWKVSrsPredictor,
                RWKVSrsRlEnv=RWKVSrsRlEnv,
                predictor_model_path=predictor_model_path,
                predictor_device=resolved_predictor_device,
                torch_dtype=torch_dtype,
            )
            final_rows.extend(user_final)
            traj_rows.extend(user_traj)

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
