"""Policy-driven simulator evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tianshou.data import Batch

from src.core.exceptions import ConfigurationError
from src.data.bc_dataset_builder import CARD_FEATURE_COLUMNS
from src.evaluation.dataset import load_user_data_map, sample_eval_targets
from src.evaluation.deps import require_sprwkv
from src.evaluation.scoring import add_score_column, summarize_scored_metrics
from src.evaluation.types import PolicyEvalResult


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


def _extract_obs_vector(last_target_row: dict[str, Any] | None) -> np.ndarray:
    if last_target_row is None:
        raise ConfigurationError(
            "Simulator observation has no last_target_row; cannot build BC feature vector."
        )

    values: list[float] = []
    missing: list[str] = []
    for col in CARD_FEATURE_COLUMNS:
        if col not in last_target_row:
            missing.append(col)
        else:
            values.append(float(last_target_row[col]))
    if missing:
        preview = ", ".join(missing[:5])
        raise ConfigurationError(
            "Simulator row missing BC feature columns. "
            f"Missing examples: {preview}"
        )
    return np.asarray(values, dtype=np.float32)


def _normalize_obs_vector(
    obs_vector: np.ndarray,
    obs_mean: np.ndarray | None,
    obs_var: np.ndarray | None,
) -> np.ndarray:
    if obs_mean is None or obs_var is None:
        return obs_vector
    eps = np.finfo(np.float32).eps.item()
    scale = np.sqrt(obs_var + eps)
    return ((obs_vector - obs_mean) / scale).astype(np.float32, copy=False)


def _policy_act_days(policy: Any, obs_vector: np.ndarray) -> float:
    batch = Batch(obs=obs_vector[None, :], info=Batch())
    output = policy(batch)
    act = getattr(output, "act", output)
    if hasattr(act, "detach"):
        act = act.detach().cpu().numpy()
    arr = np.asarray(act, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ConfigurationError("Policy returned empty action output.")
    return float(arr[0])


def _clip_delta_days(delta_days: float, action_max: float) -> float:
    return float(np.clip(delta_days, 0.0, action_max))


def _final_retention_from_step_result(step_result: Any) -> float:
    sim_result = getattr(step_result, "sim_result", None)
    if sim_result is None:
        return float("nan")

    # Best-effort extraction for both SimulatedReview and TerminalSettlement fields.
    for attr in ("query_imm_prob", "step_imm_prob", "imm_prob", "final_retention"):
        value = getattr(sim_result, attr, None)
        if value is not None:
            try:
                if hasattr(value, "item"):
                    return float(value.item())
                return float(value)
            except Exception:  # noqa: BLE001
                continue
    return float("nan")


def evaluate_policy_with_simulator(
    *,
    policy: Any,
    data_dir: str,
    predictor_model_path: str,
    predictor_device: str,
    predictor_dtype: str,
    user_ids: list[int] | None,
    cards_per_user: int,
    min_target_occurrences: int,
    warmup_mode: str,
    seed: int,
    action_max: float,
    score_weights: dict[str, Any],
    obs_mean: list[float] | None = None,
    obs_var: list[float] | None = None,
    eval_workers: int = 1,
) -> PolicyEvalResult:
    """Evaluate policy by rolling out in RWKVSrsRlEnv."""

    RWKVSrsPredictor, RWKVSrsRlEnv, _ = require_sprwkv()
    torch_dtype = _resolve_torch_dtype(predictor_dtype)
    data_map = load_user_data_map(data_dir=Path(data_dir), user_ids=user_ids)
    obs_mean_arr = (
        np.asarray(obs_mean, dtype=np.float32)
        if obs_mean is not None and len(obs_mean) > 0
        else None
    )
    obs_var_arr = (
        np.asarray(obs_var, dtype=np.float32)
        if obs_var is not None and len(obs_var) > 0
        else None
    )
    targets = sample_eval_targets(
        user_data=data_map,
        cards_per_user=cards_per_user,
        min_target_occurrences=min_target_occurrences,
        warmup_mode=warmup_mode,
        seed=seed,
    )

    _ = eval_workers  # Policy inference object is shared; keep per-user serial execution.
    targets_by_user: dict[int, list[Any]] = defaultdict(list)
    for target in targets:
        targets_by_user[int(target.user_id)].append(target)

    rows: list[dict[str, Any]] = []
    for user_id in sorted(targets_by_user):
        user_data = data_map[user_id]

        predictor = RWKVSrsPredictor(
            model_path=predictor_model_path or None,
            device=predictor_device,
            dtype=torch_dtype,
        )
        user_df = user_data.frame
        for target in targets_by_user[user_id]:
            target_df = user_data.get_card_frame(target.card_id)
            if target_df is None or target_df.empty:
                continue

            env = RWKVSrsRlEnv(
                predictor=predictor,
                # Keep full feature columns for BC observation extraction.
                background_rows=user_df.to_dict("records"),
                target_card_id=int(target.card_id),
                warmup_end_day_offset=float(target.warmup_end_day_offset),
            )
            if hasattr(env, "prepare"):
                env.prepare()

            obs = env.reset()
            done = False
            steps = 0
            max_steps = max(1, int(target.occurrences - target.warmup_occurrence))
            final_retention = float("nan")

            while not done and steps < max_steps:
                last_target_row = getattr(obs, "last_target_row", None)
                obs_vec = _extract_obs_vector(last_target_row)
                obs_vec = _normalize_obs_vector(obs_vec, obs_mean_arr, obs_var_arr)
                delta_days = _policy_act_days(policy, obs_vec)
                delta_days = _clip_delta_days(delta_days, action_max=action_max)
                step_result = env.step(delta_days)
                final_retention = _final_retention_from_step_result(step_result)
                obs = step_result.observation
                done = bool(step_result.done)
                steps += 1

            metrics = getattr(obs, "metrics", None)
            retention_area = float(getattr(metrics, "retention_area", float("nan")))
            review_count = float(getattr(metrics, "target_review_count", steps))
            rows.append(
                {
                    "user_id": target.user_id,
                    "card_id": target.card_id,
                    "retention_area": retention_area,
                    "final_retention": final_retention,
                    "review_count": review_count,
                    "episode_steps": float(steps),
                    "warmup_mode": warmup_mode,
                    "warmup_end_day_offset": target.warmup_end_day_offset,
                }
            )

    metrics_frame = pd.DataFrame(rows)
    scored_frame = add_score_column(metrics_frame, weights=score_weights)
    summary = summarize_scored_metrics(scored_frame)
    summary["warmup_mode"] = warmup_mode
    summary["cards_per_user"] = float(cards_per_user)
    summary["targets_total"] = float(len(targets))
    return PolicyEvalResult(per_target_metrics=scored_frame, summary=summary)
