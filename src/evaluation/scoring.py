"""Composite scoring for simulator evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_float_weight(weights: dict[str, Any], key: str, default: float) -> float:
    value = weights.get(key, default)
    return float(value)


def compute_score_from_metrics(
    retention_area: np.ndarray,
    final_retention: np.ndarray,
    review_count: np.ndarray,
    weights: dict[str, Any],
) -> np.ndarray:
    """Compute per-target composite score."""

    w_area = _to_float_weight(weights, "retention_area", 0.5)
    w_final = _to_float_weight(weights, "final_retention", 0.3)
    w_review = _to_float_weight(weights, "review_count_norm", 0.2)

    review = np.asarray(review_count, dtype=np.float32)
    review_norm = review / max(1.0, float(review.max(initial=1.0)))

    score = (
        w_area * np.asarray(retention_area, dtype=np.float32)
        + w_final * np.asarray(final_retention, dtype=np.float32)
        - w_review * review_norm
    )
    return score.astype(np.float32, copy=False)


def add_score_column(
    frame: pd.DataFrame,
    weights: dict[str, Any],
) -> pd.DataFrame:
    """Append composite score column to a metrics frame."""

    if frame.empty:
        out = frame.copy()
        out["score"] = np.array([], dtype=np.float32)
        return out

    required = {"retention_area", "final_retention", "review_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Metrics frame missing required columns: {missing_text}")

    out = frame.copy()
    out["score"] = compute_score_from_metrics(
        retention_area=out["retention_area"].to_numpy(dtype=np.float32, copy=False),
        final_retention=out["final_retention"].to_numpy(dtype=np.float32, copy=False),
        review_count=out["review_count"].to_numpy(dtype=np.float32, copy=False),
        weights=weights,
    )
    return out


def summarize_scored_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Return summary stats for scored per-target metrics."""

    if frame.empty:
        return {
            "num_targets": 0.0,
            "score_mean": float("nan"),
            "score_std": float("nan"),
            "retention_area_mean": float("nan"),
            "final_retention_mean": float("nan"),
            "review_count_mean": float("nan"),
        }

    return {
        "num_targets": float(len(frame)),
        "score_mean": float(frame["score"].mean()),
        "score_std": float(frame["score"].std(ddof=0)),
        "retention_area_mean": float(frame["retention_area"].mean()),
        "final_retention_mean": float(frame["final_retention"].mean()),
        "review_count_mean": float(frame["review_count"].mean()),
    }

