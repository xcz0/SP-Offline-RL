"""Metrics extraction utilities."""

from __future__ import annotations

import numpy as np
from tianshou.data import CollectStats


def _to_numpy(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    return arr


def collect_stats_to_metrics(stats: CollectStats) -> dict[str, float]:
    """Extract mean/std reward from collector stats objects."""

    arr = _to_numpy(stats.returns)
    if arr is None:
        return {"test_reward_mean": float("nan"), "test_reward_std": float("nan")}

    return {
        "test_reward_mean": float(arr.mean()),
        "test_reward_std": float(arr.std()),
    }
