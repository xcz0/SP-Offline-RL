from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_offline_dataset(
    n: int,
    *,
    seed: int = 0,
    obs_dim: int = 3,
    act_dim: int = 1,
    done: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    act = rng.uniform(-1.0, 1.0, size=(n, act_dim)).astype(np.float32)
    rew = rng.normal(size=n).astype(np.float32)
    obs_next = rng.normal(size=(n, obs_dim)).astype(np.float32)

    done_array = np.zeros(n, dtype=np.bool_) if done is None else np.asarray(done, dtype=np.bool_)
    terminated_array = (
        done_array.copy() if terminated is None else np.asarray(terminated, dtype=np.bool_)
    )
    truncated_array = (
        np.zeros(n, dtype=np.bool_) if truncated is None else np.asarray(truncated, dtype=np.bool_)
    )

    return {
        "obs": obs,
        "act": act,
        "rew": rew,
        "done": done_array,
        "terminated": terminated_array,
        "truncated": truncated_array,
        "obs_next": obs_next,
    }


def build_obs_act_dataset(
    n: int,
    *,
    seed: int = 0,
    obs_dim: int = 3,
    act_dim: int = 1,
) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    return {
        "obs": rng.normal(size=(n, obs_dim)).astype(np.float32),
        "act": rng.uniform(-1.0, 1.0, size=(n, act_dim)).astype(np.float32),
    }


def build_parquet_frame(data: dict[str, np.ndarray]) -> pd.DataFrame:
    frame_payload: dict[str, object] = {}
    for key, value in data.items():
        if key in {"obs", "act", "obs_next"}:
            frame_payload[key] = [row.tolist() for row in value]
        else:
            frame_payload[key] = value
    return pd.DataFrame(frame_payload)


def write_offline_parquet(
    path: Path,
    *,
    n: int = 64,
    seed: int = 0,
    done: np.ndarray | None = None,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = build_parquet_frame(
        build_offline_dataset(
            n,
            seed=seed,
            done=done,
            terminated=terminated,
            truncated=truncated,
        )
    )
    frame.to_parquet(path, index=False)
    return frame


def write_obs_act_parquet(
    path: Path,
    *,
    n: int = 64,
    seed: int = 0,
) -> pd.DataFrame:
    frame = build_parquet_frame(build_obs_act_dataset(n, seed=seed))
    frame.to_parquet(path, index=False)
    return frame
