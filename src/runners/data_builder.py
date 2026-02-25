"""Dataset preparation helpers shared by train/eval runtimes."""

from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig
from tianshou.data import ReplayBuffer

from src.core.types import PreparedDataset
from src.data.dataset_adapter import BC_DATA_FIELDS, RL_DATA_FIELDS, build_dataset_adapter
from src.data.obs_act_buffer import ObsActBuffer
from src.data.replay_buffer_builder import build_replay_buffer
from src.data.schema import (
    validate_against_env,
    validate_and_standardize_dataset,
    validate_obs_act_dataset,
)
from src.data.transforms import normalize_obs_array, normalize_obs_with_stats


def _build_meta(data: dict[str, np.ndarray]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "fields": sorted(data.keys()),
        "size": int(next(iter(data.values())).shape[0]) if data else 0,
    }
    if "obs" in data:
        meta["obs_shape"] = tuple(int(v) for v in data["obs"].shape[1:])
    if "act" in data:
        meta["act_shape"] = tuple(int(v) for v in data["act"].shape[1:])
    return meta


def _normalize_prepared_data(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    normalized_obs, obs_rms = normalize_obs_array(data["obs"])
    obs_mean = np.asarray(obs_rms.mean, dtype=np.float32)
    obs_var = np.asarray(obs_rms.var, dtype=np.float32)
    normalized = dict(data)
    normalized["obs"] = normalized_obs
    if "obs_next" in normalized:
        normalized["obs_next"] = normalize_obs_with_stats(
            normalized["obs_next"],
            obs_mean,
            obs_var,
        )
    return normalized, obs_mean, obs_var


def load_rl_prepared_dataset(
    cfg: DictConfig,
    env: Any,
) -> PreparedDataset:
    """Load canonical RL fields and optionally normalize observations."""

    adapter = build_dataset_adapter(cfg.data)
    raw = adapter.load_prepared(fields=RL_DATA_FIELDS)
    data = validate_and_standardize_dataset(raw)
    validate_against_env(data, env)

    obs_mean: np.ndarray | None = None
    obs_var: np.ndarray | None = None
    if bool(cfg.data.obs_norm):
        data, obs_mean, obs_var = _normalize_prepared_data(data)

    return PreparedDataset(
        arrays=data,
        obs_norm_mean=obs_mean,
        obs_norm_var=obs_var,
        meta=_build_meta(data),
    )


def load_bc_prepared_dataset(
    cfg: DictConfig,
    env: Any | None,
) -> PreparedDataset:
    """Load observation/action fields for behavior cloning workflows."""

    adapter = build_dataset_adapter(cfg.data)
    raw = adapter.load_prepared(fields=BC_DATA_FIELDS)
    data = validate_obs_act_dataset(raw)
    if env is not None:
        validate_against_env(data, env)

    obs_mean: np.ndarray | None = None
    obs_var: np.ndarray | None = None
    if bool(cfg.data.obs_norm):
        data, obs_mean, obs_var = _normalize_prepared_data(data)

    return PreparedDataset(
        arrays=data,
        obs_norm_mean=obs_mean,
        obs_norm_var=obs_var,
        meta=_build_meta(data),
    )


def build_replay_buffer_from_prepared(
    prepared: PreparedDataset,
    buffer_size: int | None,
) -> ReplayBuffer:
    """Build replay buffer from prepared canonical fields."""

    return build_replay_buffer(prepared.to_dict(), buffer_size, validate=False)


def build_obs_act_buffer_from_prepared(
    prepared: PreparedDataset,
    seed: int,
) -> ObsActBuffer:
    """Build obs/act buffer from prepared behavior-cloning fields."""

    arrays = prepared.to_dict()
    return ObsActBuffer(obs=arrays["obs"], act=arrays["act"], seed=seed)


def obs_norm_stats_as_lists(
    prepared: PreparedDataset,
) -> tuple[list[float] | None, list[float] | None]:
    """Return optional normalization stats as plain Python lists."""

    if prepared.obs_norm_mean is None or prepared.obs_norm_var is None:
        return None, None
    return (
        np.asarray(prepared.obs_norm_mean, dtype=np.float32).tolist(),
        np.asarray(prepared.obs_norm_var, dtype=np.float32).tolist(),
    )
