"""Replay buffer preprocessing transforms."""

from __future__ import annotations

import numpy as np
from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd


def compute_obs_norm_stats(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute observation normalization mean/variance in float32."""

    obs_rms = RunningMeanStd()
    obs_rms.update(obs)
    mean = np.asarray(obs_rms.mean, dtype=np.float32)
    var = np.asarray(obs_rms.var, dtype=np.float32)
    return mean, var


def normalize_obs_with_stats(
    obs: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    """Normalize observations from precomputed mean/variance."""

    eps = np.finfo(np.float32).eps.item()
    scale = np.sqrt(var + eps)
    return ((obs - mean) / scale).astype(np.float32, copy=False)


def normalize_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer,
) -> tuple[ReplayBuffer, RunningMeanStd]:
    """Normalize obs/obs_next in-place using running mean and variance."""

    mean, var = compute_obs_norm_stats(replay_buffer.obs)
    normalized_obs = normalize_obs_with_stats(replay_buffer.obs, mean, var)
    normalized_obs_next = normalize_obs_with_stats(replay_buffer.obs_next, mean, var)

    replay_buffer.set_array_at_key(normalized_obs, key="obs")
    replay_buffer.set_array_at_key(normalized_obs_next, key="obs_next")
    obs_rms = RunningMeanStd(mean=mean, std=var)
    return replay_buffer, obs_rms


def normalize_obs_array(
    obs: np.ndarray,
) -> tuple[np.ndarray, RunningMeanStd]:
    """Normalize an observation array and return normalization statistics."""

    mean, var = compute_obs_norm_stats(obs)
    normalized_obs = normalize_obs_with_stats(obs, mean, var)
    obs_rms = RunningMeanStd(mean=mean, std=var)
    return normalized_obs, obs_rms
