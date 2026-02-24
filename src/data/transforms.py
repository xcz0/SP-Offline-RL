"""Replay buffer preprocessing transforms."""

from __future__ import annotations

import numpy as np
from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd


def normalize_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer,
) -> tuple[ReplayBuffer, RunningMeanStd]:
    """Normalize obs/obs_next in-place using running mean and variance."""

    obs_rms = RunningMeanStd()
    obs_rms.update(replay_buffer.obs)
    eps = np.finfo(np.float32).eps.item()
    scale = np.sqrt(obs_rms.var + eps)

    normalized_obs = ((replay_buffer.obs - obs_rms.mean) / scale).astype(
        np.float32, copy=False
    )
    normalized_obs_next = ((replay_buffer.obs_next - obs_rms.mean) / scale).astype(
        np.float32, copy=False
    )

    replay_buffer.set_array_at_key(normalized_obs, key="obs")
    replay_buffer.set_array_at_key(normalized_obs_next, key="obs_next")
    return replay_buffer, obs_rms


def normalize_obs_array(
    obs: np.ndarray,
) -> tuple[np.ndarray, RunningMeanStd]:
    """Normalize an observation array and return normalization statistics."""

    obs_rms = RunningMeanStd()
    obs_rms.update(obs)
    eps = np.finfo(np.float32).eps.item()
    scale = np.sqrt(obs_rms.var + eps)
    normalized_obs = ((obs - obs_rms.mean) / scale).astype(np.float32, copy=False)
    return normalized_obs, obs_rms
