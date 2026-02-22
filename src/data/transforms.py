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

    replay_buffer._meta["obs"] = (replay_buffer.obs - obs_rms.mean) / np.sqrt(
        obs_rms.var + eps
    )
    replay_buffer._meta["obs_next"] = (replay_buffer.obs_next - obs_rms.mean) / np.sqrt(
        obs_rms.var + eps
    )
    return replay_buffer, obs_rms
