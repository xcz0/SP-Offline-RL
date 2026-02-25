"""Observation normalization helpers for runtime builders."""

from __future__ import annotations

import numpy as np
from tianshou.env import BaseVectorEnv, VectorEnvNormObs
from tianshou.utils import RunningMeanStd


def make_obs_rms(mean: np.ndarray, var: np.ndarray) -> RunningMeanStd:
    """Build a RunningMeanStd container from explicit statistics."""

    return RunningMeanStd(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(var, dtype=np.float32),
    )


def apply_obs_norm_to_envs(
    test_envs: BaseVectorEnv,
    mean: np.ndarray | None,
    var: np.ndarray | None,
) -> BaseVectorEnv:
    """Attach fixed observation normalization stats to vector envs."""

    if mean is None or var is None:
        return test_envs
    normalized_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    normalized_envs.set_obs_rms(make_obs_rms(mean, var))
    return normalized_envs
