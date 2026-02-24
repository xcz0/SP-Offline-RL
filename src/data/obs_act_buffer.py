"""Lightweight sampler for behavior-cloning observation/action datasets."""

from __future__ import annotations

import numpy as np
from tianshou.data import Batch


class ObsActBuffer:
    """A minimal buffer that supports Tianshou's offline update API."""

    def __init__(self, obs: np.ndarray, act: np.ndarray, seed: int | None = None) -> None:
        if obs.shape[0] != act.shape[0]:
            raise ValueError(
                "Observation/action length mismatch: "
                f"{obs.shape[0]} vs {act.shape[0]}."
            )
        if obs.shape[0] == 0:
            raise ValueError("Behavior-cloning dataset is empty.")

        self._obs = obs
        self._act = act
        self._size = int(obs.shape[0])
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int | None):
        if batch_size is None:
            batch_size = self._size
        if batch_size < 0:
            indices = np.array([], dtype=np.int64)
        elif batch_size == 0:
            indices = np.arange(self._size, dtype=np.int64)
        else:
            indices = self._rng.choice(self._size, int(batch_size), replace=True)

        return (
            Batch(
                obs=self._obs[indices],
                act=self._act[indices],
                info=Batch(),
            ),
            indices,
        )
