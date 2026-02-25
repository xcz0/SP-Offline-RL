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
        self._workspace_obs: dict[int, np.ndarray] = {}
        self._workspace_act: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int | None):
        full_batch = batch_size == 0
        if batch_size is None:
            batch_size = self._size
        if batch_size < 0:
            indices = np.array([], dtype=np.int64)
        elif batch_size == 0:
            indices = np.arange(self._size, dtype=np.int64)
        else:
            indices = self._rng.choice(self._size, int(batch_size), replace=True)

        if indices.size == 0:
            obs_batch = self._obs[:0]
            act_batch = self._act[:0]
        elif full_batch:
            # Full-batch path returns source arrays to avoid an extra copy.
            obs_batch = self._obs
            act_batch = self._act
        else:
            size = int(indices.size)
            obs_batch = self._workspace_obs.get(size)
            act_batch = self._workspace_act.get(size)
            if obs_batch is None:
                obs_batch = np.empty((size, *self._obs.shape[1:]), dtype=self._obs.dtype)
                self._workspace_obs[size] = obs_batch
            if act_batch is None:
                act_batch = np.empty((size, *self._act.shape[1:]), dtype=self._act.dtype)
                self._workspace_act[size] = act_batch
            np.take(self._obs, indices, axis=0, out=obs_batch)
            np.take(self._act, indices, axis=0, out=act_batch)

        return (
            Batch(
                obs=obs_batch,
                act=act_batch,
                info=Batch(),
            ),
            indices,
        )
