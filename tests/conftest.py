from __future__ import annotations

import os
import random
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from tests.factories.config_factory import FULL_PARQUET_COLUMNS, OBS_ACT_COLUMNS, build_train_cfg
from tests.factories.dataset_factory import write_offline_parquet, write_obs_act_parquet

# Keep torch on CPU-only path in test process to avoid expensive GPU probing.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


class _SPRLTestEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 5) -> None:
        self._max_steps = int(max_steps)
        self._step_count = 0
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        clipped_action = np.asarray(action, dtype=np.float32).reshape(1)
        obs = np.array(
            [
                min(self._step_count / float(self._max_steps), 1.0),
                float(np.clip(clipped_action[0], -1.0, 1.0)),
                0.0,
            ],
            dtype=np.float32,
        )
        reward = float(1.0 - abs(obs[1]))
        terminated = self._step_count >= self._max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}


if "SPRLTestEnv-v0" not in gym.registry:
    gym.register(id="SPRLTestEnv-v0", entry_point=_SPRLTestEnv, max_episode_steps=5)


@pytest.fixture(autouse=True)
def _stable_random_seed() -> None:
    random.seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session", autouse=True)
def _patch_torch_cuda_probes() -> None:
    import torch

    if hasattr(torch.cuda, "is_current_stream_capturing"):
        torch.cuda.is_current_stream_capturing = lambda: False  # type: ignore[assignment]


@pytest.fixture
def full_columns() -> dict[str, str]:
    return dict(FULL_PARQUET_COLUMNS)


@pytest.fixture
def obs_act_columns() -> dict[str, str]:
    return dict(OBS_ACT_COLUMNS)


@pytest.fixture
def write_offline_dataset(tmp_path: Path) -> Callable[..., Path]:
    def _write(name: str = "offline.parquet", *, n: int = 64, seed: int = 0) -> Path:
        path = tmp_path / name
        write_offline_parquet(path, n=n, seed=seed)
        return path

    return _write


@pytest.fixture
def write_obs_act_dataset(tmp_path: Path) -> Callable[..., Path]:
    def _write(name: str = "offline_obs_act.parquet", *, n: int = 64, seed: int = 0) -> Path:
        path = tmp_path / name
        write_obs_act_parquet(path, n=n, seed=seed)
        return path

    return _write


@pytest.fixture
def make_train_cfg(tmp_path: Path) -> Callable[..., object]:
    def _make(
        dataset_path: Path,
        *,
        columns: dict[str, str] | None = None,
        obs_norm: bool = False,
    ) -> object:
        return build_train_cfg(
            tmp_path,
            dataset_path,
            data_columns=columns,
            obs_norm=obs_norm,
        )

    return _make
