from __future__ import annotations

import numpy as np

from src.data.replay_buffer_builder import build_replay_buffer


def _sample_data(n: int = 16) -> dict[str, np.ndarray]:
    return {
        "obs": np.random.randn(n, 3).astype(np.float32),
        "act": np.random.randn(n, 1).astype(np.float32),
        "rew": np.random.randn(n).astype(np.float32),
        "done": np.zeros(n, dtype=np.bool_),
        "obs_next": np.random.randn(n, 3).astype(np.float32),
        "terminated": np.zeros(n, dtype=np.bool_),
        "truncated": np.zeros(n, dtype=np.bool_),
    }


def test_build_replay_buffer_full_and_truncated() -> None:
    data = _sample_data(16)
    buffer = build_replay_buffer(data)
    assert len(buffer) == 16

    truncated = build_replay_buffer(data, buffer_size=5)
    assert len(truncated) == 5
