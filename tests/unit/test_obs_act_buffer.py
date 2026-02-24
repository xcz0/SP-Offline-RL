from __future__ import annotations

import numpy as np
import pytest

from src.data.obs_act_buffer import ObsActBuffer


def test_obs_act_buffer_sample_shapes() -> None:
    n = 12
    buffer = ObsActBuffer(
        obs=np.random.randn(n, 3).astype(np.float32),
        act=np.random.randn(n, 1).astype(np.float32),
        seed=0,
    )

    batch, indices = buffer.sample(batch_size=5)
    assert len(indices) == 5
    assert batch.obs.shape == (5, 3)
    assert batch.act.shape == (5, 1)


def test_obs_act_buffer_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        ObsActBuffer(
            obs=np.empty((0, 3), dtype=np.float32),
            act=np.empty((0, 1), dtype=np.float32),
        )
