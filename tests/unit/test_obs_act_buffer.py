from __future__ import annotations

import numpy as np
import pytest

from src.data.obs_act_buffer import ObsActBuffer
from tests.factories.dataset_factory import build_obs_act_dataset


def test_obs_act_buffer_sample_shapes() -> None:
    n = 12
    data = build_obs_act_dataset(n=n, seed=61)
    buffer = ObsActBuffer(
        obs=data["obs"],
        act=data["act"],
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
