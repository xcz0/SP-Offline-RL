from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.parquet_adapter import ParquetOfflineDatasetAdapter


def test_parquet_adapter_generates_terminated_and_truncated(tmp_path: Path) -> None:
    path = tmp_path / "offline.parquet"
    n = 6
    frame = pd.DataFrame(
        {
            "obs": [x.tolist() for x in np.random.randn(n, 3).astype(np.float32)],
            "act": [x.tolist() for x in np.random.randn(n, 1).astype(np.float32)],
            "rew": np.random.randn(n).astype(np.float32),
            "done": np.array([1, 0, 0, 1, 0, 0], dtype=np.bool_),
            "obs_next": [x.tolist() for x in np.random.randn(n, 3).astype(np.float32)],
        }
    )
    frame.to_parquet(path, index=False)

    adapter = ParquetOfflineDatasetAdapter(
        path=str(path),
        columns={
            "obs": "obs",
            "act": "act",
            "rew": "rew",
            "done": "done",
            "obs_next": "obs_next",
            "terminated": None,
            "truncated": None,
        },
    )
    data = adapter.load()

    np.testing.assert_array_equal(data["terminated"], data["done"])
    np.testing.assert_array_equal(data["truncated"], np.zeros(n, dtype=np.bool_))
