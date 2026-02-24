from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.parquet_adapter import ParquetOfflineDatasetAdapter


def _write_dataset(path: Path, n: int = 6) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "obs": [x.tolist() for x in np.random.randn(n, 3).astype(np.float32)],
            "act": [x.tolist() for x in np.random.randn(n, 1).astype(np.float32)],
            "rew": np.random.randn(n).astype(np.float32),
            "done": np.array([1, 0, 0, 1, 0, 0], dtype=np.bool_),
            "terminated": np.array([1, 0, 0, 0, 0, 1], dtype=np.bool_),
            "truncated": np.array([0, 0, 0, 1, 0, 0], dtype=np.bool_),
            "obs_next": [x.tolist() for x in np.random.randn(n, 3).astype(np.float32)],
        }
    )
    frame.to_parquet(path, index=False)
    return frame


def _columns() -> dict[str, str]:
    return {
        "obs": "obs",
        "act": "act",
        "rew": "rew",
        "done": "done",
        "obs_next": "obs_next",
        "terminated": "terminated",
        "truncated": "truncated",
    }


def test_parquet_adapter_reads_explicit_terminated_and_truncated(tmp_path: Path) -> None:
    path = tmp_path / "offline.parquet"
    frame = _write_dataset(path)

    adapter = ParquetOfflineDatasetAdapter(
        path=str(path),
        columns=_columns(),
    )
    data = adapter.load()

    np.testing.assert_array_equal(
        data["terminated"],
        frame["terminated"].to_numpy(dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        data["truncated"],
        frame["truncated"].to_numpy(dtype=np.bool_),
    )


@pytest.mark.parametrize("missing_key", ["terminated", "truncated"])
def test_parquet_adapter_requires_terminal_mappings(
    tmp_path: Path,
    missing_key: str,
) -> None:
    path = tmp_path / "offline.parquet"
    _write_dataset(path)
    columns: dict[str, str | None] = _columns()
    columns[missing_key] = None

    adapter = ParquetOfflineDatasetAdapter(path=str(path), columns=columns)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=f"Column mapping for '{missing_key}' is required",
    ):
        adapter.load()


def test_parquet_adapter_rejects_irregular_vector_column(tmp_path: Path) -> None:
    path = tmp_path / "offline_irregular.parquet"
    frame = pd.DataFrame(
        {
            "obs": [[0.1, 0.2, 0.3], [0.4, 0.5]],
            "act": [[0.0], [0.1]],
            "rew": np.array([0.0, 1.0], dtype=np.float32),
            "done": np.array([0, 1], dtype=np.bool_),
            "terminated": np.array([0, 1], dtype=np.bool_),
            "truncated": np.array([0, 0], dtype=np.bool_),
            "obs_next": [[0.2, 0.3, 0.4], [0.5, 0.6]],
        }
    )
    frame.to_parquet(path, index=False)

    adapter = ParquetOfflineDatasetAdapter(path=str(path), columns=_columns())
    with pytest.raises(ValueError, match="must contain equal-length vectors"):
        adapter.load()
