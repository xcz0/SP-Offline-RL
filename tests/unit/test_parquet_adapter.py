from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.parquet_adapter import ParquetOfflineDatasetAdapter
from tests.factories.dataset_factory import write_offline_parquet, write_obs_act_parquet


def test_parquet_adapter_load_obs_act_with_minimal_mappings(
    tmp_path: Path,
    obs_act_columns: dict[str, str],
) -> None:
    path = tmp_path / "offline_obs_act.parquet"
    write_obs_act_parquet(path, n=5, seed=31)

    adapter = ParquetOfflineDatasetAdapter(
        path=str(path),
        columns=obs_act_columns,
    )
    data = adapter.load_obs_act()

    assert set(data.keys()) == {"obs", "act"}
    assert data["obs"].shape == (5, 3)
    assert data["act"].shape == (5, 1)


def test_parquet_adapter_reads_explicit_terminated_and_truncated(
    tmp_path: Path,
    full_columns: dict[str, str],
) -> None:
    path = tmp_path / "offline.parquet"
    terminated = np.array([1, 0, 0, 0, 0, 1], dtype=np.bool_)
    truncated = np.array([0, 0, 0, 1, 0, 0], dtype=np.bool_)
    frame = write_offline_parquet(
        path,
        n=6,
        seed=32,
        done=np.array([1, 0, 0, 1, 0, 0], dtype=np.bool_),
        terminated=terminated,
        truncated=truncated,
    )

    adapter = ParquetOfflineDatasetAdapter(
        path=str(path),
        columns=full_columns,
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
    full_columns: dict[str, str],
    missing_key: str,
) -> None:
    path = tmp_path / "offline.parquet"
    write_offline_parquet(path, n=6, seed=33)
    columns: dict[str, str | None] = dict(full_columns)
    columns[missing_key] = None

    adapter = ParquetOfflineDatasetAdapter(path=str(path), columns=columns)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=f"Column mapping for '{missing_key}' is required",
    ):
        adapter.load()


def test_parquet_adapter_rejects_irregular_vector_column(
    tmp_path: Path,
    full_columns: dict[str, str],
) -> None:
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

    adapter = ParquetOfflineDatasetAdapter(path=str(path), columns=full_columns)
    with pytest.raises(ValueError, match="must contain equal-length vectors"):
        adapter.load()
