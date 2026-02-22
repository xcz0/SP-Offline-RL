"""Parquet dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.core.types import DatasetDict
from src.data.dataset_adapter import OfflineDatasetAdapter


class ParquetOfflineDatasetAdapter(OfflineDatasetAdapter):
    """Loads offline transitions from a Parquet table with configurable columns."""

    def __init__(self, path: str, columns: dict[str, str | None]) -> None:
        self.path = Path(path)
        self.columns = columns

    def _stack_column(self, table: dict[str, Any], name: str) -> np.ndarray:
        raw = table[name]
        # supports list/array object columns.
        return np.asarray(raw.tolist(), dtype=np.float32)

    def load(self) -> DatasetDict:
        frame = pq.read_table(self.path).to_pandas()
        cols = self.columns
        done = frame[cols["done"]].to_numpy(dtype=np.bool_)

        data: DatasetDict = {
            "obs": self._stack_column(frame, cols["obs"]),
            "act": self._stack_column(frame, cols["act"]),
            "rew": frame[cols["rew"]].to_numpy(dtype=np.float32),
            "done": done,
            "obs_next": self._stack_column(frame, cols["obs_next"]),
        }

        terminated_col = cols.get("terminated")
        truncated_col = cols.get("truncated")
        if terminated_col:
            data["terminated"] = frame[terminated_col].to_numpy(dtype=np.bool_)
        else:
            data["terminated"] = done
        if truncated_col:
            data["truncated"] = frame[truncated_col].to_numpy(dtype=np.bool_)
        else:
            data["truncated"] = np.zeros_like(done, dtype=np.bool_)

        return data
