"""Parquet dataset adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.types import DatasetDict
from src.data.dataset_adapter import OfflineDatasetAdapter


class ParquetOfflineDatasetAdapter(OfflineDatasetAdapter):
    """Loads offline transitions from a Parquet table with configurable columns."""

    def __init__(self, path: str, columns: dict[str, str]) -> None:
        self.path = Path(path)
        self.columns = columns

    def _required_column_name(self, key: str) -> str:
        value = self.columns.get(key)
        if not value:
            raise ValueError(f"Column mapping for '{key}' is required.")
        return value

    def _scalar_column(self, table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
        return np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=dtype)

    def _stack_column(self, table: pa.Table, name: str) -> np.ndarray:
        column = table[name].combine_chunks()
        if pa.types.is_fixed_size_list(column.type):
            values = np.asarray(
                column.values.to_numpy(zero_copy_only=False), dtype=np.float32
            )
            return values.reshape(len(column), int(column.type.list_size))

        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            offsets = np.asarray(column.offsets.to_numpy(zero_copy_only=False))
            row_lengths = np.diff(offsets)
            if row_lengths.size == 0:
                return np.empty((len(column), 0), dtype=np.float32)
            if np.all(row_lengths == row_lengths[0]):
                values = np.asarray(
                    column.values.to_numpy(zero_copy_only=False), dtype=np.float32
                )
                return values.reshape(len(column), int(row_lengths[0]))
            raise ValueError(
                f"Column '{name}' must contain equal-length vectors, got row lengths "
                f"{row_lengths.min()}..{row_lengths.max()}."
            )

        raise ValueError(
            f"Column '{name}' must be fixed_size_list or list type, got {column.type}."
        )

    def load(self) -> DatasetDict:
        obs_col = self._required_column_name("obs")
        act_col = self._required_column_name("act")
        rew_col = self._required_column_name("rew")
        done_col = self._required_column_name("done")
        obs_next_col = self._required_column_name("obs_next")
        terminated_col = self._required_column_name("terminated")
        truncated_col = self._required_column_name("truncated")

        column_names = list(
            dict.fromkeys(
                [
                    obs_col,
                    act_col,
                    rew_col,
                    done_col,
                    obs_next_col,
                    terminated_col,
                    truncated_col,
                ]
            )
        )
        table = pq.read_table(self.path, columns=column_names, use_threads=True)
        done = self._scalar_column(table, done_col, np.bool_)

        data: DatasetDict = {
            "obs": self._stack_column(table, obs_col),
            "act": self._stack_column(table, act_col),
            "rew": self._scalar_column(table, rew_col, np.float32),
            "done": done,
            "obs_next": self._stack_column(table, obs_next_col),
            "terminated": self._scalar_column(table, terminated_col, np.bool_),
            "truncated": self._scalar_column(table, truncated_col, np.bool_),
        }

        return data

    def load_obs_act(self) -> DatasetDict:
        obs_col = self._required_column_name("obs")
        act_col = self._required_column_name("act")

        column_names = list(dict.fromkeys([obs_col, act_col]))
        table = pq.read_table(self.path, columns=column_names, use_threads=True)
        return {
            "obs": self._stack_column(table, obs_col),
            "act": self._stack_column(table, act_col),
        }
