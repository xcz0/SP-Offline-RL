"""Parquet dataset adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.types import DatasetDict
from src.data.dataset_adapter import OfflineDatasetAdapter, validate_dataset_fields


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
        out = np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=dtype)
        return np.ascontiguousarray(out)

    def _stack_column(self, table: pa.Table, name: str) -> np.ndarray:
        column = table[name].combine_chunks()
        if pa.types.is_fixed_size_list(column.type):
            values = np.asarray(
                column.values.to_numpy(zero_copy_only=False), dtype=np.float32
            )
            reshaped = values.reshape(len(column), int(column.type.list_size))
            return np.ascontiguousarray(reshaped)

        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            offsets = np.asarray(column.offsets.to_numpy(zero_copy_only=False))
            row_lengths = np.diff(offsets)
            if row_lengths.size == 0:
                return np.empty((len(column), 0), dtype=np.float32)
            if np.all(row_lengths == row_lengths[0]):
                values = np.asarray(
                    column.values.to_numpy(zero_copy_only=False), dtype=np.float32
                )
                reshaped = values.reshape(len(column), int(row_lengths[0]))
                return np.ascontiguousarray(reshaped)
            raise ValueError(
                f"Column '{name}' must contain equal-length vectors, got row lengths "
                f"{row_lengths.min()}..{row_lengths.max()}."
            )

        raise ValueError(
            f"Column '{name}' must be fixed_size_list or list type, got {column.type}."
        )

    def _load_field(self, table: pa.Table, field: str, column_name: str) -> np.ndarray:
        if field in {"obs", "act", "obs_next"}:
            return self._stack_column(table, column_name)
        if field in {"done", "terminated", "truncated"}:
            return self._scalar_column(table, column_name, np.bool_)
        if field == "rew":
            return self._scalar_column(table, column_name, np.float32)
        raise ValueError(f"Unsupported canonical field: {field}.")

    def load_prepared(self, fields: Sequence[str]) -> DatasetDict:
        canonical_fields = validate_dataset_fields(fields)
        mapped_columns = {
            field: self._required_column_name(field)
            for field in canonical_fields
        }
        column_names = list(dict.fromkeys(mapped_columns.values()))
        table = pq.read_table(self.path, columns=column_names, use_threads=True).combine_chunks()
        return {
            field: self._load_field(table, field, column_name)
            for field, column_name in mapped_columns.items()
        }
