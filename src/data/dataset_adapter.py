"""Dataset adapter interface for offline RL data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from src.core.types import DatasetDict
from src.data.pipeline import contracts as _contracts

BC_DATA_FIELDS = _contracts.BC_DATA_FIELDS
RL_DATA_FIELDS = _contracts.RL_DATA_FIELDS
SUPPORTED_DATA_FIELDS = _contracts.SUPPORTED_DATA_FIELDS
validate_dataset_fields = _contracts.validate_dataset_fields


class OfflineDatasetAdapter(ABC):
    """Adapter contract: load selected canonical fields as numpy arrays."""

    @abstractmethod
    def load_prepared(self, fields: Sequence[str]) -> DatasetDict:
        """Load selected canonical fields into numpy arrays."""

def build_dataset_adapter(cfg: Any) -> OfflineDatasetAdapter:
    """Build dataset adapter based on config group."""

    adapter_type = str(cfg.adapter).lower()
    if adapter_type == "parquet":
        from src.data.parquet_adapter import ParquetOfflineDatasetAdapter

        raw_columns = dict(cfg.columns)
        columns = {
            str(key): (str(value) if value is not None else "")
            for key, value in raw_columns.items()
        }
        return ParquetOfflineDatasetAdapter(
            path=str(cfg.path),
            columns=columns,
        )
    raise ValueError(
        f"Unsupported data adapter type: {cfg.adapter}. "
        "Only 'parquet' is supported."
    )
