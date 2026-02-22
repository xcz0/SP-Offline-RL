"""Dataset adapter interface for offline RL data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.types import DatasetDict


class OfflineDatasetAdapter(ABC):
    """Adapter contract: load raw dataset and return canonical fields."""

    @abstractmethod
    def load(self) -> DatasetDict:
        """Load offline dataset into canonical numpy arrays."""


def build_dataset_adapter(cfg: Any) -> OfflineDatasetAdapter:
    """Build dataset adapter based on config group."""

    adapter_type = str(cfg.adapter).lower()
    if adapter_type == "parquet":
        from src.data.parquet_adapter import ParquetOfflineDatasetAdapter

        return ParquetOfflineDatasetAdapter(
            path=str(cfg.path),
            columns=dict(cfg.columns),
        )
    raise ValueError(
        f"Unsupported data adapter type: {cfg.adapter}. "
        "Only 'parquet' is supported."
    )
