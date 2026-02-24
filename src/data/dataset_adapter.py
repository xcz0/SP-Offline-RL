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

    def load_obs_act(self) -> DatasetDict:
        """Load minimal behavior-cloning fields."""

        data = self.load()
        return {
            "obs": data["obs"],
            "act": data["act"],
        }


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
