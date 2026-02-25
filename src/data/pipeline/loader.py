"""Shared dataset loading helpers driven by DataSpec."""

from __future__ import annotations

from typing import Any

from src.core.types import DatasetDict
from src.data.dataset_adapter import build_dataset_adapter
from src.data.pipeline.contracts import DataSpec


def load_raw_arrays(
    *,
    data_cfg: Any,
    spec: DataSpec,
) -> DatasetDict:
    """Load raw arrays for the requested canonical fields."""

    adapter = build_dataset_adapter(data_cfg)
    return adapter.load_prepared(fields=spec.fields)

