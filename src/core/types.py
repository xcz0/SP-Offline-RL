"""Shared type definitions and lightweight data containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

DatasetDict = dict[str, np.ndarray]


@dataclass(slots=True)
class PreparedDataset:
    """Typed container for preprocessed offline arrays and normalization stats."""

    arrays: DatasetDict
    obs_norm_mean: np.ndarray | None = None
    obs_norm_var: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.arrays = {name: np.ascontiguousarray(array) for name, array in self.arrays.items()}
        if self.obs_norm_mean is not None:
            self.obs_norm_mean = np.asarray(self.obs_norm_mean, dtype=np.float32)
        if self.obs_norm_var is not None:
            self.obs_norm_var = np.asarray(self.obs_norm_var, dtype=np.float32)

    @property
    def size(self) -> int:
        if not self.arrays:
            return 0
        first = next(iter(self.arrays.values()))
        return int(first.shape[0])

    def to_dict(self) -> DatasetDict:
        return dict(self.arrays)


@dataclass(slots=True)
class ModelBundle:
    """Container for networks created by a model factory."""

    actor: torch.nn.Module
    critic1: torch.nn.Module | None = None
    critic2: torch.nn.Module | None = None


@dataclass(slots=True)
class RunContext:
    """Runtime metadata for a train/eval execution."""

    log_path: str
    run_name: str
    resolved_config: dict[str, Any]
