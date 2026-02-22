"""Shared type definitions and lightweight data containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

DatasetDict = dict[str, np.ndarray]


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
