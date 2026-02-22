"""Random seeding helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_vector_env(env: Any, seed: int) -> None:
    """Seed vectorized environments with a strict interface contract."""

    env.seed(seed)
