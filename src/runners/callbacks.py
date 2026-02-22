"""Training callback builders."""

from __future__ import annotations

from pathlib import Path

import torch


def build_save_best_fn(log_path: str):
    target = Path(log_path) / "policy.pth"

    def save_best_fn(policy) -> None:
        torch.save(policy.state_dict(), target)

    return save_best_fn
