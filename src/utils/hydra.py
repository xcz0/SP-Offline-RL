"""Hydra/OmegaConf utility helpers."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def resolve_config(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def as_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
