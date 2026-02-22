"""Algorithm factory interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.types import ModelBundle


class AlgoFactory(ABC):
    """Factory for creating Tianshou algorithm instances."""

    @abstractmethod
    def build(self, cfg: Any, env: Any, model_bundle: ModelBundle, device: str):
        """Build a configured algorithm object."""
