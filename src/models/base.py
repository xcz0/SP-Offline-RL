"""Model factory interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.types import ModelBundle


class ModelFactory(ABC):
    """Factory interface for constructing actor/critic modules."""

    @abstractmethod
    def build_actor(self, cfg: Any, space_info: Any, device: str):
        """Build actor network."""

    @abstractmethod
    def build_critic(self, cfg: Any, space_info: Any, device: str):
        """Build critic networks (for actor-critic algorithms)."""

    def build(self, cfg: Any, space_info: Any, device: str) -> ModelBundle:
        actor = self.build_actor(cfg, space_info, device)
        critic1, critic2 = self.build_critic(cfg, space_info, device)
        return ModelBundle(actor=actor, critic1=critic1, critic2=critic2)
