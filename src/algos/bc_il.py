"""Behavior Cloning (Offline IL) algorithm factory."""

from __future__ import annotations

from typing import Any

from tianshou.algorithm.imitation.imitation_base import (
    ImitationPolicy,
    OfflineImitationLearning,
)
from tianshou.algorithm.optim import AdamOptimizerFactory

from src.algos.base import AlgoFactory
from src.core.types import ModelBundle


class BCILFactory(AlgoFactory):
    """Builds offline imitation learning with deterministic actor."""

    def build(self, cfg: Any, env: Any, model_bundle: ModelBundle, device: str):
        policy = ImitationPolicy(
            actor=model_bundle.actor,
            action_space=env.action_space,
            action_scaling=True,
            action_bound_method="clip",
        )

        algo = OfflineImitationLearning(
            policy=policy,
            optim=AdamOptimizerFactory(lr=float(cfg.lr)),
        )
        return algo
