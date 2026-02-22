"""Model registry and access helpers."""

from __future__ import annotations

from src.core.registry import Registry
from src.models.base import ModelFactory
from src.models.mlp import MLPActorCriticFactory, MLPActorOnlyFactory

MODEL_REGISTRY: Registry[ModelFactory] = Registry(namespace="model")


def register_default_models() -> None:
    if "mlp_actor_critic" not in MODEL_REGISTRY.keys():
        MODEL_REGISTRY.register("mlp_actor_critic", MLPActorCriticFactory())
    if "mlp_actor" not in MODEL_REGISTRY.keys():
        MODEL_REGISTRY.register("mlp_actor", MLPActorOnlyFactory())


def get_model_factory(name: str) -> ModelFactory:
    register_default_models()
    return MODEL_REGISTRY.get(name)
