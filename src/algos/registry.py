"""Algorithm registry and access helpers."""

from __future__ import annotations

from src.algos.base import AlgoFactory
from src.algos.bc_il import BCILFactory
from src.algos.td3_bc import TD3BCFactory
from src.core.registry import Registry

ALGO_REGISTRY: Registry[AlgoFactory] = Registry(namespace="algo")


def register_default_algos() -> None:
    if "td3_bc" not in ALGO_REGISTRY.keys():
        ALGO_REGISTRY.register("td3_bc", TD3BCFactory())
    if "bc_il" not in ALGO_REGISTRY.keys():
        ALGO_REGISTRY.register("bc_il", BCILFactory())


def get_algo_factory(name: str) -> AlgoFactory:
    register_default_algos()
    return ALGO_REGISTRY.get(name)
