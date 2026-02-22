from __future__ import annotations

import pytest

from src.algos.registry import get_algo_factory
from src.core.exceptions import RegistryError
from src.models.registry import get_model_factory


def test_model_registry_get_known_and_unknown() -> None:
    assert get_model_factory("mlp_actor") is not None
    with pytest.raises(RegistryError):
        get_model_factory("unknown_model")


def test_algo_registry_get_known_and_unknown() -> None:
    assert get_algo_factory("bc_il") is not None
    with pytest.raises(RegistryError):
        get_algo_factory("unknown_algo")
