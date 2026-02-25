from __future__ import annotations

from collections.abc import Callable

import pytest

from src.algos.registry import get_algo_factory
from src.core.exceptions import RegistryError
from src.models.registry import get_model_factory


@pytest.mark.parametrize(
    ("getter", "known_name", "unknown_name"),
    [
        pytest.param(get_model_factory, "mlp_actor", "unknown_model", id="model"),
        pytest.param(get_algo_factory, "bc_il", "unknown_algo", id="algo"),
    ],
)
def test_registry_get_known_and_unknown(
    getter: Callable[[str], object],
    known_name: str,
    unknown_name: str,
) -> None:
    assert getter(known_name) is not None
    with pytest.raises(RegistryError):
        getter(unknown_name)
