"""Algorithm-to-data-spec registry."""

from __future__ import annotations

from collections.abc import Sequence

from src.core.exceptions import ConfigurationError
from src.data.pipeline.contracts import (
    BC_DATA_FIELDS,
    RL_DATA_FIELDS,
    DataSpec,
    make_data_spec,
)

_SPEC_REGISTRY: dict[str, DataSpec] = {
    "bc_il": make_data_spec(name="bc_il", fields=BC_DATA_FIELDS),
    "td3_bc": make_data_spec(name="td3_bc", fields=RL_DATA_FIELDS),
}


def register_algo_data_spec(algo_name: str, spec: DataSpec) -> None:
    """Register data spec for a new algorithm."""

    normalized = str(algo_name).strip().lower()
    if not normalized:
        raise ValueError("algo_name must not be empty.")
    _SPEC_REGISTRY[normalized] = spec


def resolve_algo_data_spec(algo_name: str) -> DataSpec:
    """Resolve data spec by algorithm name."""

    normalized = str(algo_name).strip().lower()
    spec = _SPEC_REGISTRY.get(normalized)
    if spec is None:
        known = ", ".join(sorted(_SPEC_REGISTRY))
        raise ConfigurationError(
            f"No data spec registered for algorithm '{algo_name}'. "
            f"Known algorithms: {known}. "
            "Register one in src/data/spec_registry.py or set cfg.data.fields."
        )
    return spec


def resolve_custom_data_spec(fields: Sequence[str], *, name: str = "custom") -> DataSpec:
    """Build ad-hoc spec from explicit canonical field list."""

    return make_data_spec(name=name, fields=fields)

