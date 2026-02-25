from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.runners import runtime_builder
from src.runners.common import prepare_bc_sim_components, prepare_gym_components
from tests.factories.config_factory import (
    BC_IL_ALGO_CFG,
    TD3_BC_ALGO_CFG,
    build_prepare_cfg,
    mlp_actor_cfg,
    mlp_actor_critic_cfg,
)


@dataclass(slots=True)
class _DummyAlgorithm:
    policy: object


@pytest.fixture(autouse=True)
def _mock_algorithm_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    def _dummy_builder(*_args: Any, **_kwargs: Any) -> _DummyAlgorithm:
        return _DummyAlgorithm(policy=object())

    monkeypatch.setattr(runtime_builder, "build_algorithm", _dummy_builder)
    monkeypatch.setattr(runtime_builder, "build_algorithm_from_space_info", _dummy_builder)


@pytest.mark.parametrize(
    ("model_cfg", "algo_cfg", "seed"),
    [
        pytest.param(mlp_actor_cfg(), BC_IL_ALGO_CFG, 21, id="bc_il"),
        pytest.param(mlp_actor_critic_cfg(), TD3_BC_ALGO_CFG, 22, id="td3_bc"),
    ],
)
def test_prepare_gym_components(
    write_offline_dataset: Any,
    model_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
    seed: int,
) -> None:
    dataset_path = write_offline_dataset(n=16, seed=seed)
    cfg = build_prepare_cfg(dataset_path)
    cfg.model = dict(model_cfg)
    cfg.algo = dict(algo_cfg)

    components = prepare_gym_components(
        cfg,
        "cpu",
        seed=0,
        include_train_buffer=True,
    )
    try:
        assert components.train_buffer is not None
        assert hasattr(components.algorithm, "policy")
    finally:
        components.test_envs.close()
        components.env.close()


def test_prepare_bc_sim_components_with_obs_norm(
    write_obs_act_dataset: Any,
    obs_act_columns: dict[str, str],
) -> None:
    dataset_path = write_obs_act_dataset(n=16, seed=23)
    cfg = build_prepare_cfg(
        dataset_path,
        data_columns=obs_act_columns,
        obs_norm=True,
    )
    cfg.model = mlp_actor_cfg()
    cfg.algo = dict(BC_IL_ALGO_CFG)

    components = prepare_bc_sim_components(
        cfg,
        "cpu",
        seed=0,
        include_train_buffer=True,
    )
    assert components.train_buffer is not None
    assert components.action_high > components.action_low
    assert components.obs_norm_mean is not None
    assert components.obs_norm_var is not None
