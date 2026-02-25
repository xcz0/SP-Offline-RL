from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest

from src.core.exceptions import ConfigurationError
from src.data.spec_registry import resolve_algo_data_spec
from src.runners.data_builder import load_prepared_dataset
from tests.factories.config_factory import OBS_ACT_COLUMNS, build_prepare_cfg
from tests.factories.dataset_factory import write_offline_parquet, write_obs_act_parquet


def test_resolve_algo_data_spec_defaults() -> None:
    assert resolve_algo_data_spec("bc_il").fields == ("obs", "act")
    assert resolve_algo_data_spec("td3_bc").fields == (
        "obs",
        "act",
        "rew",
        "done",
        "obs_next",
        "terminated",
        "truncated",
    )


def test_load_prepared_dataset_uses_registered_bc_spec(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bc_obs_act.parquet"
    write_obs_act_parquet(dataset_path, n=8, seed=10)
    cfg = build_prepare_cfg(dataset_path, data_columns=OBS_ACT_COLUMNS)
    env = gym.make("SPRLTestEnv-v0")
    try:
        prepared = load_prepared_dataset(cfg, env=env, algo_name="bc_il")
    finally:
        env.close()

    assert set(prepared.arrays.keys()) == {"obs", "act"}
    assert prepared.arrays["obs"].shape[0] == 8
    assert prepared.meta["spec"] == "bc_il"


def test_load_prepared_dataset_supports_custom_field_subset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "offline.parquet"
    write_offline_parquet(dataset_path, n=12, seed=11)
    cfg = build_prepare_cfg(dataset_path)
    cfg.data.fields = ["obs", "act", "rew"]

    prepared = load_prepared_dataset(cfg, env=None, algo_name="custom_algo")
    assert set(prepared.arrays.keys()) == {"obs", "act", "rew"}
    assert prepared.arrays["rew"].ndim == 1
    assert prepared.meta["requested_fields"] == ["obs", "act", "rew"]


def test_load_prepared_dataset_unknown_algo_requires_registered_spec(tmp_path: Path) -> None:
    dataset_path = tmp_path / "offline.parquet"
    write_offline_parquet(dataset_path, n=4, seed=12)
    cfg = build_prepare_cfg(dataset_path)

    with pytest.raises(ConfigurationError, match="No data spec registered"):
        load_prepared_dataset(cfg, env=None, algo_name="new_algo")

