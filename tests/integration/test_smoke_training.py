from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.runners import evaluate, train
from tests.factories.config_factory import (
    BC_IL_ALGO_CFG,
    TD3_BC_ALGO_CFG,
    mlp_actor_cfg,
    mlp_actor_critic_cfg,
)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model_cfg", "algo_cfg"),
    [
        pytest.param(mlp_actor_cfg(), BC_IL_ALGO_CFG, id="bc_il"),
        pytest.param(mlp_actor_critic_cfg(), TD3_BC_ALGO_CFG, id="td3_bc"),
    ],
)
def test_smoke_train(
    write_offline_dataset: Any,
    make_train_cfg: Any,
    model_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
) -> None:
    dataset_path = write_offline_dataset(n=16, seed=11)
    cfg = make_train_cfg(dataset_path)
    cfg.model = dict(model_cfg)
    cfg.algo = dict(algo_cfg)

    result = train(cfg)
    assert result.mode == "train"
    assert Path(result.log_path).exists()


@pytest.mark.integration
def test_train_then_eval_checkpoint(
    write_offline_dataset: Any,
    make_train_cfg: Any,
) -> None:
    dataset_path = write_offline_dataset(n=16, seed=12)
    cfg = make_train_cfg(dataset_path)
    cfg.model = mlp_actor_cfg()
    cfg.algo = dict(BC_IL_ALGO_CFG)

    train_result = train(cfg)
    assert train_result.checkpoint_path is not None
    checkpoint = Path(train_result.checkpoint_path)
    assert checkpoint.exists()

    eval_result = evaluate(cfg, str(checkpoint))
    assert np.isfinite(eval_result.evaluation.test_reward_mean)
    assert np.isfinite(eval_result.evaluation.test_reward_std)


@pytest.mark.integration
def test_bc_il_smoke_train_obs_act_only_dataset(
    write_obs_act_dataset: Any,
    make_train_cfg: Any,
    obs_act_columns: dict[str, str],
) -> None:
    dataset_path = write_obs_act_dataset(n=16, seed=13)
    cfg = make_train_cfg(dataset_path, columns=obs_act_columns)
    cfg.model = mlp_actor_cfg()
    cfg.algo = dict(BC_IL_ALGO_CFG)

    result = train(cfg)
    assert result.mode == "train"
    assert Path(result.log_path).exists()


@pytest.mark.integration
def test_bc_il_train_then_eval_obs_norm_obs_act_only_dataset(
    write_obs_act_dataset: Any,
    make_train_cfg: Any,
    obs_act_columns: dict[str, str],
) -> None:
    dataset_path = write_obs_act_dataset(n=16, seed=14)
    cfg = make_train_cfg(dataset_path, columns=obs_act_columns, obs_norm=True)
    cfg.model = mlp_actor_cfg()
    cfg.algo = dict(BC_IL_ALGO_CFG)

    train_result = train(cfg)
    assert train_result.checkpoint_path is not None
    checkpoint = Path(train_result.checkpoint_path)
    assert checkpoint.exists()

    eval_result = evaluate(cfg, str(checkpoint))
    assert np.isfinite(eval_result.evaluation.test_reward_mean)
    assert np.isfinite(eval_result.evaluation.test_reward_std)
