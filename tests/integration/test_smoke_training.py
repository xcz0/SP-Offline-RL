from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.runners.evaluator import run_evaluation
from src.runners.trainer import run_offline_training


def _write_parquet_dataset(path: Path, n: int = 64) -> None:
    obs = np.random.randn(n, 3).astype(np.float32)
    act = np.random.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    rew = np.random.randn(n).astype(np.float32)
    done = np.zeros(n, dtype=np.bool_)
    obs_next = np.random.randn(n, 3).astype(np.float32)

    frame = pd.DataFrame(
        {
            "obs": [x.tolist() for x in obs],
            "act": [x.tolist() for x in act],
            "rew": rew,
            "done": done,
            "obs_next": [x.tolist() for x in obs_next],
        }
    )
    frame.to_parquet(path, index=False)


def _base_cfg(tmp_path: Path, dataset_path: Path):
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "resume_path": None,
            "resume_id": None,
            "watch": False,
            "buffer_size": None,
            "checkpoint_path": None,
            "train": {"epoch": 1, "epoch_num_steps": 1, "batch_size": 8},
            "paths": {
                "logdir": str(tmp_path / "log"),
                "save_metrics_filename": "final_metrics.json",
            },
            "env": {"task": "Pendulum-v1", "num_test_envs": 1, "render": 0.0},
            "data": {
                "adapter": "parquet",
                "path": str(dataset_path),
                "obs_norm": False,
                "columns": {
                    "obs": "obs",
                    "act": "act",
                    "rew": "rew",
                    "done": "done",
                    "obs_next": "obs_next",
                    "terminated": None,
                    "truncated": None,
                },
            },
            "logger": {"type": "tensorboard", "wandb_project": "test"},
        }
    )


@pytest.mark.integration
def test_bc_il_smoke_train(tmp_path: Path) -> None:
    dataset_path = tmp_path / "offline.parquet"
    _write_parquet_dataset(dataset_path)

    cfg = _base_cfg(tmp_path, dataset_path)
    cfg.model = {"name": "mlp_actor", "hidden_sizes": [32, 32]}
    cfg.algo = {"name": "bc_il", "lr": 1e-4}

    result = run_offline_training(cfg)
    assert "final_metrics" in result
    assert Path(result["final_metrics"]["log_path"]).exists()


@pytest.mark.integration
def test_td3_bc_smoke_train(tmp_path: Path) -> None:
    dataset_path = tmp_path / "offline.parquet"
    _write_parquet_dataset(dataset_path)

    cfg = _base_cfg(tmp_path, dataset_path)
    cfg.model = {"name": "mlp_actor_critic", "hidden_sizes": [32, 32]}
    cfg.algo = {
        "name": "td3_bc",
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha": 2.5,
        "exploration_noise": 0.1,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "update_actor_freq": 2,
        "tau": 0.005,
        "gamma": 0.99,
        "n_step": 1,
    }

    result = run_offline_training(cfg)
    assert "final_metrics" in result
    assert Path(result["final_metrics"]["log_path"]).exists()


@pytest.mark.integration
def test_train_then_eval_checkpoint(tmp_path: Path) -> None:
    dataset_path = tmp_path / "offline.parquet"
    _write_parquet_dataset(dataset_path)

    cfg = _base_cfg(tmp_path, dataset_path)
    cfg.model = {"name": "mlp_actor", "hidden_sizes": [32, 32]}
    cfg.algo = {"name": "bc_il", "lr": 1e-4}

    train_result = run_offline_training(cfg)
    log_path = Path(train_result["final_metrics"]["log_path"])
    checkpoint = log_path / "policy.pth"
    assert checkpoint.exists()

    eval_result = run_evaluation(cfg, str(checkpoint))
    assert "test_reward_mean" in eval_result
    assert "test_reward_std" in eval_result
