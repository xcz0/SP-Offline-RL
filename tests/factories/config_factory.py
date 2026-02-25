from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

FULL_PARQUET_COLUMNS: dict[str, str] = {
    "obs": "obs",
    "act": "act",
    "rew": "rew",
    "done": "done",
    "obs_next": "obs_next",
    "terminated": "terminated",
    "truncated": "truncated",
}
OBS_ACT_COLUMNS: dict[str, str] = {"obs": "obs", "act": "act"}

BC_IL_ALGO_CFG: dict[str, Any] = {"name": "bc_il", "lr": 1e-4}
TD3_BC_ALGO_CFG: dict[str, Any] = {
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


def mlp_actor_cfg(*, hidden_sizes: tuple[int, int] = (16, 16)) -> dict[str, Any]:
    return {"name": "mlp_actor", "hidden_sizes": list(hidden_sizes)}


def mlp_actor_critic_cfg(*, hidden_sizes: tuple[int, int] = (16, 16)) -> dict[str, Any]:
    return {"name": "mlp_actor_critic", "hidden_sizes": list(hidden_sizes)}


def build_train_cfg(
    tmp_path: Path,
    dataset_path: Path,
    *,
    data_columns: dict[str, str] | None = None,
    obs_norm: bool = False,
) -> DictConfig:
    columns = FULL_PARQUET_COLUMNS if data_columns is None else data_columns
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "resume_path": None,
            "resume_id": None,
            "watch": False,
            "buffer_size": None,
            "checkpoint_path": None,
            # Keep smoke tests fast: exercise trainer lifecycle with zero update steps.
            "train": {"epoch": 1, "epoch_num_steps": 0, "batch_size": 4},
            "paths": {
                "logdir": str(tmp_path / "log"),
                "save_metrics_filename": "final_metrics.json",
            },
            "env": {"task": "SPRLTestEnv-v0", "num_test_envs": 1, "render": 0.0},
            "data": {
                "adapter": "parquet",
                "path": str(dataset_path),
                "obs_norm": bool(obs_norm),
                "columns": dict(columns),
            },
            "logger": {"type": "tensorboard", "wandb_project": "test"},
        }
    )


def build_prepare_cfg(
    dataset_path: Path,
    *,
    data_columns: dict[str, str] | None = None,
    obs_norm: bool = False,
) -> DictConfig:
    columns = FULL_PARQUET_COLUMNS if data_columns is None else data_columns
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "buffer_size": None,
            "env": {"task": "SPRLTestEnv-v0", "num_test_envs": 1, "render": 0.0},
            "data": {
                "adapter": "parquet",
                "path": str(dataset_path),
                "obs_norm": bool(obs_norm),
                "columns": dict(columns),
            },
        }
    )
