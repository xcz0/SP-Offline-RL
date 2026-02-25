from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def test_default_config_load_and_override() -> None:
    cfg = OmegaConf.load(CONFIG_ROOT / "config.yaml")
    assert cfg.seed == 0
    assert cfg.device == "cuda:0"
    assert cfg.train.epoch == 200
    assert cfg.eval_mode == "auto"
    assert cfg.compile.enabled is True
    assert cfg.compile.mode == "reduce-overhead"
    assert cfg.compile.backend == "inductor"
    assert cfg.compile.dynamic is False
    assert cfg.compile.fullgraph is False
    assert cfg.perf.eval_workers == "auto"

    data_cfg = OmegaConf.load(CONFIG_ROOT / "data" / "parquet_sp.yaml")
    assert data_cfg.columns.terminated == "terminated"
    assert data_cfg.columns.truncated == "truncated"

    bc_train_data_cfg = OmegaConf.load(CONFIG_ROOT / "data" / "parquet_sp_bc_train.yaml")
    bc_val_data_cfg = OmegaConf.load(CONFIG_ROOT / "data" / "parquet_sp_bc_val.yaml")
    assert set(bc_train_data_cfg.columns.keys()) == {"obs", "act"}
    assert set(bc_val_data_cfg.columns.keys()) == {"obs", "act"}

    sim_eval_cfg = OmegaConf.load(CONFIG_ROOT / "sim_eval" / "default.yaml")
    assert sim_eval_cfg.eval_every_n_epoch == 5
    assert sim_eval_cfg.warmup_mode == "fifth"

    cfg.train.epoch = 2
    cfg.algo = {"name": "bc_il", "lr": 1e-4}
    assert cfg.train.epoch == 2
    assert cfg.algo.name == "bc_il"


def test_validation_config_for_user3_single_card() -> None:
    val_cfg = OmegaConf.load(CONFIG_ROOT / "config_val_user3.yaml")
    assert val_cfg.train.epoch == 2
    assert val_cfg.device == "cuda:0"
    assert val_cfg.compile.enabled is True
    assert val_cfg.compile.mode == "reduce-overhead"
    assert val_cfg.perf.eval_workers == "auto"

    sim_eval_cfg = OmegaConf.load(CONFIG_ROOT / "sim_eval" / "user3_card1.yaml")
    assert list(sim_eval_cfg.user_ids) == [3]
    assert sim_eval_cfg.cards_per_user == 1
