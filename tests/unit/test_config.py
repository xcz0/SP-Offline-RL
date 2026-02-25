from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def test_default_config_loads_key_contracts() -> None:
    cfg = OmegaConf.load(CONFIG_ROOT / "config.yaml")
    assert cfg.seed == 0
    assert cfg.device == "cuda:0"
    assert cfg.train.epoch == 200
    assert cfg.eval_mode == "auto"
    assert cfg.compile.enabled is True
    assert cfg.perf.eval_workers == "auto"

    data_cfg = OmegaConf.load(CONFIG_ROOT / "data" / "parquet_sp.yaml")
    assert data_cfg.columns.terminated == "terminated"
    assert data_cfg.columns.truncated == "truncated"

    sim_eval_cfg = OmegaConf.load(CONFIG_ROOT / "sim_eval" / "default.yaml")
    assert sim_eval_cfg.eval_every_n_epoch == 5
    assert sim_eval_cfg.warmup_mode == "fifth"


def test_validation_config_for_user3_single_card_contract() -> None:
    val_cfg = OmegaConf.load(CONFIG_ROOT / "config_val_user3.yaml")
    assert val_cfg.train.epoch == 2
    assert val_cfg.device == "cuda:0"
    assert val_cfg.compile.enabled is True
    assert val_cfg.perf.eval_workers == "auto"

    sim_eval_cfg = OmegaConf.load(CONFIG_ROOT / "sim_eval" / "user3_card1.yaml")
    assert list(sim_eval_cfg.user_ids) == [3]
    assert sim_eval_cfg.cards_per_user == 1
