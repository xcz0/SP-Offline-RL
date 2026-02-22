from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def test_default_config_load_and_override() -> None:
    cfg = OmegaConf.load(CONFIG_ROOT / "config.yaml")
    assert cfg.seed == 0
    assert cfg.train.epoch == 200

    cfg.train.epoch = 2
    cfg.algo = {"name": "bc_il", "lr": 1e-4}
    assert cfg.train.epoch == 2
    assert cfg.algo.name == "bc_il"
