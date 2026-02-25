from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.core.exceptions import ConfigurationError
from src.runners.common import build_sim_eval_cfg


def test_build_sim_eval_cfg_injects_perf_workers_and_obs_stats() -> None:
    cfg = OmegaConf.create(
        {
            "sim_eval": {
                "enabled": True,
                "data_dir": "data",
            },
            "perf": {
                "eval_workers": 4,
            },
        }
    )

    sim_cfg = build_sim_eval_cfg(
        cfg,
        obs_norm_mean=[0.1, 0.2],
        obs_norm_var=[1.0, 1.1],
    )
    assert sim_cfg["eval_workers"] == 4
    assert sim_cfg["obs_norm_mean"] == [0.1, 0.2]
    assert sim_cfg["obs_norm_var"] == [1.0, 1.1]


def test_build_sim_eval_cfg_resolves_auto_workers_from_cpu_count(monkeypatch) -> None:
    monkeypatch.setattr("src.runners.common.os.cpu_count", lambda: 12)
    cfg = OmegaConf.create(
        {
            "sim_eval": {
                "enabled": True,
                "data_dir": "data",
            },
            "perf": {
                "eval_workers": "auto",
            },
        }
    )

    sim_cfg = build_sim_eval_cfg(cfg)
    assert sim_cfg["eval_workers"] == 12


def test_build_sim_eval_cfg_rejects_invalid_workers_value() -> None:
    cfg = OmegaConf.create(
        {
            "sim_eval": {
                "enabled": True,
                "data_dir": "data",
            },
            "perf": {
                "eval_workers": "many",
            },
        }
    )

    with pytest.raises(ConfigurationError, match="perf.eval_workers"):
        build_sim_eval_cfg(cfg)
