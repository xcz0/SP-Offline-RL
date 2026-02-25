from __future__ import annotations

from omegaconf import OmegaConf

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
