from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.core.exceptions import ConfigurationError
from src.runners.mode import resolve_eval_mode, resolve_train_mode


def _base_cfg() -> object:
    return OmegaConf.create(
        {
            "watch": False,
            "eval_mode": "auto",
            "sim_eval": {"enabled": False},
        }
    )


def test_resolve_train_mode_watch_and_sim() -> None:
    cfg = _base_cfg()
    cfg.watch = True
    cfg.sim_eval.enabled = True

    run_mode, use_sim = resolve_train_mode(cfg, "bc_il")
    assert run_mode == "watch"
    assert use_sim is True


def test_resolve_train_mode_td3_bc_disables_sim() -> None:
    cfg = _base_cfg()
    cfg.sim_eval.enabled = True

    run_mode, use_sim = resolve_train_mode(cfg, "td3_bc")
    assert run_mode == "train"
    assert use_sim is False


def test_resolve_eval_mode_auto_prefers_sim_for_bc() -> None:
    cfg = _base_cfg()
    cfg.sim_eval.enabled = True
    assert resolve_eval_mode(cfg, "bc_il") == "sim"


def test_resolve_eval_mode_auto_defaults_gym_for_non_bc() -> None:
    cfg = _base_cfg()
    cfg.sim_eval.enabled = True
    assert resolve_eval_mode(cfg, "td3_bc") == "gym"


def test_resolve_eval_mode_explicit_replay() -> None:
    cfg = _base_cfg()
    cfg.eval_mode = "replay"
    assert resolve_eval_mode(cfg, "bc_il") == "replay"


def test_resolve_eval_mode_invalid_raises() -> None:
    cfg = _base_cfg()
    cfg.eval_mode = "invalid"
    with pytest.raises(ConfigurationError):
        resolve_eval_mode(cfg, "bc_il")
