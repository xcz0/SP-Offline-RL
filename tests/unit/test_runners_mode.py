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


@pytest.mark.parametrize(
    ("watch", "sim_enabled", "algo_name", "expected_mode", "expected_use_sim"),
    [
        pytest.param(True, True, "bc_il", "watch", True, id="watch-bc-sim-enabled"),
        pytest.param(False, True, "td3_bc", "train", False, id="train-td3-sim-disabled"),
    ],
)
def test_resolve_train_mode_cases(
    watch: bool,
    sim_enabled: bool,
    algo_name: str,
    expected_mode: str,
    expected_use_sim: bool,
) -> None:
    cfg = _base_cfg()
    cfg.watch = watch
    cfg.sim_eval.enabled = sim_enabled

    run_mode, use_sim = resolve_train_mode(cfg, algo_name)
    assert run_mode == expected_mode
    assert use_sim is expected_use_sim


@pytest.mark.parametrize(
    ("sim_enabled", "algo_name", "expected_mode"),
    [
        pytest.param(True, "bc_il", "sim", id="auto-prefers-sim-for-bc"),
        pytest.param(True, "td3_bc", "gym", id="auto-defaults-gym-for-td3"),
    ],
)
def test_resolve_eval_mode_auto_cases(
    sim_enabled: bool,
    algo_name: str,
    expected_mode: str,
) -> None:
    cfg = _base_cfg()
    cfg.sim_eval.enabled = sim_enabled

    assert resolve_eval_mode(cfg, algo_name) == expected_mode


@pytest.mark.parametrize(
    ("eval_mode", "expected_mode", "error_match"),
    [
        pytest.param("replay", "replay", None, id="explicit-replay"),
        pytest.param("invalid", None, "Unsupported eval_mode", id="invalid-raises"),
    ],
)
def test_resolve_eval_mode_explicit_and_invalid_cases(
    eval_mode: str,
    expected_mode: str | None,
    error_match: str | None,
) -> None:
    cfg = _base_cfg()
    cfg.eval_mode = eval_mode
    if error_match is None:
        assert resolve_eval_mode(cfg, "bc_il") == expected_mode
    else:
        with pytest.raises(ConfigurationError, match=error_match):
            resolve_eval_mode(cfg, "bc_il")
