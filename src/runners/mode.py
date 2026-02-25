"""Mode resolution helpers for train/eval runners."""

from __future__ import annotations

from typing import Literal

from omegaconf import DictConfig

from src.core.exceptions import ConfigurationError

TrainMode = Literal["train", "watch"]
EvalMode = Literal["gym", "sim", "replay"]


def is_bc_sim_eval_enabled(cfg: DictConfig, algo_name: str) -> bool:
    """Return whether bc_il simulator-eval flow is enabled."""

    if algo_name != "bc_il":
        return False
    sim_eval_cfg = cfg.get("sim_eval")
    if sim_eval_cfg is None:
        return False
    return bool(sim_eval_cfg.get("enabled", False))


def resolve_train_mode(cfg: DictConfig, algo_name: str) -> tuple[TrainMode, bool]:
    """Resolve train/watch mode and simulator-eval usage."""

    run_mode: TrainMode = "watch" if bool(cfg.watch) else "train"
    return run_mode, is_bc_sim_eval_enabled(cfg, algo_name)


def resolve_eval_mode(cfg: DictConfig, algo_name: str) -> EvalMode:
    """Resolve eval mode from explicit config or auto rules."""

    mode = str(cfg.get("eval_mode", "auto")).lower()
    if mode == "auto":
        return "sim" if is_bc_sim_eval_enabled(cfg, algo_name) else "gym"
    if mode in {"gym", "sim", "replay"}:
        return mode  # type: ignore[return-value]
    raise ConfigurationError(
        f"Unsupported eval_mode '{mode}'. Expected one of: auto, gym, sim, replay."
    )
