"""torch.compile helpers for runtime model optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from src.core.exceptions import ConfigurationError
from src.core.types import ModelBundle


@dataclass(slots=True, frozen=True)
class CompileConfig:
    """Resolved torch.compile configuration."""

    enabled: bool = False
    mode: str = "default"
    backend: str | None = None
    dynamic: bool = False
    fullgraph: bool = False


def build_compile_config(cfg: DictConfig) -> CompileConfig:
    """Resolve compile config from top-level Hydra config."""

    compile_cfg = cfg.get("compile")
    if compile_cfg is None:
        return CompileConfig(enabled=False)

    raw_backend = str(compile_cfg.get("backend", "")).strip()
    backend = raw_backend or None
    return CompileConfig(
        enabled=bool(compile_cfg.get("enabled", False)),
        mode=str(compile_cfg.get("mode", "default")),
        backend=backend,
        dynamic=bool(compile_cfg.get("dynamic", False)),
        fullgraph=bool(compile_cfg.get("fullgraph", False)),
    )


def compile_module_if_enabled(
    module: torch.nn.Module,
    compile_cfg: CompileConfig,
) -> torch.nn.Module:
    """Compile module with torch.compile when enabled."""

    if not compile_cfg.enabled:
        return module

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise ConfigurationError(
            "compile.enabled=true requires torch.compile, but current PyTorch does not provide it."
        )

    kwargs: dict[str, object] = {
        "mode": compile_cfg.mode,
        "dynamic": compile_cfg.dynamic,
        "fullgraph": compile_cfg.fullgraph,
    }
    if compile_cfg.backend is not None:
        kwargs["backend"] = compile_cfg.backend

    return compile_fn(module, **kwargs)


def compile_model_bundle(
    model_bundle: ModelBundle,
    compile_cfg: CompileConfig,
) -> ModelBundle:
    """Compile actor/critics in a model bundle based on compile config."""

    actor = compile_module_if_enabled(model_bundle.actor, compile_cfg)
    critic1 = (
        compile_module_if_enabled(model_bundle.critic1, compile_cfg)
        if model_bundle.critic1 is not None
        else None
    )
    critic2 = (
        compile_module_if_enabled(model_bundle.critic2, compile_cfg)
        if model_bundle.critic2 is not None
        else None
    )
    return ModelBundle(actor=actor, critic1=critic1, critic2=critic2)
