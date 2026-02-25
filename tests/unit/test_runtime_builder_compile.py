from __future__ import annotations

from typing import Any

import torch
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from tianshou.utils.space_info import ActionSpaceInfo, ObservationSpaceInfo, SpaceInfo

from src.core.types import ModelBundle
from src.runners import runtime_builder


class _DummyEnv:
    def __init__(self) -> None:
        self.observation_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=float)


class _DummyModelFactory:
    def build(self, _cfg: Any, _space_info: SpaceInfo, _device: str) -> ModelBundle:
        return ModelBundle(
            actor=torch.nn.Linear(3, 1),
            critic1=torch.nn.Linear(3, 1),
            critic2=torch.nn.Linear(3, 1),
        )


class _DummyAlgoFactory:
    def build(
        self,
        _cfg: Any,
        _env: Any,
        model_bundle: ModelBundle,
        device: str,
    ) -> dict[str, Any]:
        return {"model_bundle": model_bundle, "device": device}


def _compile_cfg() -> object:
    return OmegaConf.create(
        {
            "model": {"name": "mlp_actor_critic"},
            "algo": {"name": "td3_bc"},
            "compile": {
                "enabled": True,
                "mode": "reduce-overhead",
                "backend": "inductor",
                "dynamic": False,
                "fullgraph": False,
            },
        }
    )


def test_build_algorithm_applies_compile_config(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[bool, str, str | None, bool, bool]] = []

    monkeypatch.setattr(
        runtime_builder,
        "get_model_factory",
        lambda _name: _DummyModelFactory(),
    )
    monkeypatch.setattr(
        runtime_builder,
        "get_algo_factory",
        lambda _name: _DummyAlgoFactory(),
    )

    def _capture_compile(model_bundle: ModelBundle, compile_cfg: Any) -> ModelBundle:
        calls.append(
            (
                bool(compile_cfg.enabled),
                str(compile_cfg.mode),
                compile_cfg.backend,
                bool(compile_cfg.dynamic),
                bool(compile_cfg.fullgraph),
            )
        )
        return model_bundle

    monkeypatch.setattr(runtime_builder, "compile_model_bundle", _capture_compile)

    result = runtime_builder.build_algorithm(_compile_cfg(), _DummyEnv(), "cpu")
    assert result["device"] == "cpu"
    assert calls == [(True, "reduce-overhead", "inductor", False, False)]


def test_build_algorithm_from_space_info_applies_compile_config(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[bool, str, str | None, bool, bool]] = []

    monkeypatch.setattr(
        runtime_builder,
        "get_model_factory",
        lambda _name: _DummyModelFactory(),
    )
    monkeypatch.setattr(
        runtime_builder,
        "get_algo_factory",
        lambda _name: _DummyAlgoFactory(),
    )

    def _capture_compile(model_bundle: ModelBundle, compile_cfg: Any) -> ModelBundle:
        calls.append(
            (
                bool(compile_cfg.enabled),
                str(compile_cfg.mode),
                compile_cfg.backend,
                bool(compile_cfg.dynamic),
                bool(compile_cfg.fullgraph),
            )
        )
        return model_bundle

    monkeypatch.setattr(runtime_builder, "compile_model_bundle", _capture_compile)

    space_info = SpaceInfo(
        observation_info=ObservationSpaceInfo(obs_shape=(3,)),
        action_info=ActionSpaceInfo(action_shape=(1,), min_action=-1.0, max_action=1.0),
    )
    action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
    result = runtime_builder.build_algorithm_from_space_info(
        _compile_cfg(),
        space_info=space_info,
        action_space=action_space,
        device="cpu",
    )

    assert result["device"] == "cpu"
    assert calls == [(True, "reduce-overhead", "inductor", False, False)]
