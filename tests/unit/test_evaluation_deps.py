from __future__ import annotations

from src.evaluation.deps import resolve_predictor_device


def test_resolve_predictor_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_predictor_device("auto") == "cpu"


def test_resolve_predictor_device_auto_uses_single_gpu(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_predictor_device("auto") == "cuda:0"


def test_resolve_predictor_device_normalizes_cuda_alias() -> None:
    assert resolve_predictor_device("cuda") == "cuda:0"
