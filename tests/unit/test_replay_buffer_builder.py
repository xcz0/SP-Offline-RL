from __future__ import annotations

from tests.factories.dataset_factory import build_offline_dataset

from src.data.replay_buffer_builder import build_replay_buffer


def _sample_data(n: int = 16):
    return build_offline_dataset(n=n, seed=51)


def test_build_replay_buffer_full_and_truncated() -> None:
    data = _sample_data(16)
    buffer = build_replay_buffer(data)
    assert len(buffer) == 16

    truncated = build_replay_buffer(data, buffer_size=5)
    assert len(truncated) == 5
