from __future__ import annotations

from pathlib import Path

import numpy as np

from src.runners.types import EvalMetrics, TrainingMetrics, TrainingResult


def test_training_result_to_dict_converts_numpy_and_path() -> None:
    result = TrainingResult(
        mode="train",
        log_path="/tmp/log",
        run_name="run",
        checkpoint_path="/tmp/log/policy.pth",
        evaluation=EvalMetrics(
            test_reward_mean=1.0,
            test_reward_std=0.1,
            extra={"vector": np.array([1.0, 2.0], dtype=np.float32)},
        ),
        training=TrainingMetrics(
            best_reward=np.float32(2.5).item(),
            best_reward_std=np.float32(0.2).item(),
            update_step=10,
            train_step=20,
            test_step=30,
            raw={
                "scalar": np.float32(3.14),
                "path": Path("/tmp/log/policy.pth"),
            },
        ),
    )

    data = result.to_dict()
    assert data["evaluation"]["extra"]["vector"] == [1.0, 2.0]
    assert isinstance(data["training"]["raw"]["scalar"], float)
    assert data["training"]["raw"]["path"] == "/tmp/log/policy.pth"
