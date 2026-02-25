from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation.dataset import UserEvalData
from src.evaluation.replay_evaluator import (
    _evaluate_replay_for_user,
    evaluate_replay_with_simulator,
)
from src.evaluation.types import EvalTarget


class _DummyPredictor:
    def __init__(self, **_: object) -> None:
        self.review_count = 0

    def reset_state(self) -> None:
        self.review_count = 0

    def predict_query(self, _row: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(imm_prob=np.asarray(0.8, dtype=np.float32))

    def sample_rating(
        self,
        _query_row: dict[str, object],
        generator: object = None,
    ) -> tuple[int, SimpleNamespace]:
        _ = generator
        return 3, self.predict_query({})

    def step_review(self, _row: dict[str, object]) -> SimpleNamespace:
        self.review_count += 1
        return SimpleNamespace(curve=None)


class _DummyEnv:
    saw_rating_generator = False

    def __init__(self, **kwargs: object) -> None:
        _DummyEnv.saw_rating_generator = "rating_generator" in kwargs
        self._steps = 0
        self._obs = SimpleNamespace(
            current_day_offset=0.0,
            metrics=SimpleNamespace(
                retention_area=0.0,
                target_review_count=0,
                total_review_count=0,
            ),
        )

    def prepare(self) -> None:
        return None

    def reset(self) -> SimpleNamespace:
        return self._obs

    def step(self, _delta_days: float) -> SimpleNamespace:
        self._steps += 1
        done = self._steps >= 1
        obs = SimpleNamespace(
            current_day_offset=1.0,
            metrics=SimpleNamespace(
                retention_area=1.0,
                target_review_count=1,
                total_review_count=1,
            ),
        )
        return SimpleNamespace(
            done=done,
            observation=obs,
            sim_result=SimpleNamespace(query_imm_prob=0.9),
        )


def test_replay_evaluator_does_not_pass_rating_callback_as_generator() -> None:
    frame = pd.DataFrame(
        [
            {"card_id": 7, "day_offset": 0.0, "rating": 3, "duration": 10.0, "state": 1.0},
            {"card_id": 7, "day_offset": 1.0, "rating": 4, "duration": 11.0, "state": 2.0},
            {"card_id": 7, "day_offset": 3.0, "rating": 3, "duration": 12.0, "state": 2.0},
        ]
    )
    user_data = UserEvalData(
        frame=frame,
        card_frames={7: frame.copy()},
        card_counts=pd.Series([3], index=np.asarray([7], dtype=np.int64)),
    )
    targets = [
        EvalTarget(
            user_id=3,
            card_id=7,
            occurrences=3,
            warmup_occurrence=2,
            warmup_end_day_offset=1.0,
        )
    ]

    final_rows, traj_rows = _evaluate_replay_for_user(
        user_id=3,
        user_data=user_data,
        user_targets=targets,
        RWKVSrsPredictor=_DummyPredictor,
        RWKVSrsRlEnv=_DummyEnv,
        predictor_model_path="dummy.pth",
        predictor_device="cpu",
        torch_dtype="float32",
    )

    assert not _DummyEnv.saw_rating_generator
    assert len(final_rows) == 1
    assert len(traj_rows) == 1
    assert traj_rows[0]["replay_rating"] == 3


def test_replay_evaluator_resolves_auto_predictor_device(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {"card_id": 7, "day_offset": 0.0, "rating": 3, "duration": 10.0, "state": 1.0},
            {"card_id": 7, "day_offset": 1.0, "rating": 4, "duration": 11.0, "state": 2.0},
            {"card_id": 7, "day_offset": 3.0, "rating": 3, "duration": 12.0, "state": 2.0},
        ]
    )
    user_data = UserEvalData(
        frame=frame,
        card_frames={7: frame.copy()},
        card_counts=pd.Series([3], index=np.asarray([7], dtype=np.int64)),
    )
    targets = [
        EvalTarget(
            user_id=3,
            card_id=7,
            occurrences=3,
            warmup_occurrence=2,
            warmup_end_day_offset=1.0,
        )
    ]

    monkeypatch.setattr(
        "src.evaluation.replay_evaluator.load_user_data_map",
        lambda **_: {3: user_data},
    )
    monkeypatch.setattr(
        "src.evaluation.replay_evaluator.sample_eval_targets",
        lambda **_: targets,
    )
    monkeypatch.setattr(
        "src.evaluation.replay_evaluator.resolve_predictor_device",
        lambda _: "cuda:0",
    )

    captured_devices: list[str] = []

    class _CapturePredictor(_DummyPredictor):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            captured_devices.append(str(kwargs["device"]))

    monkeypatch.setattr(
        "src.evaluation.replay_evaluator.require_sprwkv",
        lambda: (_CapturePredictor, _DummyEnv, None),
    )

    result = evaluate_replay_with_simulator(
        data_dir="data",
        predictor_model_path="dummy.pth",
        predictor_device="auto",
        predictor_dtype="float32",
        user_ids=[3],
        cards_per_user=1,
        min_target_occurrences=3,
        warmup_mode="second",
        seed=0,
        eval_workers=1,
    )

    assert captured_devices == ["cuda:0"]
    assert result.summary["num_targets"] == 1
