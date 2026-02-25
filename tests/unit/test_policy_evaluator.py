from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.bc_dataset_builder import CARD_FEATURE_COLUMNS
from src.evaluation.dataset import UserEvalData
from src.evaluation.policy_evaluator import _policy_act_days, evaluate_policy_with_simulator
from src.evaluation.types import EvalTarget


class _DummyPolicy:
    def __call__(self, _batch: object) -> SimpleNamespace:
        return SimpleNamespace(act=np.asarray([[1.0]], dtype=np.float32))


def test_policy_evaluator_uses_background_rows_to_keep_feature_columns(
    monkeypatch,
) -> None:
    row = {name: 0.1 for name in CARD_FEATURE_COLUMNS}
    row.update(
        {
            "card_id": 123,
            "day_offset": 1.0,
            "elapsed_days": 1.0,
            "elapsed_seconds": 86400.0,
            "rating": 3,
            "duration": 30.0,
            "state": 2.0,
            "user_id": 3,
        }
    )
    frame = pd.DataFrame([row, {**row, "day_offset": 2.0}, {**row, "day_offset": 3.0}])
    user_data = UserEvalData(
        frame=frame,
        card_frames={123: frame.copy()},
        card_counts=pd.Series([3], index=np.asarray([123], dtype=np.int64)),
    )

    monkeypatch.setattr(
        "src.evaluation.policy_evaluator.load_user_data_map",
        lambda **_: {3: user_data},
    )
    monkeypatch.setattr(
        "src.evaluation.policy_evaluator.sample_eval_targets",
        lambda **_: [
            EvalTarget(
                user_id=3,
                card_id=123,
                occurrences=3,
                warmup_occurrence=2,
                warmup_end_day_offset=2.0,
            )
        ],
    )

    class _DummyPredictor:
        def __init__(self, **_: object) -> None:
            pass

    class _DummyEnv:
        saw_feature_column = False

        def __init__(self, **kwargs: object) -> None:
            rows = kwargs.get("background_rows")
            assert isinstance(rows, list)
            assert kwargs.get("parquet_df") is None
            assert len(rows) > 0
            self._row = rows[0]
            self._obs = SimpleNamespace(
                last_target_row=self._row,
                metrics=SimpleNamespace(retention_area=0.5, target_review_count=1),
            )
            self._step_obs = SimpleNamespace(
                last_target_row=self._row,
                metrics=SimpleNamespace(retention_area=0.6, target_review_count=1),
            )
            _DummyEnv.saw_feature_column = CARD_FEATURE_COLUMNS[0] in self._row

        def prepare(self) -> None:
            return None

        def reset(self) -> SimpleNamespace:
            return self._obs

        def step(self, _delta_days: float) -> SimpleNamespace:
            return SimpleNamespace(
                observation=self._step_obs,
                done=True,
                sim_result=SimpleNamespace(query_imm_prob=0.9),
            )

    monkeypatch.setattr(
        "src.evaluation.policy_evaluator.require_sprwkv",
        lambda: (_DummyPredictor, _DummyEnv, None),
    )

    result = evaluate_policy_with_simulator(
        policy=_DummyPolicy(),
        data_dir="data",
        predictor_model_path="dummy.pth",
        predictor_device="cpu",
        predictor_dtype="float32",
        user_ids=[3],
        cards_per_user=1,
        min_target_occurrences=3,
        warmup_mode="second",
        seed=0,
        action_max=10.0,
        score_weights={
            "retention_area": 0.5,
            "final_retention": 0.3,
            "review_count_norm": 0.2,
        },
    )

    assert _DummyEnv.saw_feature_column
    assert result.summary["num_targets"] == 1.0


def test_policy_act_days_accepts_tensor_requiring_grad() -> None:
    class _GradPolicy:
        def __call__(self, _batch: object) -> SimpleNamespace:
            tensor = torch.tensor([[2.5]], dtype=torch.float32, requires_grad=True)
            return SimpleNamespace(act=tensor)

    action = _policy_act_days(_GradPolicy(), np.zeros((24,), dtype=np.float32))
    assert action == 2.5
