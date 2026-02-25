from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.dataset import _build_user_eval_data


def test_build_user_eval_data_keeps_global_temporal_order_for_simulator() -> None:
    frame = pd.DataFrame(
        [
            {"card_id": 20, "day_offset": 9.0, "review_th": 3},
            {"card_id": 10, "day_offset": 1.0, "review_th": 1},
            {"card_id": 20, "day_offset": 2.0, "review_th": 2},
        ]
    )

    user_data = _build_user_eval_data(frame)

    assert user_data.frame["review_th"].tolist() == [1, 2, 3]
    assert user_data.frame["card_id"].tolist() == [10, 20, 20]


def test_build_user_eval_data_sorts_per_card_for_target_sampling() -> None:
    frame = pd.DataFrame(
        [
            {"card_id": 30, "day_offset": 4.0, "review_th": 3},
            {"card_id": 30, "day_offset": 1.0, "review_th": 1},
            {"card_id": 30, "day_offset": 2.0, "review_th": 2},
            {"card_id": 40, "day_offset": 8.0, "review_th": 4},
        ]
    )

    user_data = _build_user_eval_data(frame)

    assert user_data.card_frames[30]["day_offset"].tolist() == [1.0, 2.0, 4.0]
    assert int(user_data.card_counts.loc[30]) == 3
    assert user_data.card_counts.index.dtype == np.int64
