from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.sp_sim.dataset import sample_eval_targets
from src.evaluation.sp_sim.scoring import add_score_column, summarize_scored_metrics


def test_sample_eval_targets_uses_warmup_mode_fifth() -> None:
    rows = []
    for idx in range(7):
        rows.append({"card_id": 10, "day_offset": float(idx)})
    for idx in range(6):
        rows.append({"card_id": 20, "day_offset": float(idx + 10)})
    user_df = pd.DataFrame(rows)

    targets = sample_eval_targets(
        user_data={1: user_df},
        cards_per_user=1,
        min_target_occurrences=5,
        warmup_mode="fifth",
        seed=0,
    )
    assert len(targets) == 1
    assert targets[0].warmup_occurrence == 5
    assert targets[0].warmup_end_day_offset in {4.0, 14.0}


def test_add_score_column_and_summary() -> None:
    frame = pd.DataFrame(
        {
            "retention_area": [0.8, 0.4],
            "final_retention": [0.9, 0.2],
            "review_count": [4.0, 2.0],
        }
    )
    scored = add_score_column(
        frame,
        weights={
            "retention_area": 0.5,
            "final_retention": 0.3,
            "review_count_norm": 0.2,
        },
    )
    expected = np.array(
        [
            0.5 * 0.8 + 0.3 * 0.9 - 0.2 * (4 / 4),
            0.5 * 0.4 + 0.3 * 0.2 - 0.2 * (2 / 4),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(scored["score"].to_numpy(dtype=np.float32), expected)

    summary = summarize_scored_metrics(scored)
    assert summary["num_targets"] == 2.0
    assert np.isfinite(summary["score_mean"])

