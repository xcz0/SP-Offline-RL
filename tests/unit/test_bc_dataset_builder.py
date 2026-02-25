from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.bc_dataset_builder import (
    CARD_FEATURE_COLUMNS,
    build_bc_datasets,
    build_labeled_bc_examples,
    clip_action_array,
    compute_action_clip_max,
    split_examples_by_card_id,
    write_bc_artifacts,
)
from src.data.dataset_adapter import BC_DATA_FIELDS
from src.data.parquet_adapter import ParquetOfflineDatasetAdapter


def _feature_values(base: float) -> dict[str, float]:
    return {name: float(base + idx) for idx, name in enumerate(CARD_FEATURE_COLUMNS)}


def _make_row(
    user_id: int,
    card_id: int,
    review_th: int,
    day_offset: int,
    elapsed_days: float,
    base: float,
) -> dict[str, float]:
    row: dict[str, float] = {
        "user_id": float(user_id),
        "card_id": float(card_id),
        "review_th": float(review_th),
        "day_offset": float(day_offset),
        "elapsed_days": float(elapsed_days),
    }
    row.update(_feature_values(base))
    return row


def test_build_labeled_examples_next_elapsed_days_is_action_label() -> None:
    rows = [
        _make_row(1, 100, 2, 4, 4.0, 20.0),
        _make_row(1, 200, 1, 0, -1.0, 30.0),
        _make_row(1, 100, 1, 0, -1.0, 10.0),
        _make_row(1, 100, 3, 13, 9.0, 40.0),
        _make_row(1, 200, 2, 2, 2.0, 50.0),
    ]
    frame = pd.DataFrame(rows)

    examples, stats = build_labeled_bc_examples(frame)
    actions_100 = examples.loc[examples["card_id"] == 100, "act"].tolist()
    actions_200 = examples.loc[examples["card_id"] == 200, "act"].tolist()

    assert actions_100 == pytest.approx([4.0, 9.0], abs=1e-6)
    assert actions_200 == pytest.approx([2.0], abs=1e-6)
    assert stats["rows_terminal_dropped"] == 2


def test_split_examples_by_card_id_has_no_leakage() -> None:
    frame = pd.DataFrame(
        [
            _make_row(1, 101, 1, 0, -1.0, 1.0),
            _make_row(1, 101, 2, 3, 3.0, 2.0),
            _make_row(1, 102, 1, 0, -1.0, 3.0),
            _make_row(1, 102, 2, 5, 5.0, 4.0),
            _make_row(2, 201, 1, 0, -1.0, 5.0),
            _make_row(2, 201, 2, 4, 4.0, 6.0),
            _make_row(2, 202, 1, 0, -1.0, 7.0),
            _make_row(2, 202, 2, 6, 6.0, 8.0),
        ]
    )
    examples, _ = build_labeled_bc_examples(frame)
    train, val = split_examples_by_card_id(examples, val_ratio=0.5, seed=0)

    train_cards = set(train["card_id"].tolist())
    val_cards = set(val["card_id"].tolist())
    assert train_cards.isdisjoint(val_cards)
    assert len(train) + len(val) == len(examples)


def test_action_clipping_helpers() -> None:
    assert compute_action_clip_max(5.0) == pytest.approx(30.0)
    assert compute_action_clip_max(100.0) == pytest.approx(150.0)

    raw = np.array([-3.0, 4.0, 12.0], dtype=np.float32)
    clipped = clip_action_array(raw, action_max=10.0)
    np.testing.assert_allclose(clipped, np.array([0.0, 4.0, 10.0], dtype=np.float32))


def test_build_bc_datasets_outputs_parquet_compatible_obs_act(tmp_path: Path) -> None:
    file_1 = tmp_path / "user_id=1.parquet"
    file_2 = tmp_path / "user_id=2.parquet"
    frame_1 = pd.DataFrame(
        [
            _make_row(1, 101, 1, 0, -1.0, 11.0),
            _make_row(1, 101, 2, 3, 3.0, 12.0),
            _make_row(1, 102, 1, 0, -1.0, 13.0),
            _make_row(1, 102, 2, 4, 4.0, 14.0),
        ]
    )
    frame_2 = pd.DataFrame(
        [
            _make_row(2, 201, 1, 0, -1.0, 21.0),
            _make_row(2, 201, 2, 5, 5.0, 22.0),
            _make_row(2, 202, 1, 0, -1.0, 23.0),
            _make_row(2, 202, 2, 6, 6.0, 24.0),
        ]
    )
    frame_1.to_parquet(file_1, index=False)
    frame_2.to_parquet(file_2, index=False)

    artifacts = build_bc_datasets(input_dir=tmp_path, val_ratio=0.5, seed=0)
    assert artifacts.report["card_overlap_count"] == 0
    assert artifacts.report["train_rows"] > 0
    assert artifacts.report["val_rows"] > 0

    train_output = tmp_path / "bc_train.parquet"
    val_output = tmp_path / "bc_val.parquet"
    report_output = tmp_path / "bc_report.json"
    write_bc_artifacts(artifacts, train_output, val_output, report_output)

    adapter = ParquetOfflineDatasetAdapter(
        path=str(train_output),
        columns={"obs": "obs", "act": "act"},
    )
    data = adapter.load_prepared(fields=BC_DATA_FIELDS)
    assert data["obs"].shape[1] == len(CARD_FEATURE_COLUMNS)
    assert data["act"].shape[1] == 1
