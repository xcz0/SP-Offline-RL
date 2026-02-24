"""Dataset loading and target sampling for simulator evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.core.exceptions import DataValidationError
from src.evaluation.sp_sim.types import EvalTarget

SIM_REQUIRED_COLUMNS = (
    "card_id",
    "day_offset",
    "elapsed_days",
    "elapsed_seconds",
    "rating",
    "duration",
    "state",
)


def _parse_user_id(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("user_id="):
        return None
    try:
        return int(stem.split("=", 1)[1])
    except ValueError:
        return None


def list_processed_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(data_dir.glob("user_id=*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No processed parquet files found under {data_dir} with pattern user_id=*.parquet."
        )
    return files


def load_user_dataframe(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(SIM_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(f"{path} missing required columns: {missing_text}")

    inferred_user_id = _parse_user_id(path)
    if "user_id" not in frame.columns:
        if inferred_user_id is None:
            raise DataValidationError(
                f"{path} has no user_id column and filename is not user_id=<id>.parquet."
            )
        frame = frame.copy()
        frame["user_id"] = inferred_user_id
    elif inferred_user_id is not None:
        frame = frame.copy()
        frame["user_id"] = frame["user_id"].fillna(inferred_user_id)

    sort_cols = ["review_th"] if "review_th" in frame.columns else ["day_offset"]
    frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return frame


def load_user_data_map(
    data_dir: Path,
    user_ids: Sequence[int] | None = None,
) -> dict[int, pd.DataFrame]:
    files = list_processed_files(data_dir)
    selected = set(int(uid) for uid in user_ids) if user_ids else None

    data_map: dict[int, pd.DataFrame] = {}
    for path in files:
        user_id = _parse_user_id(path)
        if user_id is None:
            continue
        if selected is not None and user_id not in selected:
            continue
        data_map[user_id] = load_user_dataframe(path)

    if not data_map:
        users_text = ",".join(str(v) for v in sorted(selected)) if selected else "ALL"
        raise DataValidationError(
            f"No user data loaded from {data_dir}. requested_user_ids={users_text}."
        )
    return data_map


def _warmup_occurrence_from_mode(mode: str) -> int:
    normalized = mode.strip().lower()
    if normalized == "second":
        return 2
    if normalized == "fifth":
        return 5
    raise ValueError(f"Unsupported warmup mode: {mode}. Use 'second' or 'fifth'.")


def sample_eval_targets(
    user_data: dict[int, pd.DataFrame],
    cards_per_user: int,
    min_target_occurrences: int,
    warmup_mode: str,
    seed: int,
) -> list[EvalTarget]:
    """Sample target cards per user for simulator evaluation."""

    if cards_per_user <= 0:
        raise ValueError(f"cards_per_user must be > 0, got {cards_per_user}.")

    warmup_occurrence = _warmup_occurrence_from_mode(warmup_mode)
    required_occurrence = max(int(min_target_occurrences), warmup_occurrence + 1)
    rng = np.random.default_rng(seed)
    targets: list[EvalTarget] = []

    for user_id in sorted(user_data):
        df = user_data[user_id]
        counts = df["card_id"].value_counts().sort_index()
        eligible = counts[counts >= required_occurrence].index.to_numpy(dtype=np.int64)
        if eligible.size == 0:
            continue
        n_pick = min(cards_per_user, int(eligible.size))
        chosen = rng.choice(eligible, size=n_pick, replace=False)
        chosen_set = set(int(v) for v in chosen.tolist())

        for card_id in sorted(chosen_set):
            card_rows = df[df["card_id"] == card_id].reset_index(drop=True)
            warmup_row = card_rows.iloc[warmup_occurrence - 1]
            targets.append(
                EvalTarget(
                    user_id=int(user_id),
                    card_id=int(card_id),
                    occurrences=int(len(card_rows)),
                    warmup_occurrence=warmup_occurrence,
                    warmup_end_day_offset=float(warmup_row["day_offset"]),
                )
            )
    return targets

