"""Build behavior-cloning datasets from processed spaced-repetition parquet files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CARD_FEATURE_COLUMNS: tuple[str, ...] = (
    "scaled_elapsed_days",
    "scaled_elapsed_days_cumulative",
    "scaled_elapsed_seconds",
    "elapsed_seconds_sin",
    "elapsed_seconds_cos",
    "scaled_elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin",
    "elapsed_seconds_cumulative_cos",
    "scaled_duration",
    "rating_1",
    "rating_2",
    "rating_3",
    "rating_4",
    "note_id_is_nan",
    "deck_id_is_nan",
    "preset_id_is_nan",
    "day_offset_diff",
    "day_of_week",
    "diff_new_cards",
    "diff_reviews",
    "cum_new_cards_today",
    "cum_reviews_today",
    "scaled_state",
    "is_query",
)


@dataclass(frozen=True)
class BCDatasetArtifacts:
    train: pd.DataFrame
    val: pd.DataFrame
    report: dict[str, Any]


def discover_processed_parquet_files(input_dir: Path) -> list[Path]:
    """Return processed parquet files under input directory."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")

    user_files = sorted(input_dir.glob("user_id=*.parquet"))
    if user_files:
        return user_files

    all_parquet = sorted(input_dir.glob("*.parquet"))
    if not all_parquet:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")
    return all_parquet


def _infer_user_id(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("user_id="):
        return None
    value = stem.split("=", 1)[1]
    try:
        return int(value)
    except ValueError:
        return None


def _load_single_parquet(
    path: Path, feature_columns: Sequence[str]
) -> tuple[pd.DataFrame, int | None]:
    schema = pq.read_schema(path)
    available = set(schema.names)

    required = {"card_id", "elapsed_days", *feature_columns}
    missing = sorted(required - available)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{path} missing required columns: {missing_text}")

    columns_to_read = [
        "card_id",
        "elapsed_days",
        *feature_columns,
    ]
    for optional_col in ("user_id", "review_th", "day_offset"):
        if optional_col in available:
            columns_to_read.append(optional_col)

    frame = pd.read_parquet(path, columns=columns_to_read)
    inferred_user_id = _infer_user_id(path)
    if "user_id" not in frame.columns:
        if inferred_user_id is None:
            raise ValueError(
                f"{path} has no 'user_id' column and filename does not encode user_id."
            )
        frame["user_id"] = inferred_user_id
    elif inferred_user_id is not None:
        frame["user_id"] = frame["user_id"].fillna(inferred_user_id)

    return frame, inferred_user_id


def load_processed_dataframe(
    paths: Sequence[Path],
    feature_columns: Sequence[str] = CARD_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Load and concatenate processed rows from parquet files."""

    if not paths:
        raise ValueError("At least one parquet file is required.")

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame, _ = _load_single_parquet(path, feature_columns)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def compute_action_clip_max(global_max_interval: float) -> float:
    """Compute action upper bound from simulator rule."""

    return float(max(30.0, 1.5 * float(global_max_interval)))


def clip_action_array(actions: np.ndarray, action_max: float) -> np.ndarray:
    """Clip action labels to valid interval range."""

    if action_max <= 0:
        raise ValueError(f"action_max must be > 0, got {action_max}.")
    return np.clip(actions, 0.0, float(action_max)).astype(np.float32, copy=False)


def _coerce_required_numeric(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    working = frame.copy()
    numeric_columns = ["user_id", "card_id", "elapsed_days", *feature_columns]
    for col in numeric_columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    before_rows = len(working)
    working = working.dropna(subset=numeric_columns).copy()
    dropped = before_rows - len(working)

    working["user_id"] = working["user_id"].astype(np.int64)
    working["card_id"] = working["card_id"].astype(np.int64)
    working["elapsed_days"] = working["elapsed_days"].astype(np.float32)
    return working, dropped


def _sort_for_next_interval(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    working = frame.copy()
    has_review_th = "review_th" in working.columns
    has_day_offset = "day_offset" in working.columns
    if not has_review_th and not has_day_offset:
        raise ValueError("Input rows must contain 'review_th' or 'day_offset' for sorting.")

    sort_columns = ["user_id", "card_id"]
    if has_review_th:
        working["review_th"] = pd.to_numeric(working["review_th"], errors="coerce")
        sort_columns.append("review_th")
    if has_day_offset:
        working["day_offset"] = pd.to_numeric(working["day_offset"], errors="coerce")
        sort_columns.append("day_offset")

    return (
        working.sort_values(sort_columns, kind="mergesort").reset_index(drop=True),
        sort_columns,
    )


def build_labeled_bc_examples(
    processed_rows: pd.DataFrame,
    feature_columns: Sequence[str] = CARD_FEATURE_COLUMNS,
    action_clip_mode: str = "simulator_bound",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build labeled behavior-cloning examples from processed review rows."""

    if action_clip_mode != "simulator_bound":
        raise ValueError(
            f"Unsupported action_clip_mode '{action_clip_mode}'. "
            "Only 'simulator_bound' is supported."
        )

    required = {"user_id", "card_id", "elapsed_days", *feature_columns}
    missing_required = sorted(required - set(processed_rows.columns))
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise ValueError(f"Input rows missing required columns: {missing_text}")

    numeric_rows, dropped_non_numeric_rows = _coerce_required_numeric(
        processed_rows, feature_columns
    )
    sorted_rows, sort_columns = _sort_for_next_interval(numeric_rows)

    sorted_rows["next_elapsed_days"] = (
        sorted_rows.groupby(["user_id", "card_id"], sort=False)["elapsed_days"]
        .shift(-1)
        .astype(np.float32)
    )

    terminal_rows = int(sorted_rows["next_elapsed_days"].isna().sum())
    examples = sorted_rows[sorted_rows["next_elapsed_days"].notna()].copy()
    if examples.empty:
        raise ValueError("No behavior-cloning examples were produced from input rows.")

    raw_actions = examples["next_elapsed_days"].to_numpy(dtype=np.float32, copy=False)
    global_max_interval = float(np.max(raw_actions))
    action_max = compute_action_clip_max(global_max_interval)
    clipped_actions = clip_action_array(raw_actions, action_max)
    clipped_count = int(np.count_nonzero(np.abs(clipped_actions - raw_actions) > 1e-6))

    obs_matrix = examples.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(obs_matrix).all():
        raise ValueError("Observation matrix contains NaN or Inf.")
    if not np.isfinite(clipped_actions).all():
        raise ValueError("Action labels contain NaN or Inf.")

    out = pd.DataFrame(
        {
            "user_id": examples["user_id"].to_numpy(dtype=np.int64),
            "card_id": examples["card_id"].to_numpy(dtype=np.int64),
            "obs": [row.copy() for row in obs_matrix],
            "act": clipped_actions.astype(np.float32),
            "act_raw": raw_actions.astype(np.float32),
        }
    )

    stats: dict[str, Any] = {
        "rows_input": int(len(processed_rows)),
        "rows_after_numeric_drop": int(len(numeric_rows)),
        "rows_dropped_non_numeric": dropped_non_numeric_rows,
        "rows_terminal_dropped": terminal_rows,
        "rows_examples": int(len(out)),
        "cards_examples": int(out["card_id"].nunique()),
        "users_examples": int(out["user_id"].nunique()),
        "feature_dim": int(len(feature_columns)),
        "sort_columns": sort_columns,
        "global_max_interval": global_max_interval,
        "action_max": action_max,
        "action_clip_count": clipped_count,
        "action_clip_fraction": float(clipped_count / len(out)),
    }
    return out, stats


def split_examples_by_card_id(
    examples: pd.DataFrame,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split examples by card_id to avoid leakage across train/val."""

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")
    if examples.empty:
        raise ValueError("examples must not be empty.")

    unique_cards = examples["card_id"].dropna().astype(np.int64).unique()
    unique_cards = np.asarray(unique_cards, dtype=np.int64)
    if unique_cards.size == 0:
        raise ValueError("No card_id values found for split.")

    rng = np.random.default_rng(seed)
    shuffled_cards = unique_cards.copy()
    rng.shuffle(shuffled_cards)

    if val_ratio == 0.0 or unique_cards.size == 1:
        val_cards = np.array([], dtype=np.int64)
    else:
        n_val = int(np.floor(unique_cards.size * val_ratio))
        if n_val <= 0:
            n_val = 1
        if n_val >= unique_cards.size:
            n_val = unique_cards.size - 1
        val_cards = shuffled_cards[:n_val]

    val_card_set = set(val_cards.tolist())
    is_val = examples["card_id"].isin(val_card_set)
    train = examples.loc[~is_val].reset_index(drop=True)
    val = examples.loc[is_val].reset_index(drop=True)

    if train.empty:
        raise ValueError("Train split is empty after card_id split.")

    return train, val


def to_obs_act_parquet_frame(examples: pd.DataFrame) -> pd.DataFrame:
    """Convert examples to parquet-compatible obs/act list columns."""

    if examples.empty:
        return pd.DataFrame({"user_id": [], "card_id": [], "obs": [], "act": []})

    obs_col = [np.asarray(obs, dtype=np.float32).tolist() for obs in examples["obs"]]
    act_col = [
        [float(a)] for a in examples["act"].to_numpy(dtype=np.float32, copy=False)
    ]
    return pd.DataFrame(
        {
            "user_id": examples["user_id"].to_numpy(dtype=np.int64, copy=False),
            "card_id": examples["card_id"].to_numpy(dtype=np.int64, copy=False),
            "obs": obs_col,
            "act": act_col,
        }
    )


def build_bc_datasets(
    input_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 0,
    feature_columns: Sequence[str] = CARD_FEATURE_COLUMNS,
    action_clip_mode: str = "simulator_bound",
) -> BCDatasetArtifacts:
    """Build train/val BC datasets from processed parquet directory."""

    files = discover_processed_parquet_files(input_dir)
    processed = load_processed_dataframe(files, feature_columns=feature_columns)
    examples, stats = build_labeled_bc_examples(
        processed_rows=processed,
        feature_columns=feature_columns,
        action_clip_mode=action_clip_mode,
    )
    train_examples, val_examples = split_examples_by_card_id(
        examples, val_ratio=val_ratio, seed=seed
    )
    train_frame = to_obs_act_parquet_frame(train_examples)
    val_frame = to_obs_act_parquet_frame(val_examples)

    train_cards = set(train_examples["card_id"].tolist())
    val_cards = set(val_examples["card_id"].tolist())
    report = {
        "input_dir": str(input_dir),
        "input_files": [str(path) for path in files],
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        **stats,
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "train_cards": int(len(train_cards)),
        "val_cards": int(len(val_cards)),
        "card_overlap_count": int(len(train_cards.intersection(val_cards))),
    }
    return BCDatasetArtifacts(train=train_frame, val=val_frame, report=report)


def write_bc_artifacts(
    artifacts: BCDatasetArtifacts,
    train_output: Path,
    val_output: Path,
    report_output: Path,
) -> None:
    """Write train/val parquet and report JSON."""

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    artifacts.train.to_parquet(train_output, index=False)
    artifacts.val.to_parquet(val_output, index=False)
    report_output.write_text(
        json.dumps(artifacts.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

