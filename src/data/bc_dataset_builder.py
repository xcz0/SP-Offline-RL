"""Build behavior-cloning datasets from processed spaced-repetition parquet files."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import polars as pl
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
    train: pl.LazyFrame | pl.DataFrame | pd.DataFrame
    val: pl.LazyFrame | pl.DataFrame | pd.DataFrame
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


def _scan_single_parquet_polars(
    path: Path,
    feature_columns: Sequence[str],
) -> tuple[pl.LazyFrame, bool, bool]:
    schema = pq.read_schema(path)
    available = set(schema.names)

    required = {"card_id", "elapsed_days", *feature_columns}
    missing = sorted(required - available)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{path} missing required columns: {missing_text}")

    inferred_user_id = _infer_user_id(path)
    if "user_id" not in available and inferred_user_id is None:
        raise ValueError(
            f"{path} has no 'user_id' column and filename does not encode user_id."
        )

    user_expr: pl.Expr
    if "user_id" in available:
        if inferred_user_id is None:
            user_expr = pl.col("user_id")
        else:
            user_expr = pl.coalesce([pl.col("user_id"), pl.lit(inferred_user_id)])
    else:
        user_expr = pl.lit(inferred_user_id, dtype=pl.Int64)

    exprs: list[pl.Expr] = [
        user_expr.alias("user_id"),
        pl.col("card_id"),
        pl.col("elapsed_days"),
    ]
    exprs.extend(pl.col(col) for col in feature_columns)

    has_review_th = "review_th" in available
    has_day_offset = "day_offset" in available
    exprs.append((pl.col("review_th") if has_review_th else pl.lit(None)).alias("review_th"))
    exprs.append((pl.col("day_offset") if has_day_offset else pl.lit(None)).alias("day_offset"))

    return pl.scan_parquet(str(path)).select(exprs), has_review_th, has_day_offset


def _scalar_from_lazyframe(
    frame: pl.LazyFrame,
    *,
    column: str,
) -> Any:
    result = frame.collect(engine="streaming")
    return result.get_column(column)[0]


def _split_cards(
    unique_cards: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> np.ndarray:
    if unique_cards.size == 0:
        raise ValueError("No card_id values found for split.")

    rng = np.random.default_rng(seed)
    shuffled_cards = unique_cards.copy()
    rng.shuffle(shuffled_cards)

    if val_ratio == 0.0 or unique_cards.size == 1:
        return np.array([], dtype=np.int64)

    n_val = int(np.floor(unique_cards.size * val_ratio))
    if n_val <= 0:
        n_val = 1
    if n_val >= unique_cards.size:
        n_val = unique_cards.size - 1
    return shuffled_cards[:n_val]


def _to_obs_act_lazy(
    examples: pl.LazyFrame,
    feature_columns: Sequence[str],
) -> pl.LazyFrame:
    return examples.select(
        pl.col("user_id").cast(pl.Int64),
        pl.col("card_id").cast(pl.Int64),
        pl.concat_list([pl.col(name).cast(pl.Float32) for name in feature_columns]).alias("obs"),
        pl.concat_list([pl.col("act").cast(pl.Float32)]).alias("act"),
    )


def _build_bc_datasets_polars(
    *,
    input_dir: Path,
    files: Sequence[Path],
    val_ratio: float,
    seed: int,
    feature_columns: Sequence[str],
    action_clip_mode: str,
    stream_batch_rows: int | None,
    threads: int | None,
) -> BCDatasetArtifacts:
    if action_clip_mode != "simulator_bound":
        raise ValueError(
            f"Unsupported action_clip_mode '{action_clip_mode}'. "
            "Only 'simulator_bound' is supported."
        )
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")

    if threads is not None and threads > 0:
        # Best-effort override. Polars reads this env var during runtime setup.
        os.environ["POLARS_MAX_THREADS"] = str(int(threads))
    if stream_batch_rows is not None and stream_batch_rows > 0:
        pl.Config.set_streaming_chunk_size(int(stream_batch_rows))

    lazy_frames: list[pl.LazyFrame] = []
    has_review_th = False
    has_day_offset = False
    for path in files:
        lf, file_has_review, file_has_day = _scan_single_parquet_polars(path, feature_columns)
        lazy_frames.append(lf)
        has_review_th = has_review_th or file_has_review
        has_day_offset = has_day_offset or file_has_day

    if not has_review_th and not has_day_offset:
        raise ValueError("Input rows must contain 'review_th' or 'day_offset' for sorting.")

    base = pl.concat(lazy_frames, how="vertical_relaxed")
    rows_input = int(
        _scalar_from_lazyframe(base.select(pl.len().alias("rows_input")), column="rows_input")
    )

    numeric_columns = ["user_id", "card_id", "elapsed_days", *feature_columns]
    cast_exprs: list[pl.Expr] = [
        pl.col(name).cast(pl.Float64, strict=False).alias(name) for name in numeric_columns
    ]
    if has_review_th:
        cast_exprs.append(pl.col("review_th").cast(pl.Float64, strict=False).alias("review_th"))
    if has_day_offset:
        cast_exprs.append(pl.col("day_offset").cast(pl.Float64, strict=False).alias("day_offset"))

    numeric = (
        base.with_columns(cast_exprs)
        .drop_nulls(numeric_columns)
        .with_columns(
            pl.col("user_id").cast(pl.Int64),
            pl.col("card_id").cast(pl.Int64),
            pl.col("elapsed_days").cast(pl.Float32),
            *(pl.col(name).cast(pl.Float32) for name in feature_columns),
        )
    )

    sort_columns = ["user_id", "card_id"]
    if has_review_th:
        sort_columns.append("review_th")
    if has_day_offset:
        sort_columns.append("day_offset")

    labeled = (
        numeric.sort(sort_columns)
        .with_columns(
            pl.col("elapsed_days")
            .shift(-1)
            .over(["user_id", "card_id"])
            .cast(pl.Float32)
            .alias("next_elapsed_days")
        )
    )

    stats_row = labeled.select(
        pl.len().alias("rows_after_numeric_drop"),
        pl.col("next_elapsed_days").is_null().sum().alias("rows_terminal_dropped"),
        pl.col("next_elapsed_days").is_not_null().sum().alias("rows_examples"),
        pl.col("card_id")
        .filter(pl.col("next_elapsed_days").is_not_null())
        .n_unique()
        .alias("cards_examples"),
        pl.col("user_id")
        .filter(pl.col("next_elapsed_days").is_not_null())
        .n_unique()
        .alias("users_examples"),
        pl.col("next_elapsed_days")
        .filter(pl.col("next_elapsed_days").is_not_null())
        .max()
        .alias("global_max_interval"),
    ).collect(engine="streaming").row(0, named=True)

    rows_after_numeric_drop = int(stats_row["rows_after_numeric_drop"])
    rows_terminal_dropped = int(stats_row["rows_terminal_dropped"])
    rows_examples = int(stats_row["rows_examples"])
    cards_examples = int(stats_row["cards_examples"])
    users_examples = int(stats_row["users_examples"])
    global_max_interval_value = stats_row["global_max_interval"]
    if global_max_interval_value is None:
        raise ValueError("No behavior-cloning examples were produced from input rows.")
    global_max_interval = float(global_max_interval_value)
    action_max = compute_action_clip_max(global_max_interval)

    examples = labeled.filter(pl.col("next_elapsed_days").is_not_null()).with_columns(
        pl.col("next_elapsed_days").cast(pl.Float32).alias("act_raw"),
        pl.col("next_elapsed_days").clip(0.0, action_max).cast(pl.Float32).alias("act"),
    )

    quality_row = examples.select(
        pl.any_horizontal([~pl.col(name).is_finite() for name in feature_columns])
        .sum()
        .alias("non_finite_obs"),
        (~pl.col("act").is_finite()).sum().alias("non_finite_act"),
        ((pl.col("act") - pl.col("act_raw")).abs() > 1e-6).sum().alias("clipped_count"),
    ).collect(engine="streaming").row(0, named=True)

    non_finite_obs = int(quality_row["non_finite_obs"])
    if non_finite_obs > 0:
        raise ValueError("Observation matrix contains NaN or Inf.")

    non_finite_act = int(quality_row["non_finite_act"])
    if non_finite_act > 0:
        raise ValueError("Action labels contain NaN or Inf.")

    clipped_count = int(quality_row["clipped_count"])
    unique_cards_series = (
        examples.select(pl.col("card_id").cast(pl.Int64).unique().alias("card_id"))
        .collect(engine="streaming")
        .get_column("card_id")
    )
    unique_cards = np.asarray(unique_cards_series.to_numpy(), dtype=np.int64)
    if unique_cards.size == 0:
        raise ValueError("No card_id values found for split.")

    val_cards = _split_cards(unique_cards, val_ratio=val_ratio, seed=seed)
    val_card_list = [int(v) for v in val_cards.tolist()]
    train = examples.filter(~pl.col("card_id").is_in(val_card_list))
    val = examples.filter(pl.col("card_id").is_in(val_card_list))

    train_frame = _to_obs_act_lazy(train, feature_columns)
    val_frame = _to_obs_act_lazy(val, feature_columns)
    split_row = examples.select(
        (~pl.col("card_id").is_in(val_card_list)).sum().alias("train_rows"),
        (pl.col("card_id").is_in(val_card_list)).sum().alias("val_rows"),
    ).collect(engine="streaming").row(0, named=True)
    train_rows = int(split_row["train_rows"])
    val_rows = int(split_row["val_rows"])
    if train_rows == 0:
        raise ValueError("Train split is empty after card_id split.")

    train_card_set = set(int(v) for v in unique_cards.tolist()) - set(int(v) for v in val_cards.tolist())
    val_card_set = set(int(v) for v in val_cards.tolist())
    report = {
        "input_dir": str(input_dir),
        "input_files": [str(path) for path in files],
        "engine": "polars",
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "rows_input": rows_input,
        "rows_after_numeric_drop": rows_after_numeric_drop,
        "rows_dropped_non_numeric": int(rows_input - rows_after_numeric_drop),
        "rows_terminal_dropped": rows_terminal_dropped,
        "rows_examples": rows_examples,
        "cards_examples": cards_examples,
        "users_examples": users_examples,
        "feature_dim": int(len(feature_columns)),
        "sort_columns": sort_columns,
        "global_max_interval": global_max_interval,
        "action_max": action_max,
        "action_clip_count": clipped_count,
        "action_clip_fraction": float(clipped_count / rows_examples) if rows_examples else 0.0,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_cards": int(len(train_card_set)),
        "val_cards": int(len(val_card_set)),
        "card_overlap_count": int(len(train_card_set.intersection(val_card_set))),
    }
    return BCDatasetArtifacts(train=train_frame, val=val_frame, report=report)


def _build_bc_datasets_pandas_legacy(
    *,
    input_dir: Path,
    files: Sequence[Path],
    val_ratio: float,
    seed: int,
    feature_columns: Sequence[str],
    action_clip_mode: str,
) -> BCDatasetArtifacts:
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
        "engine": "pandas_legacy",
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


def build_bc_datasets(
    input_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 0,
    feature_columns: Sequence[str] = CARD_FEATURE_COLUMNS,
    action_clip_mode: str = "simulator_bound",
    *,
    engine: str = "polars",
    stream_batch_rows: int | None = None,
    threads: int | None = None,
) -> BCDatasetArtifacts:
    """Build train/val BC datasets from processed parquet directory."""

    files = discover_processed_parquet_files(input_dir)
    normalized_engine = str(engine).strip().lower()
    if normalized_engine == "polars":
        return _build_bc_datasets_polars(
            input_dir=input_dir,
            files=files,
            val_ratio=val_ratio,
            seed=seed,
            feature_columns=feature_columns,
            action_clip_mode=action_clip_mode,
            stream_batch_rows=stream_batch_rows,
            threads=threads,
        )
    if normalized_engine in {"pandas", "pandas_legacy"}:
        return _build_bc_datasets_pandas_legacy(
            input_dir=input_dir,
            files=files,
            val_ratio=val_ratio,
            seed=seed,
            feature_columns=feature_columns,
            action_clip_mode=action_clip_mode,
        )
    raise ValueError(
        f"Unsupported engine '{engine}'. Expected one of: polars, pandas_legacy."
    )


def _write_frame_parquet(
    frame: pl.LazyFrame | pl.DataFrame | pd.DataFrame,
    output_path: Path,
) -> None:
    if isinstance(frame, pl.LazyFrame):
        frame.sink_parquet(str(output_path))
        return
    if isinstance(frame, pl.DataFrame):
        frame.write_parquet(str(output_path))
        return
    frame.to_parquet(output_path, index=False)


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

    _write_frame_parquet(artifacts.train, train_output)
    _write_frame_parquet(artifacts.val, val_output)
    report_output.write_text(
        json.dumps(artifacts.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
