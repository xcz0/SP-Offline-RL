"""CLI to build behavior-cloning datasets from processed review logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.bc_dataset_builder import build_bc_datasets, write_bc_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build behavior-cloning train/val parquet files with obs/act columns "
            "from processed spaced-repetition parquet data."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing processed parquet files (default: data).",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        default=Path("data/bc_train.parquet"),
        help="Output parquet path for train split.",
    )
    parser.add_argument(
        "--output-val",
        type=Path,
        default=Path("data/bc_val.parquet"),
        help="Output parquet path for validation split.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/bc_dataset_report.json"),
        help="Output JSON path for dataset stats report.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio at card_id level (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for card_id split (default: 0).",
    )
    parser.add_argument(
        "--action-clip-mode",
        type=str,
        choices=["simulator_bound"],
        default="simulator_bound",
        help="Action clipping rule. Only simulator_bound is supported currently.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["polars", "pandas_legacy"],
        default="polars",
        help="Dataset build engine.",
    )
    parser.add_argument(
        "--stream-batch-rows",
        type=int,
        default=None,
        help="Optional streaming batch rows for polars engine.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional thread count hint for polars engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_bc_datasets(
        input_dir=args.input_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        action_clip_mode=args.action_clip_mode,
        engine=args.engine,
        stream_batch_rows=args.stream_batch_rows,
        threads=args.threads,
    )
    write_bc_artifacts(
        artifacts=artifacts,
        train_output=args.output_train,
        val_output=args.output_val,
        report_output=args.output_report,
    )
    print(json.dumps(artifacts.report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
