"""Evaluate offline policy trajectory by ground-truth replay in simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from src.evaluation.replay_evaluator import evaluate_replay_with_simulator

load_dotenv(override=False)


def _parse_user_ids(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        return None
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground-truth replay evaluation on processed user parquet data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Input processed parquet directory (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval_offline_policy"),
        help="Output directory for replay artifacts.",
    )
    parser.add_argument(
        "--user-ids",
        type=str,
        default=None,
        help="Comma-separated user IDs (e.g. 1,2,5). Default: all users.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Predictor model weights path for RWKVSrsPredictor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Predictor device (cpu/cuda/cuda:0...).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Predictor dtype.",
    )
    parser.add_argument(
        "--cards-per-user",
        type=int,
        default=1000000,
        help="Maximum sampled target cards per user. Large default approximates all.",
    )
    parser.add_argument(
        "--min-target-occurrences",
        type=int,
        default=4,
        help="Minimum occurrences required for target cards.",
    )
    parser.add_argument(
        "--warmup-mode",
        type=str,
        choices=["second", "fifth"],
        default="second",
        help="Warmup cutoff mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = evaluate_replay_with_simulator(
        data_dir=str(args.data_dir),
        predictor_model_path=str(args.model_path),
        predictor_device=str(args.device),
        predictor_dtype=str(args.dtype),
        user_ids=_parse_user_ids(args.user_ids),
        cards_per_user=int(args.cards_per_user),
        min_target_occurrences=int(args.min_target_occurrences),
        warmup_mode=str(args.warmup_mode),
        seed=int(args.seed),
    )

    final_metrics_path = output_dir / "final_metrics.parquet"
    trajectory_path = output_dir / "trajectory.parquet"
    summary_path = output_dir / "run_summary.json"

    replay.final_metrics.to_parquet(final_metrics_path, index=False)
    replay.trajectory.to_parquet(trajectory_path, index=False)
    summary_payload = {
        **replay.summary,
        "final_metrics_path": str(final_metrics_path),
        "trajectory_path": str(trajectory_path),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
