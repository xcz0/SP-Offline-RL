"""CLI for offline policy evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.exceptions import ConfigurationError  # noqa: E402
from src.runners.evaluator import run_evaluation  # noqa: E402
from src.utils.env import load_env_file  # noqa: E402

load_env_file()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    checkpoint = cfg.get("checkpoint_path")
    if not checkpoint:
        raise ConfigurationError(
            "checkpoint_path is required for eval. "
            "Example: python scripts/eval.py checkpoint_path=/path/policy.pth"
        )

    result = run_evaluation(cfg, str(checkpoint))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
