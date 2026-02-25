"""CLI for offline policy evaluation."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from src.core.exceptions import ConfigurationError
from src.runners import evaluate
from src.utils.env import load_env_file

load_env_file()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    eval_mode = str(cfg.get("eval_mode", "auto")).lower()
    checkpoint = cfg.get("checkpoint_path")
    if eval_mode != "replay" and not checkpoint:
        raise ConfigurationError(
            "checkpoint_path is required for eval unless eval_mode=replay. "
            "Example: python scripts/eval.py checkpoint_path=/path/policy.pth"
        )

    result = evaluate(cfg, str(checkpoint or ""))
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
