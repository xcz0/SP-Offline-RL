"""CLI for offline training."""

from __future__ import annotations

import json

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.runners import train

load_dotenv(override=False)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    result = train(cfg)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
