"""Logger factory for TensorBoard/WandB."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger, WandbLogger


@dataclass(slots=True)
class LoggerArtifacts:
    logger: TensorboardLogger | WandbLogger
    writer: SummaryWriter


def build_logger(
    logger_cfg: Any,
    log_path: str,
    run_name: str,
    resume_id: str | None,
    config_dict: dict[str, Any],
) -> LoggerArtifacts:
    writer = SummaryWriter(log_path)
    writer.add_text("config", str(config_dict))

    if logger_cfg.type == "tensorboard":
        logger = TensorboardLogger(writer)
    elif logger_cfg.type == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=run_name.replace(os.path.sep, "__"),
            run_id=resume_id,
            config=config_dict,
            project=logger_cfg.wandb_project,
        )
        logger.load(writer)
    else:
        raise ValueError(f"Unsupported logger type: {logger_cfg.type}")

    return LoggerArtifacts(logger=logger, writer=writer)
