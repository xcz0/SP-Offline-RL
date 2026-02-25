"""Manual performance benchmark for train/eval pipelines.

Usage:
    source .venv/bin/activate
    python tests/perf/benchmark_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.runners import evaluate, train

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from factories.config_factory import (  # noqa: E402
    BC_IL_ALGO_CFG,
    TD3_BC_ALGO_CFG,
    build_train_cfg,
    mlp_actor_cfg,
    mlp_actor_critic_cfg,
)
from factories.dataset_factory import write_offline_parquet  # noqa: E402


class _SPRLTestEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 5) -> None:
        self._max_steps = int(max_steps)
        self._step_count = 0
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        clipped_action = np.asarray(action, dtype=np.float32).reshape(1)
        obs = np.array(
            [
                min(self._step_count / float(self._max_steps), 1.0),
                float(np.clip(clipped_action[0], -1.0, 1.0)),
                0.0,
            ],
            dtype=np.float32,
        )
        reward = float(1.0 - abs(obs[1]))
        terminated = self._step_count >= self._max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}


if "SPRLTestEnv-v0" not in gym.registry:
    gym.register(id="SPRLTestEnv-v0", entry_point=_SPRLTestEnv, max_episode_steps=5)


def _run_once(
    *,
    algo_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sprl-perf-") as tmp_dir:
        root = Path(tmp_dir)
        dataset_path = root / "offline.parquet"
        write_offline_parquet(dataset_path, n=n_samples, seed=seed)

        cfg = build_train_cfg(root, dataset_path)
        cfg.algo = dict(algo_cfg)
        cfg.model = dict(model_cfg)
        cfg.perf.enabled = True
        cfg.train.epoch = 3
        cfg.train.epoch_num_steps = 500
        cfg.train.batch_size = 256

        started_train = time.perf_counter()
        train_result = train(cfg)
        train_seconds = time.perf_counter() - started_train

        started_eval = time.perf_counter()
        eval_result = evaluate(cfg, train_result.checkpoint_path)
        eval_seconds = time.perf_counter() - started_eval

        return {
            "algo": str(cfg.algo.name),
            "samples": int(n_samples),
            "train_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
            "train_metrics": train_result.to_dict(),
            "eval_metrics": eval_result.to_dict(),
        }


def main() -> None:
    report = {
        "bc_il": _run_once(
            algo_cfg=BC_IL_ALGO_CFG,
            model_cfg=mlp_actor_cfg(),
            n_samples=4096,
            seed=101,
        ),
        "td3_bc": _run_once(
            algo_cfg=TD3_BC_ALGO_CFG,
            model_cfg=mlp_actor_critic_cfg(),
            n_samples=4096,
            seed=102,
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
