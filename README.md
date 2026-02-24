# SP-Offline-RL

模块化离线强化学习项目（连续控制），当前仅支持 `Parquet` 数据源。

## 目录结构

- `src/`: 核心实现（算法、模型、数据、训练/评估流程）
- `configs/`: Hydra 分层配置
- `scripts/`: 命令行入口（`train.py` / `eval.py`）
- `tests/`: 单元测试与集成测试

## 运行前提

直接使用 `python` 命令前，必须先激活虚拟环境：

```bash
source .venv/bin/activate
```

## 快速开始

训练（默认配置：`td3_bc + parquet_sp + Pendulum-v1`）：

```bash
python scripts/train.py
```

切换为行为克隆：

```bash
python scripts/train.py algo=bc_il model=mlp_actor
```

评估 checkpoint：

```bash
python scripts/eval.py checkpoint_path=/abs/path/policy.pth
```

## 环境变量（`.env`）

项目在 `scripts/train.py` 与 `scripts/eval.py` 启动时会自动加载 `.env`（不覆盖已有系统环境变量）。

可用变量：

- `SPRL_DATA_PATH`: 对应 `configs/data/parquet_sp.yaml:path`
- `SPRL_LOGDIR`: 对应 `configs/config.yaml:paths.logdir`
- `SPRL_WANDB_PROJECT`: 对应 `configs/logger/wandb.yaml:wandb_project`
- `SPRL_CHECKPOINT_PATH`: 对应 `configs/config.yaml:checkpoint_path`

示例：

```dotenv
SPRL_DATA_PATH=data/offline_dataset.parquet
SPRL_LOGDIR=log
SPRL_WANDB_PROJECT=offline_rl
SPRL_CHECKPOINT_PATH=
```

## Parquet 数据约定

必须包含以下 canonical 字段：

- `obs`: 向量（list/array<float>）
- `act`: 向量（list/array<float>）
- `rew`: `float`
- `done`: `bool`
- `obs_next`: 向量（list/array<float>）
- `terminated`: `bool`
- `truncated`: `bool`

说明：
- `terminated` / `truncated` 必须在 `configs/data/parquet_sp.yaml` 中映射到真实列。
- 向量列需要是固定长度或等长 list；不再支持不规则长度向量列。

## 测试

```bash
source .venv/bin/activate
python -m pytest -q
```
