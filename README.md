# SP-Offline-RL

模块化离线强化学习项目（连续控制），支持 `Parquet` 数据源与 BC 仿真评估流程。

## 目录结构

- `src/`: 核心实现（算法、模型、数据、训练/评估流程）
- `configs/`: Hydra 分层配置
- `scripts/`: 命令行入口（`train.py` / `eval.py` / `build_bc_dataset.py` / `evaluate_offline_policy.py`）
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

构建 BC 训练/验证数据（`obs/act`）：

```bash
python scripts/build_bc_dataset.py --input-dir data
```

评估 checkpoint：

```bash
python scripts/eval.py checkpoint_path=/abs/path/policy.pth
```

BC 仿真评估（`eval_mode=sim`）：

```bash
python scripts/eval.py algo=bc_il model=mlp_actor data=parquet_sp_bc_train eval_mode=sim checkpoint_path=/abs/path/policy.pth
```

离线真值回放评估：

```bash
python scripts/evaluate_offline_policy.py --data-dir data --user-ids 1,2
```

## 环境变量（`.env`）

项目在 `scripts/train.py` 与 `scripts/eval.py` 启动时会自动加载 `.env`（不覆盖已有系统环境变量）。

可用变量：

- `SPRL_DATA_PATH`: 对应 `configs/data/parquet_sp.yaml:path`
- `SPRL_LOGDIR`: 对应 `configs/config.yaml:paths.logdir`
- `SPRL_WANDB_PROJECT`: 对应 `configs/logger/wandb.yaml:wandb_project`
- `SPRL_CHECKPOINT_PATH`: 对应 `configs/config.yaml:checkpoint_path`
- `SPRL_BC_TRAIN_PATH`: 对应 `configs/data/parquet_sp_bc_train.yaml:path`
- `SPRL_BC_VAL_PATH`: 对应 `configs/data/parquet_sp_bc_val.yaml:path`
- `SPRL_PREDICTOR_MODEL_PATH`: 对应 `configs/sim_eval/default.yaml:predictor.model_path`

示例：

```dotenv
SPRL_DATA_PATH=data/offline_dataset.parquet
SPRL_LOGDIR=log
SPRL_WANDB_PROJECT=offline_rl
SPRL_CHECKPOINT_PATH=
SPRL_BC_TRAIN_PATH=data/bc_train.parquet
SPRL_BC_VAL_PATH=data/bc_val.parquet
SPRL_PREDICTOR_MODEL_PATH=/abs/path/predictor_weights.pth
```

## Parquet 数据约定

按算法区分字段要求：

- `bc_il`（行为克隆）只要求：
  - `obs`: 向量（list/array<float>）
  - `act`: 向量（list/array<float>）
- `td3_bc` 要求完整离线 RL 字段：
  - `obs`: 向量（list/array<float>）
  - `act`: 向量（list/array<float>）
  - `rew`: `float`
  - `done`: `bool`
  - `obs_next`: 向量（list/array<float>）
  - `terminated`: `bool`
  - `truncated`: `bool`

说明：
- `td3_bc` 下，`terminated` / `truncated` 必须在 `configs/data/parquet_sp.yaml` 中映射到真实列。
- 向量列需要是固定长度或等长 list；不支持不规则长度向量列。

## BC 仿真评估

- 配置在 `configs/sim_eval/default.yaml`：
  - 每 `5` 个 epoch 触发一次周期评估
  - 每用户抽样 `20` 张目标卡
  - `warmup_mode` 支持 `second` / `fifth`（默认 `fifth`）
- 启用 `sim` / `replay` 评估前需要确保运行环境可导入 `sprwkv`。
- 训练中会基于综合分数选 best checkpoint：  
  `0.5*retention_area + 0.3*final_retention - 0.2*review_count_norm`

## 测试

```bash
source .venv/bin/activate
python -m pytest -q
```
