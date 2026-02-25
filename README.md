# SP-Offline-RL

模块化离线强化学习项目（连续控制），支持 `Parquet` 离线数据、`BC`（行为克隆）训练，以及基于 `sprwkv` 的仿真评估。

## 项目结构

- `src/`: 核心实现（`algos/`、`models/`、`data/`、`runners/`、`evaluation/` 等）
- `configs/`: Hydra 配置（`algo/`、`data/`、`env/`、`model/`、`logger/`、`sim_eval/`）
- `scripts/`: CLI 入口
- `tests/`: 单元测试、集成测试、性能基准脚本

常用脚本：

- `scripts/train.py`: 训练入口
- `scripts/eval.py`: 模型评估入口（`gym` / `sim` / `replay`）
- `scripts/build_bc_dataset.py`: 从 processed parquet 构建 BC `obs/act` 数据
- `scripts/evaluate_offline_policy.py`: ground-truth replay 评估
- `scripts/run_bc_experiment.sh`: BC 全流程实验脚本（数据构建 + 训练 + 仿真评估 + replay）

## 环境准备

使用项目前，先激活虚拟环境：

```bash
source .venv/bin/activate
```

如果你是首次初始化仓库，建议先安装依赖（任选其一）：

```bash
# 方式 A（推荐，使用 uv）
uv sync --group dev

# 方式 B（pip）
pip install -e .
```

## 配置体系（Hydra）

根配置为 `configs/config.yaml`，默认组合：

- `env=custom_sp_env`（默认 `Pendulum-v1`）
- `algo=td3_bc`
- `model=mlp_actor_critic`
- `data=parquet_sp`
- `sim_eval=default`
- `logger=tensorboard`

常用覆盖方式：

```bash
python scripts/train.py algo=bc_il model=mlp_actor
python scripts/train.py env=halfcheetah_v2 train.epoch=50 train.batch_size=512
python scripts/eval.py eval_mode=gym checkpoint_path=/abs/path/policy.pth
```

关键开关：

- `watch=true`: 只加载 checkpoint 并评估，不做训练
- `eval_mode=auto|gym|sim|replay`: 评估模式
- `data.obs_norm=true|false`: 是否启用观测归一化
- `paths.logdir=...`: 训练输出目录

## 数据准备与格式

### 1) `td3_bc` 所需离线 RL 字段

需要以下 canonical 字段：

- `obs`: 向量（定长 list/array）
- `act`: 向量（定长 list/array）
- `rew`: float
- `done`: bool
- `obs_next`: 向量（定长 list/array）
- `terminated`: bool
- `truncated`: bool

注意：

- 向量列必须是定长；不支持不规则长度 list。
- 如果真实列名不同，请在 `configs/data/parquet_sp.yaml` 的 `columns` 中做映射。

### 2) `bc_il` 所需字段

仅需：

- `obs`
- `act`

可直接使用 `configs/data/parquet_sp_bc_train.yaml` / `parquet_sp_bc_val.yaml`。

### 3) 从 processed parquet 构建 BC 数据

输入目录需包含 `user_id=*.parquet` 文件：

```bash
python scripts/build_bc_dataset.py \
  --input-dir data \
  --output-train data/bc_train.parquet \
  --output-val data/bc_val.parquet \
  --output-report data/bc_dataset_report.json \
  --val-ratio 0.1 \
  --seed 0
```

## 训练使用说明

### 1) 默认训练（TD3-BC）

```bash
python scripts/train.py
```

### 2) BC 训练（不启用仿真）

```bash
python scripts/train.py \
  algo=bc_il \
  model=mlp_actor \
  data=parquet_sp_bc_train \
  sim_eval.enabled=false
```

### 3) BC 训练并启用仿真周期评估

```bash
python scripts/train.py \
  algo=bc_il \
  model=mlp_actor \
  data=parquet_sp_bc_train \
  sim_eval.enabled=true \
  sim_eval.data_dir=data \
  sim_eval.cards_per_user=20 \
  sim_eval.min_target_occurrences=5 \
  sim_eval.warmup_mode=fifth
```

### 4) 使用 WandB 日志

```bash
python scripts/train.py logger=wandb
```

## 评估使用说明

### 1) Gym 环境评估（需 checkpoint）

```bash
python scripts/eval.py \
  eval_mode=gym \
  checkpoint_path=/abs/path/policy.pth
```

### 2) Simulator 策略评估（`sim`，需 checkpoint，当前仅支持 `bc_il`）

```bash
python scripts/eval.py \
  algo=bc_il \
  model=mlp_actor \
  data=parquet_sp_bc_train \
  eval_mode=sim \
  checkpoint_path=/abs/path/policy.pth \
  sim_eval.enabled=true \
  sim_eval.data_dir=data
```

### 3) Replay 评估（`eval_mode=replay`，不需要 checkpoint）

```bash
python scripts/eval.py \
  eval_mode=replay \
  sim_eval.enabled=true \
  sim_eval.data_dir=data
```

或使用更细粒度 CLI：

```bash
python scripts/evaluate_offline_policy.py \
  --data-dir data \
  --output-dir data/eval_offline_policy \
  --user-ids 1,2 \
  --model-path /abs/path/predictor_weights.pth \
  --device cpu \
  --dtype float32
```

## 训练产物说明

默认输出目录为 `paths.logdir`（默认 `log`），结构如下：

```text
log/<task>/<algo>/<seed>/<timestamp>/
  ├── resolved_config.yaml
  ├── policy.pth
  ├── final_metrics.json
  └── sim_eval/                 # 仅 sim/replay 相关评估时生成
```

`final_metrics.json` 中包含：

- `training`: 训练聚合指标
- `evaluation`: 最终评估指标
- `evaluation.extra.perf`: 当 `perf.enabled=true` 时的阶段耗时

## 环境变量（`.env`）

`scripts/train.py`、`scripts/eval.py`、`scripts/evaluate_offline_policy.py` 会自动加载 `.env`（不覆盖已存在系统环境变量）。

可用变量：

- `SPRL_DATA_PATH`
- `SPRL_LOGDIR`
- `SPRL_WANDB_PROJECT`
- `SPRL_CHECKPOINT_PATH`
- `SPRL_BC_TRAIN_PATH`
- `SPRL_BC_VAL_PATH`
- `SPRL_PREDICTOR_MODEL_PATH`

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

## 性能配置与基准

`configs/config.yaml` 中的 `perf` 配置：

- `perf.enabled`: 在 train/eval 结果中记录分阶段耗时
- `perf.eval_workers`: replay evaluator 用户级并发 worker 数
- `perf.profile_steps`: 是否显示训练进度条（默认关闭，适合基准测试）
- `perf.pin_memory` / `perf.prefetch_batches`: 预留性能开关（当前不直接驱动 DataLoader）

运行手工基准：

```bash
python tests/perf/benchmark_pipeline.py
```

## 一键 BC 实验脚本

`scripts/run_bc_experiment.sh` 提供完整流水线：

1. 构建 BC 数据
2. 训练 BC 策略
3. 执行 sim 评估
4. 执行 replay 评估

示例：

```bash
SEED=0 \
PROCESSED_DATA_DIR=data \
PREDICTOR_MODEL_PATH=/abs/path/predictor_weights.pth \
bash scripts/run_bc_experiment.sh
```

## 测试

```bash
source .venv/bin/activate

# 单元测试
python -m pytest -q -m "not integration"

# 集成测试
python -m pytest -q -m integration

# 全量测试
python -m pytest -q
```

## 常见问题

### 1) `checkpoint_path is required`

`eval_mode=gym` 或 `eval_mode=sim` 必须提供 `checkpoint_path`。  
只有 `eval_mode=replay` 可不传 checkpoint。

### 2) `sprwkv is required for simulator-based evaluation`

你正在运行 `sim` 或 `replay` 评估，但环境未安装/无法导入 `sprwkv`。

### 3) Parquet 向量列报错（不等长）

请检查 `obs` / `act` / `obs_next` 是否为固定长度向量列。

### 4) 找不到 processed 用户数据

`build_bc_dataset.py` 和 replay/sim 流程默认读取 `user_id=*.parquet`，请确认文件命名与目录。
