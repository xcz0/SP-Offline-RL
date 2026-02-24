#!/usr/bin/env bash
set -euo pipefail

# End-to-end BC experiment runner:
# 1) Build BC obs/act dataset from processed parquet files
# 2) Train BC policy
# 3) Evaluate checkpoint in simulator mode
# 4) Run ground-truth replay evaluation

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

die() {
  echo "[$(timestamp)] ERROR: $*" >&2
  exit 1
}

# ----------------------------
# Configurable variables
# ----------------------------
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-data}"
BC_TRAIN_PATH="${BC_TRAIN_PATH:-data/bc_train.parquet}"
BC_VAL_PATH="${BC_VAL_PATH:-data/bc_val.parquet}"
BC_REPORT_PATH="${BC_REPORT_PATH:-data/bc_dataset_report.json}"
VAL_RATIO="${VAL_RATIO:-0.1}"

SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-50}"
EPOCH_NUM_STEPS="${EPOCH_NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

SIM_EVAL_ENABLED="${SIM_EVAL_ENABLED:-true}"
SIM_EVAL_EVERY_N="${SIM_EVAL_EVERY_N:-5}"
SIM_EVAL_CARDS_PER_USER="${SIM_EVAL_CARDS_PER_USER:-20}"
SIM_MIN_TARGET_OCCURRENCES="${SIM_MIN_TARGET_OCCURRENCES:-5}"
SIM_WARMUP_MODE="${SIM_WARMUP_MODE:-fifth}"   # second | fifth
REPLAY_WARMUP_MODE="${REPLAY_WARMUP_MODE:-second}"
USER_IDS="${USER_IDS:-}"                      # e.g. "1,2,5"

PREDICTOR_MODEL_PATH="${PREDICTOR_MODEL_PATH:-}"
PREDICTOR_DTYPE="${PREDICTOR_DTYPE:-float32}" # float32 | float16 | bfloat16

EXP_ROOT="${EXP_ROOT:-$ROOT_DIR/experiments/bc_$(date +%Y%m%d_%H%M%S)}"
TRAIN_LOGDIR="$EXP_ROOT/log"
mkdir -p "$EXP_ROOT"

# ----------------------------
# Preflight checks
# ----------------------------
[[ -d ".venv" ]] || die ".venv not found. Please create and install dependencies first."
source .venv/bin/activate

command -v python >/dev/null 2>&1 || die "python not found in PATH."

[[ -d "$PROCESSED_DATA_DIR" ]] || die "PROCESSED_DATA_DIR does not exist: $PROCESSED_DATA_DIR"

if ! compgen -G "$PROCESSED_DATA_DIR/user_id=*.parquet" >/dev/null; then
  die "No processed parquet found under $PROCESSED_DATA_DIR (expected user_id=*.parquet)."
fi

if [[ "$SIM_EVAL_ENABLED" == "true" ]]; then
  [[ -n "$PREDICTOR_MODEL_PATH" ]] || die "PREDICTOR_MODEL_PATH is required when SIM_EVAL_ENABLED=true."
  [[ -f "$PREDICTOR_MODEL_PATH" ]] || die "Predictor model not found: $PREDICTOR_MODEL_PATH"
  python - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("sprwkv") is None:
    sys.exit("sprwkv is required for simulator evaluation. Please install sprwkv first.")
PY
fi

if [[ "$SIM_EVAL_ENABLED" != "true" ]]; then
  die "This script is for full BC experiment flow and requires SIM_EVAL_ENABLED=true."
fi

if [[ "$DEVICE" == "auto" ]]; then
  PREDICTOR_DEVICE="$(python - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
else
  PREDICTOR_DEVICE="$DEVICE"
fi

if [[ -n "$USER_IDS" ]]; then
  SIM_USER_IDS_LIST="[$USER_IDS]"
else
  SIM_USER_IDS_LIST="[]"
fi

log "Experiment root: $EXP_ROOT"
log "Step 1/4: Build BC dataset"
python scripts/build_bc_dataset.py \
  --input-dir "$PROCESSED_DATA_DIR" \
  --output-train "$BC_TRAIN_PATH" \
  --output-val "$BC_VAL_PATH" \
  --output-report "$BC_REPORT_PATH" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" | tee "$EXP_ROOT/build_bc_dataset.stdout.log"

[[ -f "$BC_TRAIN_PATH" ]] || die "BC train parquet not generated: $BC_TRAIN_PATH"
[[ -f "$BC_VAL_PATH" ]] || die "BC val parquet not generated: $BC_VAL_PATH"

export SPRL_BC_TRAIN_PATH="$BC_TRAIN_PATH"
export SPRL_BC_VAL_PATH="$BC_VAL_PATH"
export SPRL_LOGDIR="$TRAIN_LOGDIR"
export SPRL_PREDICTOR_MODEL_PATH="$PREDICTOR_MODEL_PATH"

log "Step 2/4: Train BC policy"
python scripts/train.py \
  algo=bc_il \
  model=mlp_actor \
  data=parquet_sp_bc_train \
  seed="$SEED" \
  device="$DEVICE" \
  train.epoch="$EPOCHS" \
  train.epoch_num_steps="$EPOCH_NUM_STEPS" \
  train.batch_size="$BATCH_SIZE" \
  paths.logdir="$TRAIN_LOGDIR" \
  sim_eval.enabled="$SIM_EVAL_ENABLED" \
  sim_eval.data_dir="$PROCESSED_DATA_DIR" \
  "sim_eval.user_ids=$SIM_USER_IDS_LIST" \
  sim_eval.cards_per_user="$SIM_EVAL_CARDS_PER_USER" \
  sim_eval.min_target_occurrences="$SIM_MIN_TARGET_OCCURRENCES" \
  sim_eval.warmup_mode="$SIM_WARMUP_MODE" \
  sim_eval.eval_every_n_epoch="$SIM_EVAL_EVERY_N" \
  sim_eval.predictor.model_path="$PREDICTOR_MODEL_PATH" \
  sim_eval.predictor.device="$PREDICTOR_DEVICE" \
  sim_eval.predictor.dtype="$PREDICTOR_DTYPE" \
  | tee "$EXP_ROOT/train.stdout.log"

FINAL_METRICS_PATH="$(find "$TRAIN_LOGDIR" -name final_metrics.json -type f | sort | tail -n 1)"
[[ -n "$FINAL_METRICS_PATH" ]] || die "No final_metrics.json found under $TRAIN_LOGDIR"
CHECKPOINT_PATH="$(dirname "$FINAL_METRICS_PATH")/policy.pth"
[[ -f "$CHECKPOINT_PATH" ]] || die "Checkpoint not found: $CHECKPOINT_PATH"

log "Training done. final_metrics: $FINAL_METRICS_PATH"
log "Checkpoint: $CHECKPOINT_PATH"

log "Step 3/4: Evaluate BC checkpoint (sim mode)"
python scripts/eval.py \
  algo=bc_il \
  model=mlp_actor \
  data=parquet_sp_bc_train \
  seed="$SEED" \
  device="$DEVICE" \
  eval_mode=sim \
  checkpoint_path="$CHECKPOINT_PATH" \
  sim_eval.enabled=true \
  sim_eval.data_dir="$PROCESSED_DATA_DIR" \
  "sim_eval.user_ids=$SIM_USER_IDS_LIST" \
  sim_eval.cards_per_user="$SIM_EVAL_CARDS_PER_USER" \
  sim_eval.min_target_occurrences="$SIM_MIN_TARGET_OCCURRENCES" \
  sim_eval.warmup_mode="$SIM_WARMUP_MODE" \
  sim_eval.predictor.model_path="$PREDICTOR_MODEL_PATH" \
  sim_eval.predictor.device="$PREDICTOR_DEVICE" \
  sim_eval.predictor.dtype="$PREDICTOR_DTYPE" \
  | tee "$EXP_ROOT/eval_sim.stdout.log"

log "Step 4/4: Ground-truth replay evaluation"
replay_cmd=(
  python scripts/evaluate_offline_policy.py
  --data-dir "$PROCESSED_DATA_DIR"
  --output-dir "$EXP_ROOT/replay_eval"
  --model-path "$PREDICTOR_MODEL_PATH"
  --device "$PREDICTOR_DEVICE"
  --dtype "$PREDICTOR_DTYPE"
  --cards-per-user "$SIM_EVAL_CARDS_PER_USER"
  --min-target-occurrences "$SIM_MIN_TARGET_OCCURRENCES"
  --warmup-mode "$REPLAY_WARMUP_MODE"
  --seed "$SEED"
)
if [[ -n "$USER_IDS" ]]; then
  replay_cmd+=(--user-ids "$USER_IDS")
fi
"${replay_cmd[@]}" | tee "$EXP_ROOT/eval_replay.stdout.log"

cat <<EOF
============================================================
BC experiment completed.

Artifacts:
- Experiment root: $EXP_ROOT
- BC train data:   $BC_TRAIN_PATH
- BC val data:     $BC_VAL_PATH
- BC report:       $BC_REPORT_PATH
- Final metrics:   $FINAL_METRICS_PATH
- Checkpoint:      $CHECKPOINT_PATH

Logs:
- Build:           $EXP_ROOT/build_bc_dataset.stdout.log
- Train:           $EXP_ROOT/train.stdout.log
- Sim eval:        $EXP_ROOT/eval_sim.stdout.log
- Replay eval:     $EXP_ROOT/eval_replay.stdout.log
============================================================
EOF
