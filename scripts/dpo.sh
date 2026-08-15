#!/usr/bin/env bash
# =============================================================================
# FORGE-1B DPO (Direct Preference Optimization) Launch Script
# Runs AFTER SFT is complete (uses the sft/final/ checkpoint).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-/workspace/checkpoints/forge_1b_sft/final}"
DATA_PATH="${DATA_PATH:-/workspace/data/dpo/preferences.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/forge_1b_dpo}"
RESUME_FROM="${RESUME_FROM:-}"

WANDB_PROJECT="${WANDB_PROJECT:-forge_1b_dpo}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ── DPO hyperparams ───────────────────────────────────────────────────────────
BETA="${BETA:-0.1}"
LOSS_TYPE="${LOSS_TYPE:-dpo}"        # dpo | ipo | cdpo
N_EPOCHS="${N_EPOCHS:-1}"
BATCH_PAIRS="${BATCH_PAIRS:-8}"     # preference pairs per GPU step
GA_STEPS="${GA_STEPS:-4}"           # gradient accumulation steps
SEQ_LEN="${SEQ_LEN:-4096}"
LR="${LR:-5e-7}"
GRAD_CLIP="${GRAD_CLIP:-0.3}"
SAVE_EVERY="${SAVE_EVERY:-100}"

# ── Infrastructure (32x H100 GPU cluster) ────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-32}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"
DS_CONFIG="${DS_CONFIG:-./configs/ds_zero3_sft.json}"
SEED="${SEED:-42}"
TOKENIZER_PROFILE="${TOKENIZER_PROFILE:-standard}"

# ── Launcher ──────────────────────────────────────────────────────────────────
if command -v torchrun &>/dev/null; then
    if [ "$NNODES" -gt 1 ]; then
        GPUS_PER_NODE=$(( NUM_GPUS / NNODES ))
        LAUNCHER="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
    else
        LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT}"
    fi
elif command -v deepspeed &>/dev/null; then
    LAUNCHER="deepspeed --num_gpus=${NUM_GPUS}"
else
    echo "[ERROR] Neither torchrun nor deepspeed found."
    exit 1
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -d "$BASE_MODEL" ]; then
    echo "[ERROR] BASE_MODEL not found: $BASE_MODEL"
    echo "        Run SFT first: bash scripts/sft.sh"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "[ERROR] DPO preference data not found: $DATA_PATH"
    echo "        Provide a JSONL with fields: prompt (list), chosen (str), rejected (str)"
    exit 1
fi

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/dpo_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo " FORGE-3B DIRECT PREFERENCE OPTIMIZATION"
echo "========================================================================"
echo "  Base model   : $BASE_MODEL"
echo "  Data path    : $DATA_PATH"
echo "  Output dir   : $OUTPUT_DIR"
echo "  GPUs         : $NUM_GPUS"
echo "  Beta         : $BETA"
echo "  Loss type    : $LOSS_TYPE"
echo "  LR           : $LR (constant)"
echo "  Epochs       : $N_EPOCHS"
echo "========================================================================"

# ── Launch ────────────────────────────────────────────────────────────────────
${LAUNCHER} run_dpo.py \
    --base_model        "$BASE_MODEL"        \
    --data_path         "$DATA_PATH"         \
    --output_dir        "$OUTPUT_DIR"        \
    --tokenizer_profile "$TOKENIZER_PROFILE" \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"} \
    --beta              "$BETA"              \
    --loss_type         "$LOSS_TYPE"         \
    --n_epochs          "$N_EPOCHS"          \
    --batch_pairs       "$BATCH_PAIRS"       \
    --ga_steps          "$GA_STEPS"          \
    --seq_len           "$SEQ_LEN"           \
    --lr                "$LR"               \
    --grad_clip         "$GRAD_CLIP"         \
    --deepspeed_config  "$DS_CONFIG"         \
    --save_every_steps  "$SAVE_EVERY"        \
    --wandb_project     "$WANDB_PROJECT"     \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    --seed              "$SEED"              \
    --bf16                                   \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo " DPO COMPLETE ✓  |  Final model: $OUTPUT_DIR/final"
    echo "========================================================================"
else
    echo "[ERROR] DPO exited with code $EXIT_CODE. Log: $LOG_FILE"
    exit $EXIT_CODE
fi
