#!/usr/bin/env bash
# =============================================================================
# FORGE-3B Supervised Fine-Tuning Launch Script
# Runs AFTER pretraining is complete (uses the final/ checkpoint).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-/workspace/checkpoints/forge_3b_pretrain/final}"
DATA_DIR="${DATA_DIR:-/workspace/data/sft}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/forge_3b_sft}"
RESUME_FROM="${RESUME_FROM:-}"

WANDB_PROJECT="${WANDB_PROJECT:-forge_3b_sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ── Training hyperparams ──────────────────────────────────────────────────────
TOTAL_TOKENS="${TOTAL_TOKENS:-1400000000}"    # 1.4B
LR_MAX="${LR_MAX:-1e-5}"
LR_MIN="${LR_MIN:-1e-6}"
SEQ_LEN="${SEQ_LEN:-4096}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GLOBAL_BATCH_TOKENS="${GLOBAL_BATCH_TOKENS:-262144}"   # 256K tokens
GRAD_CLIP="${GRAD_CLIP:-0.5}"
SAVE_EVERY="${SAVE_EVERY:-200}"              # steps

# ── Infrastructure ────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-16}"
DS_CONFIG="${DS_CONFIG:-./configs/ds_zero3_sft.json}"
SEED="${SEED:-42}"
TOKENIZER_PROFILE="${TOKENIZER_PROFILE:-standard}"

# ── Launcher ──────────────────────────────────────────────────────────────────
if command -v deepspeed &>/dev/null; then
    LAUNCHER="deepspeed --num_gpus=${NUM_GPUS}"
elif command -v torchrun &>/dev/null; then
    echo "[WARNING] deepspeed not found — using torchrun"
    LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --master_port=29501"
else
    echo "[ERROR] Neither deepspeed nor torchrun found."
    exit 1
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -d "$BASE_MODEL" ]; then
    echo "[ERROR] BASE_MODEL not found: $BASE_MODEL"
    echo "        Run pretraining first: bash scripts/pretrain.sh"
    exit 1
fi

TRAIN_JSONL="$DATA_DIR/train.jsonl"
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[ERROR] SFT training file not found: $TRAIN_JSONL"
    echo "        Provide a JSONL file with 'messages' field (role/content pairs)."
    exit 1
fi

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/sft_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo " FORGE-3B SUPERVISED FINE-TUNING"
echo "========================================================================"
echo "  Base model   : $BASE_MODEL"
echo "  Data dir     : $DATA_DIR"
echo "  Output dir   : $OUTPUT_DIR"
echo "  GPUs         : $NUM_GPUS"
echo "  Total tokens : $(echo "$TOTAL_TOKENS / 1000000000" | bc)B"
echo "  LR           : $LR_MAX → $LR_MIN"
echo "  Seq len      : $SEQ_LEN"
echo "========================================================================"

# ── Launch ────────────────────────────────────────────────────────────────────
${LAUNCHER} run_sft.py \
    --base_model            "$BASE_MODEL"          \
    --data_dir              "$DATA_DIR"            \
    --output_dir            "$OUTPUT_DIR"          \
    --tokenizer_profile     "$TOKENIZER_PROFILE"   \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"}   \
    --total_tokens          "$TOTAL_TOKENS"        \
    --lr_max                "$LR_MAX"              \
    --lr_min                "$LR_MIN"              \
    --seq_len               "$SEQ_LEN"             \
    --micro_batch_per_gpu   "$MICRO_BATCH"         \
    --global_batch_tokens   "$GLOBAL_BATCH_TOKENS" \
    --grad_clip             "$GRAD_CLIP"           \
    --deepspeed_config      "$DS_CONFIG"           \
    --save_every_steps      "$SAVE_EVERY"          \
    --wandb_project         "$WANDB_PROJECT"       \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    --seed                  "$SEED"                \
    --bf16                                         \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo " SFT COMPLETE ✓  |  Model: $OUTPUT_DIR/final"
    echo " Next step: bash scripts/dpo.sh"
    echo "========================================================================"
else
    echo "[ERROR] SFT exited with code $EXIT_CODE. Log: $LOG_FILE"
    exit $EXIT_CODE
fi
