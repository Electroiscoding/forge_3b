#!/usr/bin/env bash
# =============================================================================
# FORGE-3B Pretraining Launch Script
# Target: 16× H100 SXM (80 GB) — RunPod community cloud
# Budget: $450 USD | Rate: $63.17/hr | ETA: ~7.1 hours
# =============================================================================
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Paths — override via env vars for cloud deployments
DATA_DIR="${DATA_DIR:-/workspace/data/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/forge_3b_pretrain}"
WANDB_PROJECT="${WANDB_PROJECT:-forge_3b_pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RESUME_FROM="${RESUME_FROM:-}"        # set to a checkpoint path to resume

# Model / Tokenizer
TOKENIZER_PROFILE="${TOKENIZER_PROFILE:-standard}"
MODEL_CONFIG="${MODEL_CONFIG:-}"      # empty → use ForgeModelConfig defaults

# Training
PHASE1_TOKENS="${PHASE1_TOKENS:-5000000000}"     # 5B
PHASE2_TOKENS="${PHASE2_TOKENS:-43000000000}"    # 43B
PHASE3_TOKENS="${PHASE3_TOKENS:-2000000000}"     # 2B
LR_MAX="${LR_MAX:-3e-4}"
BATCH_TOKENS="${BATCH_TOKENS:-2000000}"          # 2M tokens per global step (Phase 2)
MICRO_BATCH="${MICRO_BATCH:-2}"                  # sequences per GPU per step
SAVE_EVERY="${SAVE_EVERY:-2000000000}"           # checkpoint every 2B tokens
LOG_EVERY="${LOG_EVERY:-10}"

# Infrastructure
NUM_GPUS="${NUM_GPUS:-16}"
DS_CONFIG="${DS_CONFIG:-./configs/ds_zero3.json}"
SEED="${SEED:-42}"

# ── DeepSpeed / Torch launcher ────────────────────────────────────────────────
# Prefer deepspeed launcher when available (required for ZeRO-3).
if command -v deepspeed &>/dev/null; then
    LAUNCHER="deepspeed --num_gpus=${NUM_GPUS}"
elif command -v torchrun &>/dev/null; then
    echo "[WARNING] deepspeed not found — falling back to torchrun (no ZeRO-3)"
    LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500"
else
    echo "[ERROR] Neither deepspeed nor torchrun found. Install deepspeed first."
    exit 1
fi

# ── Logging setup ─────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/pretrain_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Logging to $LOG_FILE"

# ── Sanity checks ─────────────────────────────────────────────────────────────
# if [ ! -d "$DATA_DIR" ]; then
#     echo "[ERROR] DATA_DIR does not exist: $DATA_DIR"
#     echo "        Run data preprocessing first:"
#     echo "        python -m data.preprocessing --all --raw_base /data/raw --out_base $DATA_DIR"
#     exit 1
# fi

echo "========================================================================"
echo " FORGE-3B PRETRAINING"
echo "========================================================================"
echo "  Data dir        : $DATA_DIR"
echo "  Output dir      : $OUTPUT_DIR"
echo "  GPUs            : $NUM_GPUS"
echo "  DeepSpeed config: $DS_CONFIG"
echo "  Phase 1 tokens  : $(echo "$PHASE1_TOKENS / 1000000000" | bc)B"
echo "  Phase 2 tokens  : $(echo "$PHASE2_TOKENS / 1000000000" | bc)B"
echo "  Phase 3 tokens  : $(echo "$PHASE3_TOKENS / 1000000000" | bc)B"
echo "  LR max          : $LR_MAX"
echo "  Micro batch     : $MICRO_BATCH seqs/GPU"
echo "  Resume from     : ${RESUME_FROM:-<none>}"
echo "========================================================================"

# ── Launch ────────────────────────────────────────────────────────────────────
${LAUNCHER} run_pretrain.py \
    --data_dir            "$DATA_DIR"          \
    --output_dir          "$OUTPUT_DIR"        \
    --tokenizer_profile   "$TOKENIZER_PROFILE" \
    ${MODEL_CONFIG:+--model_config "$MODEL_CONFIG"} \
    ${RESUME_FROM:+--resume_from   "$RESUME_FROM"}  \
    --phase1_tokens       "$PHASE1_TOKENS"     \
    --phase2_tokens       "$PHASE2_TOKENS"     \
    --phase3_tokens       "$PHASE3_TOKENS"     \
    --lr_max              "$LR_MAX"            \
    --batch_tokens        "$BATCH_TOKENS"      \
    --micro_batch_per_gpu "$MICRO_BATCH"       \
    --save_every_tokens   "$SAVE_EVERY"        \
    --log_every           "$LOG_EVERY"         \
    --deepspeed_config    "$DS_CONFIG"         \
    --wandb_project       "$WANDB_PROJECT"     \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    --seed                "$SEED"              \
    --bf16                                     \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo " PRETRAINING COMPLETE ✓"
    echo " Checkpoints: $OUTPUT_DIR"
    echo " Log:         $LOG_FILE"
    echo "========================================================================"
else
    echo ""
    echo "[ERROR] Training exited with code $EXIT_CODE. Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
