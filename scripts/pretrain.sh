#!/usr/bin/env bash
# =============================================================================
# FORGE-1B Pretraining Launch Script
# Architecture : FORGE-1B (1.06B params, ~850M active per token)
# Target       : 1x H100 SXM 80GB (RunPod) — scalable to 16x H100s
# Token budget : 20B tokens (Chinchilla-optimal for 1B params)
# Cost math    : ~72h × $3.29/hr ≈ $237 pretrain | $26 SFT/DPO = ~$263 total
# Hard budget  : $400 (runs comfortably with $137 buffer)
# Throughput   : target ≥80k tok/s with torch.compile max-autotune + Triton
#
# SCALABLE: set NUM_GPUS=16 for 16× H100 and finish in ~4.5 hours.
# =============================================================================
set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Paths — override via env vars for cloud deployments ───────────────────────
DATA_DIR="${DATA_DIR:-Phase-Technologies/forge-3b-pretrain-data}"  # HF repo or local path
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/forge_1b_pretrain}"
WANDB_PROJECT="${WANDB_PROJECT:-forge_1b_pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RESUME_FROM="${RESUME_FROM:-}"

# ── Model ─────────────────────────────────────────────────────────────────────
TOKENIZER_PROFILE="${TOKENIZER_PROFILE:-standard}"
MODEL_CONFIG="${MODEL_CONFIG:-./configs/forge_1b.json}"   # use 1B config

# ── Training — Chinchilla-optimal for 1B params: 20B tokens ───────────────────
PHASE1_TOKENS="${PHASE1_TOKENS:-0}"               # 0 tokens (start directly on Phase 2 @ seq=2048)
PHASE2_TOKENS="${PHASE2_TOKENS:-18000000000}"    # 18B core pretrain (seq=2048)
PHASE3_TOKENS="${PHASE3_TOKENS:-2000000000}"     # 2B  ctx extension (seq=4096)
LR_MAX="${LR_MAX:-3e-4}"

# ── Batch sizes — tuned for max H100 throughput ───────────────────────────────
# micro_batch=64/24/12 fills H100 80GB VRAM and eliminates Python loop overhead
BATCH_TOKENS="${BATCH_TOKENS:-1048576}"          # 1M tokens per global step (Phase 2)
PHASE1_BATCH="${PHASE1_BATCH:-524288}"           # 512K tokens per step   (Phase 1)
PHASE3_BATCH="${PHASE3_BATCH:-524288}"           # 512K tokens per step   (Phase 3)
MICRO_BATCH="${MICRO_BATCH:-16}"                 # fallback micro batch
PHASE1_MICRO_BATCH="${PHASE1_MICRO_BATCH:-64}"   # 64 seqs × 512 = 32,768 tokens/step
PHASE2_MICRO_BATCH="${PHASE2_MICRO_BATCH:-16}"   # 16 seqs × 2048 = 32,768 tokens/step (32 accum steps)
PHASE3_MICRO_BATCH="${PHASE3_MICRO_BATCH:-8}"    # 8 seqs × 4096 = 32,768 tokens/step (16 accum steps)
SAVE_EVERY="${SAVE_EVERY:-1000000000}"           # checkpoint every 1B tokens
LOG_EVERY="${LOG_EVERY:-1}"                      # log EVERY step (full metric visibility)

# ── Infrastructure ─────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-1}"                        # 1x H100 default; set to 16 for cluster
SEED="${SEED:-42}"

# ── Launcher: prefer torchrun (no DeepSpeed needed for 1B) ────────────────────
if command -v torchrun &>/dev/null; then
    LAUNCHER="torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500"
else
    echo "[ERROR] torchrun not found. Install PyTorch >= 2.0."
    exit 1
fi

# ── Logging setup ─────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/pretrain_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Logging to $LOG_FILE"

echo "========================================================================"
echo " FORGE-1B PRETRAINING"
echo "========================================================================"
echo "  Model config    : $MODEL_CONFIG"
echo "  Data dir        : $DATA_DIR"
echo "  Output dir      : $OUTPUT_DIR"
echo "  GPUs            : $NUM_GPUS"
echo "  Phase 1 tokens  : $(( PHASE1_TOKENS / 1000000000 ))B"
echo "  Phase 2 tokens  : $(( PHASE2_TOKENS / 1000000000 ))B"
echo "  Phase 3 tokens  : $(( PHASE3_TOKENS / 1000000000 ))B"
echo "  Total tokens    : $(( (PHASE1_TOKENS + PHASE2_TOKENS + PHASE3_TOKENS) / 1000000000 ))B"
echo "  Micro batch/GPU : Phase1=$PHASE1_MICRO_BATCH | Phase2=$PHASE2_MICRO_BATCH | Phase3=$PHASE3_MICRO_BATCH"
echo "  Global batch    : ${BATCH_TOKENS} tokens/step (Phase 2)"
echo "  Resume from     : ${RESUME_FROM:-<none>}"
echo "  Budget cap      : \$400"
echo "========================================================================"

# ── Launch ────────────────────────────────────────────────────────────────────
${LAUNCHER} run_pretrain.py \
    --data_dir            "$DATA_DIR"          \
    --output_dir          "$OUTPUT_DIR"        \
    --model_config        "$MODEL_CONFIG"      \
    --tokenizer_profile   "$TOKENIZER_PROFILE" \
    ${RESUME_FROM:+--resume_from   "$RESUME_FROM"}  \
    --phase1_tokens       "$PHASE1_TOKENS"     \
    --phase2_tokens       "$PHASE2_TOKENS"     \
    --phase3_tokens       "$PHASE3_TOKENS"     \
    --lr_max              "$LR_MAX"            \
    --batch_tokens        "$BATCH_TOKENS"      \
    --phase1_batch_tokens "$PHASE1_BATCH"      \
    --phase3_batch_tokens "$PHASE3_BATCH"      \
    --micro_batch_per_gpu "$MICRO_BATCH"       \
    --phase1_micro_batch  "$PHASE1_MICRO_BATCH" \
    --phase2_micro_batch  "$PHASE2_MICRO_BATCH" \
    --phase3_micro_batch  "$PHASE3_MICRO_BATCH" \
    --save_every_tokens   "$SAVE_EVERY"        \
    --log_every           "$LOG_EVERY"         \
    --wandb_project       "$WANDB_PROJECT"     \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    --seed                "$SEED"              \
    --bf16                                     \
    --no_compile                               \
    --num_gpus            "$NUM_GPUS"          \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo " FORGE-1B PRETRAINING COMPLETE ✓"
    echo " Checkpoints: $OUTPUT_DIR"
    echo " Log:         $LOG_FILE"
    echo " Next:        Run scripts/sft.sh then scripts/dpo.sh"
    echo "========================================================================"
else
    echo ""
    echo "[ERROR] Training exited with code $EXIT_CODE. Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
