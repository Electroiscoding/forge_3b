#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FORGE-3B  Full Training Pipeline  (Pretrain → SFT → DPO)
# One command runs all three stages back-to-back.
#
# Usage:
#   bash train_full_pipeline.sh [OPTIONS]
#
# Required:
#   --pretrain_data   Path to tokenized pretrain .npy shards
#   --sft_data        Path to SFT .npz shards directory
#   --dpo_data        Path to DPO .jsonl directory (contains ultrafeedback/train.jsonl etc.)
#
# Optional:
#   --workspace       Root dir for all outputs  (default: /workspace)
#   --num_gpus        Number of GPUs            (default: auto-detect)
#   --micro_batch     Micro batch per GPU        (default: 1)
#   --skip_pretrain   Skip pretrain, load from --pretrain_ckpt instead
#   --skip_sft        Skip SFT, load from --sft_ckpt instead
#   --pretrain_ckpt   Path to existing pretrain final/ dir  (used with --skip_pretrain)
#   --sft_ckpt        Path to existing SFT final/ dir       (used with --skip_sft)
#   --wandb_project   WandB project name        (default: forge_3b)
#   --hf_token        HuggingFace token for uploads
#   --no_compile      Disable torch.compile (faster startup, slower training)
#
# Example — single cheap GPU:
#   bash train_full_pipeline.sh \
#     --pretrain_data /workspace/data/tokenized \
#     --sft_data      /workspace/data/sft \
#     --dpo_data      /workspace/data/dpo \
#     --num_gpus 1 --micro_batch 1
#
# Example — skip pretrain, start from SFT:
#   bash train_full_pipeline.sh \
#     --skip_pretrain \
#     --pretrain_ckpt /workspace/checkpoints/pretrain/final \
#     --sft_data  /workspace/data/sft \
#     --dpo_data  /workspace/data/dpo
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail   # stop on any error

# ── Activate Virtual Environment ──────────────────────────────────────────────
if [[ -f "/opt/venv/bin/activate" ]]; then
  source /opt/venv/bin/activate
elif [[ -f "/workspace/venv/bin/activate" ]]; then
  source /workspace/venv/bin/activate
fi

# ── Disable P2P and IB to prevent single-GPU RCCL deadlocks ───────────────────
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export RCCL_P2P_DISABLE=1
export RCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_CROSS_NIC=0
export NCCL_DEBUG=INFO

# ── Defaults ─────────────────────────────────────────────────────────────────
WORKSPACE="/workspace"
NUM_GPUS=""          # auto-detect below
MICRO_BATCH=1
PRETRAIN_DATA=""
SFT_DATA=""
DPO_DATA=""
PRETRAIN_CKPT=""
SFT_CKPT=""
WANDB_PROJECT="forge_3b"
HF_TOKEN=""
SKIP_PRETRAIN=0
SKIP_SFT=0
NO_COMPILE=""
NO_GC=""
LOG_EVERY=10


# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)       WORKSPACE="$2";       shift 2 ;;
    --num_gpus)        NUM_GPUS="$2";        shift 2 ;;
    --micro_batch)     MICRO_BATCH="$2";     shift 2 ;;
    --pretrain_data)   PRETRAIN_DATA="$2";   shift 2 ;;
    --sft_data)        SFT_DATA="$2";        shift 2 ;;
    --dpo_data)        DPO_DATA="$2";        shift 2 ;;
    --pretrain_ckpt)   PRETRAIN_CKPT="$2";   shift 2 ;;
    --sft_ckpt)        SFT_CKPT="$2";        shift 2 ;;
    --wandb_project)   WANDB_PROJECT="$2";   shift 2 ;;
    --hf_token)        HF_TOKEN="$2";        shift 2 ;;
    --skip_pretrain)   SKIP_PRETRAIN=1;      shift ;;
    --skip_sft)        SKIP_SFT=1;           shift ;;
    --no_compile)      NO_COMPILE="--no_compile"; shift ;;
    --no_gradient_checkpointing) NO_GC="--no_gradient_checkpointing"; shift ;;
    --log_every)       LOG_EVERY="$2";       shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Auto-detect GPU count ─────────────────────────────────────────────────────
if [[ -z "$NUM_GPUS" ]]; then
  NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
  echo "► Auto-detected $NUM_GPUS GPU(s)"
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
PRETRAIN_OUT="$WORKSPACE/checkpoints/forge_3b_pretrain"
SFT_OUT="$WORKSPACE/checkpoints/forge_3b_sft"
DPO_OUT="$WORKSPACE/checkpoints/forge_3b_dpo"
LOGS_DIR="$WORKSPACE/logs"

mkdir -p "$LOGS_DIR"

# ── Patch ds_zero3.json micro_batch ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_CFG="$SCRIPT_DIR/configs/ds_zero3.json"
python3 - <<PYEOF
import json, pathlib
cfg_path = pathlib.Path("$DS_CFG")
cfg = json.loads(cfg_path.read_text())
cfg["train_micro_batch_size_per_gpu"] = $MICRO_BATCH
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"  ds_zero3.json → train_micro_batch_size_per_gpu={$MICRO_BATCH}")
PYEOF

# ── DeepSpeed launcher helper ─────────────────────────────────────────────────
DS_LAUNCH="deepspeed --num_gpus=$NUM_GPUS"

# ── Helper: print stage banner ────────────────────────────────────────────────
banner() {
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: PRETRAIN
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_PRETRAIN -eq 1 ]]; then
  banner "STAGE 1: PRETRAIN — SKIPPED"
  if [[ -z "$PRETRAIN_CKPT" ]]; then
    echo "  ERROR: --skip_pretrain requires --pretrain_ckpt <path>"
    exit 1
  fi
  PRETRAIN_FINAL="$PRETRAIN_CKPT"
  echo "  Using pretrain checkpoint: $PRETRAIN_FINAL"
else
  banner "STAGE 1: PRETRAIN  (data: $PRETRAIN_DATA)"
  if [[ -z "$PRETRAIN_DATA" ]]; then
    echo "ERROR: --pretrain_data is required for pretrain stage"
    exit 1
  fi

  $DS_LAUNCH "$SCRIPT_DIR/run_pretrain.py" \
    --data_dir          "$PRETRAIN_DATA" \
    --output_dir        "$PRETRAIN_OUT" \
    --micro_batch_per_gpu "$MICRO_BATCH" \
    --num_gpus          "$NUM_GPUS" \
    --wandb_project     "${WANDB_PROJECT}_pretrain" \
    --deepspeed_config  "$DS_CFG" \
    --log_every         "$LOG_EVERY" \
    $NO_COMPILE \
    $NO_GC \
    2>&1 | tee "$LOGS_DIR/pretrain.log"

  PRETRAIN_FINAL="$PRETRAIN_OUT/final"
  echo ""
  echo "✅ Pretrain complete → $PRETRAIN_FINAL"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: SFT
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SKIP_SFT -eq 1 ]]; then
  banner "STAGE 2: SFT — SKIPPED"
  if [[ -z "$SFT_CKPT" ]]; then
    echo "  ERROR: --skip_sft requires --sft_ckpt <path>"
    exit 1
  fi
  SFT_FINAL="$SFT_CKPT"
  echo "  Using SFT checkpoint: $SFT_FINAL"
else
  banner "STAGE 2: SFT  (base: $PRETRAIN_FINAL)"
  if [[ -z "$SFT_DATA" ]]; then
    echo "ERROR: --sft_data is required for SFT stage"
    exit 1
  fi

  $DS_LAUNCH "$SCRIPT_DIR/run_sft.py" \
    --base_model    "$PRETRAIN_FINAL" \
    --data_dir      "$SFT_DATA" \
    --output_dir    "$SFT_OUT" \
    --wandb_project "${WANDB_PROJECT}_sft" \
    --deepspeed_config "$DS_CFG" \
    --log_every        "$LOG_EVERY" \
    $NO_COMPILE \
    $NO_GC \
    2>&1 | tee "$LOGS_DIR/sft.log"

  SFT_FINAL="$SFT_OUT/final"
  echo ""
  echo "✅ SFT complete → $SFT_FINAL"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: DPO
# ─────────────────────────────────────────────────────────────────────────────
banner "STAGE 3: DPO  (base: $SFT_FINAL)"
if [[ -z "$DPO_DATA" ]]; then
  echo "ERROR: --dpo_data is required for DPO stage"
  exit 1
fi

# run_dpo.py now accepts a directory and auto-merges JSONL files internally
$DS_LAUNCH "$SCRIPT_DIR/run_dpo.py" \
  --base_model    "$SFT_FINAL" \
  --data_path     "$DPO_DATA" \
  --output_dir    "$DPO_OUT" \
  --wandb_project "${WANDB_PROJECT}_dpo" \
  --deepspeed_config "$DS_CFG" \
  $NO_COMPILE \
  $NO_GC \
  2>&1 | tee "$LOGS_DIR/dpo.log"

DPO_FINAL="$DPO_OUT/final"

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
banner "ALL STAGES COMPLETE ✅"
echo "  Pretrain : $PRETRAIN_FINAL"
echo "  SFT      : $SFT_FINAL"
echo "  DPO      : $DPO_FINAL"
echo "  Logs     : $LOGS_DIR/"
echo ""
echo "  Final model ready at: $DPO_FINAL"
