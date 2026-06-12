#!/usr/bin/env python3
"""
FORGE-3B Pretraining Entry Point.

Usage (2× H100 on RunPod):
    deepspeed --num_gpus=2 run_pretrain.py \
        --data_dir /workspace/data/tokenized \
        --output_dir /workspace/checkpoints/forge_3b \
        --wandb_project forge_3b_pretrain

Single GPU testing:
    python run_pretrain.py --data_dir ./data --output_dir ./ckpt --num_gpus 1
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path

import torch

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"pretrain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FORGE-3B Pretraining")
    
    # Paths
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/forge_3b_pretrain")
    parser.add_argument("--resume_from", type=str, default=None)
    
    # Model
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to model config JSON (defaults to FORGE-3B default)")
    parser.add_argument("--tokenizer_profile", type=str, default="standard",
                        choices=["standard", "lite"])
    
    # Training
    parser.add_argument("--phase1_tokens", type=int, default=5_000_000_000)
    parser.add_argument("--phase2_tokens", type=int, default=43_000_000_000)
    parser.add_argument("--phase3_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--batch_tokens", type=int, default=2_000_000)
    parser.add_argument("--micro_batch_per_gpu", type=int, default=2)
    
    # GPU
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default="./configs/ds_zero3.json")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="forge_3b_pretrain")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_tokens", type=int, default=2_000_000_000)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ── Distributed Setup ─────────────────────────────────────────────────────
    from training.gpu_optimizer import setup_distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main:
        logger.info("=" * 70)
        logger.info("FORGE-3B PRETRAINING")
        logger.info(f"  Rank: {rank}/{world_size}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Data: {args.data_dir}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info("=" * 70)
    
    # ── Configs ───────────────────────────────────────────────────────────────
    from config import ForgeModelConfig, PretrainConfig
    
    if args.model_config:
        model_config = ForgeModelConfig.from_json(args.model_config)
    else:
        model_config = ForgeModelConfig()
    
    train_config = PretrainConfig(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        resume_from_checkpoint=args.resume_from,
        phase1_tokens=args.phase1_tokens,
        phase2_tokens=args.phase2_tokens,
        phase3_tokens=args.phase3_tokens,
        lr_max=args.lr_max,
        phase2_global_batch_tokens=args.batch_tokens,
        micro_batch_size_per_gpu=args.micro_batch_per_gpu,
        num_gpus=world_size,
        bf16=args.bf16,
        torch_compile=not args.no_compile,
        deepspeed_config=args.deepspeed_config,
        log_every_n_steps=args.log_every,
        save_every_n_tokens=args.save_every_tokens,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
    )
    
    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"Loading CRAYON tokenizer (profile={args.tokenizer_profile})...")
    from tokenizer.crayon_wrapper import ForgeTokenizer
    
    tokenizer = ForgeTokenizer(
        profile=args.tokenizer_profile,
        device="cpu",   # CPU is 20× faster for CRAYON
        n_workers=max(1, torch.multiprocessing.cpu_count() // 2),
        max_length=4096,
    )
    
    # Update vocab size in model config
    model_config.vocab_size = tokenizer.vocab_size
    
    if is_main:
        logger.info(f"Tokenizer: {tokenizer}")
        model_config.to_json(str(Path(args.output_dir) / "model_config.json"))
        tokenizer.save_pretrained(str(Path(args.output_dir) / "tokenizer"))
    
    # ── Model ─────────────────────────────────────────────────────────────────