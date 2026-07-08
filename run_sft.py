#!/usr/bin/env python3
"""
FORGE-3B Supervised Fine-Tuning Entry Point.

Usage (16× H100 on RunPod):
    deepspeed --num_gpus=16 run_sft.py \\
        --base_model  /workspace/checkpoints/forge_3b_pretrain/final \\
        --data_dir    /workspace/data/sft \\
        --output_dir  /workspace/checkpoints/forge_3b_sft \\
        --wandb_project forge_3b_sft

Single GPU testing:
    python run_sft.py \\
        --base_model ./checkpoints/forge_3b_pretrain/final \\
        --data_dir   ./data/sft \\
        --output_dir ./checkpoints/forge_3b_sft
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path

import torch

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_sft")


# ── CLI Arguments ─────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FORGE-3B Supervised Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--base_model",  required=True,
                   help="Path to the pretrained model checkpoint (final/ dir)")
    p.add_argument("--data_dir",    required=True,
                   help="Dir containing train.jsonl (and optionally val.jsonl)")
    p.add_argument("--output_dir",  default="./checkpoints/forge_3b_sft")
    p.add_argument("--model_config", default=None,
                   help="Path to model_config.json; defaults to base_model/model_config.json")
    p.add_argument("--resume_from", default=None,
                   help="Resume SFT from a prior SFT checkpoint directory")

    # Training
    p.add_argument("--total_tokens",       type=int, default=1_400_000_000)
    p.add_argument("--lr_max",             type=float, default=1e-5)
    p.add_argument("--lr_min",             type=float, default=1e-6)
    p.add_argument("--seq_len",            type=int, default=4096)
    p.add_argument("--micro_batch_per_gpu", type=int, default=1)
    p.add_argument("--global_batch_tokens", type=int, default=262_144)
    p.add_argument("--grad_clip",          type=float, default=0.5)
    p.add_argument("--weight_decay",       type=float, default=0.0)
    p.add_argument("--loss_on_prompt",     action="store_true",
                   help="Compute loss on prompt tokens (default: assistant tokens only)")
    p.add_argument("--pack_sequences",     action="store_true", default=True,
                   help="Pack short conversations into seq_len windows")

    # Tokenizer
    p.add_argument("--tokenizer_profile", default="standard", choices=["standard", "lite"])

    # Infrastructure
    p.add_argument("--deepspeed_config", default="./configs/ds_zero3_sft.json")
    p.add_argument("--no_compile",       action="store_true")
    p.add_argument("--bf16",             action="store_true", default=True)
    p.add_argument("--save_every_steps", type=int, default=200)
    p.add_argument("--seed",             type=int, default=42)

    # Logging
    p.add_argument("--wandb_project", default="forge_3b_sft")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--log_every",     type=int, default=10)

    return p


# ── Distributed helpers ───────────────────────────────────────────────────────
def _setup_distributed() -> tuple[int, int, int, torch.device]:
    """
    Initialise process group.
    Supports: DeepSpeed launcher (RANK/WORLD_SIZE env vars)
    and plain torch.distributed.launch.
    Returns (rank, world_size, local_rank, device).
    """
    rank       = int(os.environ.get("RANK",       os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # ── Distributed Setup ─────────────────────────────────────────────────────
    rank, world_size, local_rank, device = _setup_distributed()
    is_main = (rank == 0)

    # Per-rank log prefix
    if not is_main:
        logger.setLevel(logging.WARNING)

    torch.manual_seed(args.seed + rank)

    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"FORGE-3B SFT | world_size={world_size} | device={device}")

    # ── Configs ───────────────────────────────────────────────────────────────
    from config import ForgeModelConfig, SFTConfig

    # Load model config from base model dir if not explicitly given
    model_config_path = args.model_config or str(
        Path(args.base_model) / "model_config.json"
    )
    if Path(model_config_path).exists():
        model_config = ForgeModelConfig.from_json(model_config_path)
        logger.info(f"Loaded model config from {model_config_path}")
    else:
        model_config = ForgeModelConfig()
        logger.warning(
            f"model_config.json not found at {model_config_path} — using defaults"
        )

    sft_config = SFTConfig(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        total_tokens=args.total_tokens,
        global_batch_tokens=args.global_batch_tokens,
        micro_batch_size_per_gpu=args.micro_batch_per_gpu,
        seq_len=args.seq_len,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        loss_on_prompt=args.loss_on_prompt,
        save_every_n_steps=args.save_every_steps,
        wandb_project=args.wandb_project,
        deepspeed_config=args.deepspeed_config,
        num_gpus=world_size,
        bf16=args.bf16,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"Loading CRAYON tokenizer (profile={args.tokenizer_profile})...")
    from tokenizer.crayon_wrapper import ForgeTokenizer

    tok_dir = str(Path(args.base_model) / "tokenizer")
    if Path(tok_dir).exists():
        tokenizer = ForgeTokenizer.from_pretrained(tok_dir)
        logger.info(f"Tokenizer loaded from {tok_dir}")
    else:
        tokenizer = ForgeTokenizer(
            profile=args.tokenizer_profile,
            device="cpu",
            n_workers=max(1, torch.multiprocessing.cpu_count() // 2),
            max_length=args.seq_len,
        )
        logger.warning(f"No tokenizer dir at {tok_dir} — initialised fresh tokenizer")

    model_config.vocab_size = tokenizer.vocab_size

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building FORGE-3B model...")
    from model.forge_model import build_forge_3b

    # Detect if ZeRO Stage 3 is enabled to use deepspeed.zero.Init() context manager (required for tied weights)
    is_zero3 = False
    ds_config = None
    if args.deepspeed_config:
        try:
            with open(args.deepspeed_config) as f:
                import json as _json
                ds_config = _json.load(f)
            if ds_config.get("zero_optimization", {}).get("stage", 0) == 3:
                is_zero3 = True
                # Patch gradient_accumulation_steps and micro_batch in ds_config if they are "auto"
                if ds_config.get("gradient_accumulation_steps") == "auto":
                    ds_config["gradient_accumulation_steps"] = sft_config.gradient_accumulation_steps
                if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
                    ds_config["train_micro_batch_size_per_gpu"] = sft_config.micro_batch_size_per_gpu
        except Exception as e:
            logger.warning(f"Failed to check DeepSpeed config stage: {e}")

    if is_zero3 and ds_config is not None:
        import deepspeed
        logger.info("ZeRO-3 detected — wrapping model initialization in deepspeed.zero.Init()")
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            model = build_forge_3b(model_config)
    else:
        model = build_forge_3b(model_config)
        model = model.to(device)

    # Load pretrained weights
    logger.info(f"Loading pretrained weights from {args.base_model}...")
    for weight_filename in ("model_bf16.pt", "model.pt", "pytorch_model.bin"):
        weight_path = Path(args.base_model) / weight_filename
        if weight_path.exists():
            state_dict = torch.load(str(weight_path), map_location="cpu")
            # Handle BF16 → current dtype conversion
            state_dict = {
                k: v.to(torch.bfloat16 if args.bf16 else torch.float32)
                for k, v in state_dict.items()
            }
            if is_zero3:
                from deepspeed.zero import GatheredParameters
                with GatheredParameters(list(model.parameters()), modifier_rank=0):
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
            else:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                
            if missing:
                logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            del state_dict
            torch.cuda.empty_cache()
            logger.info(f"Pretrained weights loaded from {weight_path}")
            break
    else:
        raise FileNotFoundError(
            f"No model weights found in {args.base_model}. "
            "Expected one of: model_bf16.pt, model.pt, pytorch_model.bin"
        )

    # Resume SFT checkpoint (weights override if resuming)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        sft_ckpt = resume_path / "model.pt"
        if sft_ckpt.exists():
            sd = torch.load(str(sft_ckpt), map_location="cpu")
            if is_zero3:
                from deepspeed.zero import GatheredParameters
                with GatheredParameters(list(model.parameters()), modifier_rank=0):
                    model.load_state_dict(sd, strict=False)
            else:
                model.load_state_dict(sd, strict=False)
            logger.info(f"SFT resume weights loaded from {sft_ckpt}")

    n_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params / 1e9:.3f}B parameters")

    # Gradient checkpointing during SFT (critical — seq=4096 is memory-hungry)
    if model_config.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading SFT dataset...")
    from data.dataset import build_dataloader

    # Auto-detect format: pre-tokenized .npz shards vs raw JSONL
    npz_files = list(Path(args.data_dir).glob("**/*.npz"))
    train_jsonl = Path(args.data_dir) / "train.jsonl"

    if npz_files:
        # Pre-tokenized format (from Phase-Technologies/forge-3b-sft-data)
        logger.info(f"Found {len(npz_files)} .npz shards — using PackedSFTDataset")
        from training.sft_engine import PackedSFTDataset

        train_dataset = PackedSFTDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            seed=args.seed,
        )
    elif train_jsonl.exists():
        # Raw JSONL format — tokenize on-the-fly
        logger.info("Found train.jsonl — using SFTDataset (live tokenization)")
        from training.sft_engine import SFTDataset

        train_dataset = SFTDataset(
            data_path=str(train_jsonl),
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            loss_on_prompt=args.loss_on_prompt,
            pack_sequences=args.pack_sequences,
        )
    else:
        raise FileNotFoundError(
            f"No SFT data found in {args.data_dir}.\n"
            "Expected either:\n"
            "  - .npz shards with input_ids + loss_mask (pre-tokenized)\n"
            "  - train.jsonl with 'messages' field (raw text)"
        )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=sft_config.micro_batch_size_per_gpu,
        num_workers=4,
        prefetch_factor=2,
        shuffle=True,
        seed=args.seed,
    )
    logger.info(f"Train dataset: {len(train_dataset):,} samples")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    logger.info("Building optimizer for SFT...")
    from training.optimizer import build_optimizer

    optimizer = build_optimizer(
        model=model,
        lr_max=sft_config.lr_max,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=sft_config.weight_decay,
        # During SFT all groups get equal LR — no differential scaling needed
        embedding_lr_mult=1.0,
        ssm_lr_mult=1.0,
        router_lr_mult=1.0,
        use_fused=True,
    )

    # ── LR Scheduler — cosine: lr_max → lr_min over total_tokens ─────────────
    from training.scheduler import CosineWarmupScheduler

    lr_scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        lr_max=sft_config.lr_max,
        lr_min=sft_config.lr_min,
        warmup_tokens=max(1, sft_config.total_tokens // 20),   # 5% warmup
        total_tokens=sft_config.total_tokens,
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if is_main and sft_config.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=sft_config.wandb_project,
                entity=args.wandb_entity,
                name=f"forge3b-sft-{datetime.datetime.now().strftime('%m%d_%H%M')}",
                resume="allow" if args.resume_from else None,
                config={
                    "sft/total_tokens_b":    sft_config.total_tokens / 1e9,
                    "sft/lr_max":            sft_config.lr_max,
                    "sft/lr_min":            sft_config.lr_min,
                    "sft/seq_len":           sft_config.seq_len,
                    "sft/loss_on_prompt":    sft_config.loss_on_prompt,
                    "sft/batch_tokens":      sft_config.global_batch_tokens,
                    "model/base":            args.base_model,
                    "infra/world_size":      world_size,
                    "infra/bf16":            sft_config.bf16,
                },
                tags=["forge-3b", "sft"],
            )
            logger.info(f"WandB run: {wandb_run.url}")
        except ImportError:
            logger.warning("WandB not installed — training without logging")
        except Exception as exc:
            logger.warning(f"WandB init failed: {exc}")

    # Barrier after WandB init
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    # ── SFT Engine ────────────────────────────────────────────────────────────
    logger.info("Initializing SFTEngine...")
    from training.sft_engine import SFTEngine

    engine = SFTEngine(
        model=model,
        tokenizer=tokenizer,
        config=sft_config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        wandb_run=wandb_run,
    )

    if is_main:
        logger.info("=" * 60)
        logger.info("FORGE-3B SUPERVISED FINE-TUNING")
        logger.info(f"  Tokens  : {sft_config.total_tokens / 1e9:.1f}B")
        logger.info(f"  Seq len : {sft_config.seq_len}")
        logger.info(f"  LR      : {sft_config.lr_max:.1e} → {sft_config.lr_min:.1e}")
        logger.info(f"  Budget  : $450  |  Rate: $63.17/hr (16× H100 SXM)")
        logger.info("=" * 60)

    try:
        engine.train(
            train_dataloader=train_loader,
            optimizer=optimizer,
            scheduler=lr_scheduler,
        )
    except KeyboardInterrupt:
        logger.warning("SFT interrupted. Saving emergency checkpoint...")
        ckpt_dir = Path(sft_config.output_dir) / "sft_interrupt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(ckpt_dir / "model.pt"))
        logger.info(f"Emergency checkpoint: {ckpt_dir}")
        raise
    except Exception as exc:
        logger.exception(f"SFT crashed: {exc}")
        raise

    # ── Final Export ──────────────────────────────────────────────────────────
    if is_main:
        final_dir = Path(sft_config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        bf16_state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        torch.save(bf16_state, str(final_dir / "model_bf16.pt"))
        tokenizer.save_pretrained(str(final_dir))
        model_config.to_json(str(final_dir / "model_config.json"))
        logger.info(f"SFT complete. Final model at: {final_dir}")

        if wandb_run is not None:
            wandb_run.finish()

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    logger.info(f"[rank={rank}] SFT run complete.")


if __name__ == "__main__":
    main()
