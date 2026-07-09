#!/usr/bin/env python3
"""
FORGE-3B Pretraining Entry Point.

Usage (16× H100 on RunPod):
    deepspeed --num_gpus=16 run_pretrain.py \
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
import torch.nn as nn

# No global class patches on nn.Module (which break torch.compile).
# Instead, we call convert_to_attr_dict(model) post-construction.

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
    parser.add_argument("--num_gpus", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default="./configs/ds_zero3.json")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    
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
    
    if args.no_gradient_checkpointing:
        model_config.use_gradient_checkpointing = False
    
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
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model_config.to_json(str(Path(args.output_dir) / "model_config.json"))
        tokenizer.save_pretrained(str(Path(args.output_dir) / "tokenizer"))
    
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
                    ds_config["gradient_accumulation_steps"] = train_config.gradient_accumulation_steps_phase2
                if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
                    ds_config["train_micro_batch_size_per_gpu"] = train_config.micro_batch_size_per_gpu
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

    # Convert _parameters, _buffers, and _modules of all modules to AttrDict post-construction
    # to support DeepSpeed setting dynamic attributes (e.g. _in_forward) on PyTorch 2.5+.
    # This avoids setting class properties on nn.Module, which breaks torch.compile (Dynamo) tracing.
    from training.gpu_optimizer import convert_to_attr_dict
    convert_to_attr_dict(model)

    n_params_total = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    n_params_trainable = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"Model: {n_params_total / 1e9:.3f}B total params, "
                    f"{n_params_trainable / 1e9:.3f}B trainable")
        # Quick sanity check against expected target
        expected_min = 2.9e9
        expected_max = 3.1e9
        if not (expected_min <= n_params_total <= expected_max):
            logger.warning(
                f"⚠  Parameter count {n_params_total/1e9:.3f}B is outside the expected "
                f"[{expected_min/1e9:.1f}B, {expected_max/1e9:.1f}B] range. "
                f"Check model_config!"
            )

    # Enable gradient checkpointing (saves ~60% activation memory)
    if model_config.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    # ── Checkpoint Resume ─────────────────────────────────────────────────────
    start_tokens = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            logger.error(f"Resume path does not exist: {resume_path}")
            raise FileNotFoundError(resume_path)

        logger.info(f"Resuming from checkpoint: {resume_path}")
        # Load model weights — tolerant of ZeRO-3 consolidated or plain state_dict
        ckpt_model_path = resume_path / "model.safetensors"
        if not ckpt_model_path.exists():
            ckpt_model_path = resume_path / "pytorch_model.bin"

        if ckpt_model_path.exists():
            state_dict = torch.load(str(ckpt_model_path), map_location="cpu")
            if is_zero3:
                from deepspeed.zero import GatheredParameters
                with GatheredParameters(list(model.parameters()), modifier_rank=0):
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
            else:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                
            if missing:
                logger.warning(f"Missing keys in checkpoint ({len(missing)}): {missing[:5]}...")
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint ({len(unexpected)}): {unexpected[:5]}...")
            del state_dict
            torch.cuda.empty_cache()
            logger.info("Model weights loaded from checkpoint")

        # Read token position from metadata so the scheduler resumes correctly
        meta_path = resume_path / "checkpoint_meta.json"
        if meta_path.exists():
            import json as _json
            with open(str(meta_path)) as _f:
                _meta = _json.load(_f)
            start_tokens = _meta.get("tokens_processed", 0)
            logger.info(f"Resuming at {start_tokens / 1e9:.2f}B tokens already processed")

    # ── Datasets — Phase-Aware Weighted Mixing ────────────────────────────────
    #
    # Data layout expected under data_dir/:
    #   {data_dir}/fineweb_edu/   .npy packs
    #   {data_dir}/thestack/      .npy packs
    #   {data_dir}/wikipedia/     .npy packs
    #   {data_dir}/openwebmath/   .npy packs
    #   {data_dir}/books/         .npy packs
    #   {data_dir}/arxiv/         .npy packs
    #   {data_dir}/dolma/         .npy packs
    #   {data_dir}/stackexchange/ .npy packs
    #   {data_dir}/redpajama_cc/  .npy packs
    #   {data_dir}/multilingual/  .npy packs
    #
    # Missing domains are logged as warnings — training continues with what is found.

    from data.dataset import PackedTokenDataset, WeightedDataMixer, build_dataloader

    data_dir = Path(args.data_dir)

    def _try_load_domain(
        domain_subdir: str,
        seq_len: int,
        split: str = "train",
    ) -> "PackedTokenDataset | None":
        """Attempt to load a domain dataset, returning None on failure."""
        domain_path = data_dir / domain_subdir
        if not domain_path.exists():
            logger.warning(f"Domain directory missing — skipping: {domain_path}")
            return None
        try:
            ds = PackedTokenDataset(
                data_dir=str(domain_path),
                seq_len=seq_len,
                split=split,
                seed=train_config.seed + rank,
            )
            logger.info(f"  Loaded '{domain_subdir}': {len(ds):,} sequences")
            return ds
        except Exception as exc:
            logger.warning(f"Failed to load domain '{domain_subdir}': {exc}")
            return None

    # ── Phase 1: Vocabulary Warmup — clean, diverse, short-context ────────────
    # Wikipedia 50% | Books 30% | ArXiv 20%  (README §5.2 Phase 1)
    logger.info("Building Phase-1 dataloader (seq=512, vocab warmup mix)...")
    _p1_domains = {
        "wikipedia": ("wikipedia",  0.50),
        "books":     ("books",      0.30),
        "arxiv":     ("arxiv",      0.20),
    }
    _p1_datasets: dict = {}
    _p1_weights: dict  = {}
    for key, (subdir, weight) in _p1_domains.items():
        ds = _try_load_domain(subdir, seq_len=train_config.phase1_seq_len)
        if ds is not None:
            _p1_datasets[key] = ds
            _p1_weights[key]  = weight

    if not _p1_datasets:
        raise RuntimeError(
            "No Phase-1 domains could be loaded. "
            "Ensure tokenized data exists under --data_dir. "
            "Expected subdirectories: wikipedia/, books/, arxiv/"
        )

    p1_target_seqs = train_config.phase1_tokens // train_config.phase1_seq_len
    phase1_mixer = WeightedDataMixer(
        datasets=_p1_datasets,
        weights=_p1_weights,
        total_samples=p1_target_seqs,
        seed=train_config.seed,
    )

    ga_steps_p1 = max(1, train_config.phase1_global_batch_tokens // (
        train_config.micro_batch_size_per_gpu
        * train_config.phase1_seq_len
        * world_size
    ))
    phase1_loader = build_dataloader(
        dataset=phase1_mixer,
        batch_size=train_config.micro_batch_size_per_gpu,
        num_workers=train_config.num_dataloader_workers,
        prefetch_factor=train_config.prefetch_factor,
        shuffle=True,
        seed=train_config.seed,
    )
    logger.info(
        f"Phase-1 loader ready: {len(phase1_mixer):,} sequences, "
        f"grad_accum={ga_steps_p1}"
    )

    # ── Phase 2: Core Pretraining — full domain mix ────────────────────────────
    # FineWeb-Edu 30% | Stack 16% | Wiki 8% | Math 8% | Books 7% | ArXiv 6%
    # Dolma 10% | StackExchange 5% | RedPajama 6% | Multilingual 4%
    logger.info("Building Phase-2 dataloader (seq=2048, full domain mix)...")
    _p2_domains = {
        "fineweb_edu":   ("fineweb_edu",   0.30),
        "thestack":      ("thestack",      0.16),
        "wikipedia":     ("wikipedia",     0.08),
        "openwebmath":   ("openwebmath",   0.08),
        "books":         ("books",         0.07),
        "arxiv":         ("arxiv",         0.06),
        "dolma":         ("dolma",         0.10),
        "stackexchange": ("stackexchange", 0.05),
        "redpajama_cc":  ("redpajama_cc",  0.06),
        "multilingual":  ("multilingual",  0.04),
    }
    _p2_datasets: dict = {}
    _p2_weights: dict  = {}
    for key, (subdir, weight) in _p2_domains.items():
        ds = _try_load_domain(subdir, seq_len=train_config.phase2_seq_len)
        if ds is not None:
            _p2_datasets[key] = ds
            _p2_weights[key]  = weight

    if not _p2_datasets:
        raise RuntimeError(
            "No Phase-2 domains could be loaded. "
            "Ensure tokenized data exists under --data_dir."
        )

    p2_target_seqs = train_config.phase2_tokens // train_config.phase2_seq_len
    phase2_mixer = WeightedDataMixer(
        datasets=_p2_datasets,
        weights=_p2_weights,
        total_samples=p2_target_seqs,
        seed=train_config.seed,
    )

    phase2_loader = build_dataloader(
        dataset=phase2_mixer,
        batch_size=train_config.micro_batch_size_per_gpu,
        num_workers=train_config.num_dataloader_workers,
        prefetch_factor=train_config.prefetch_factor,
        shuffle=True,
        seed=train_config.seed,
    )
    logger.info(
        f"Phase-2 loader ready: {len(phase2_mixer):,} sequences, "
        f"grad_accum={train_config.gradient_accumulation_steps_phase2}"
    )

    # ── Phase 3: Context Extension — long-doc only (>4096 tokens) ─────────────
    # Re-uses Phase-2 domain mix but filtered at the PackedTokenDataset level.
    # The data pipeline should have written separate long-doc packs during
    # preprocessing. Fall back to Phase-2 mix if dedicated packs are absent.
    logger.info("Building Phase-3 dataloader (seq=4096, long-doc mix)...")
    _p3_long_suffix = "long"   # e.g. fineweb_edu_long/, books_long/, etc.
    _p3_datasets: dict = {}
    _p3_weights: dict  = {}

    for key, (subdir, weight) in _p2_domains.items():
        # Prefer the dedicated long-doc split, fall back to full domain.
        long_subdir = f"{subdir}_{_p3_long_suffix}"
        ds = _try_load_domain(long_subdir, seq_len=train_config.phase3_seq_len)
        if ds is None:
            ds = _try_load_domain(subdir, seq_len=train_config.phase3_seq_len)
        if ds is not None:
            _p3_datasets[key] = ds
            _p3_weights[key]  = weight

    if not _p3_datasets:
        logger.warning(
            "No Phase-3 long-doc packs found — "
            "re-loading Phase-2 domains at seq=4096."
        )
        # Re-load Phase-2 domain directories with the Phase-3 seq_len so the
        # PackedTokenDataset reshapes the raw data to (N, 4096) correctly.
        for key, (subdir, weight) in _p2_domains.items():
            ds = _try_load_domain(subdir, seq_len=train_config.phase3_seq_len)
            if ds is not None:
                _p3_datasets[key] = ds
                _p3_weights[key]  = weight

    p3_target_seqs = train_config.phase3_tokens // train_config.phase3_seq_len
    phase3_mixer = WeightedDataMixer(
        datasets=_p3_datasets,
        weights=_p3_weights,
        total_samples=p3_target_seqs,
        seed=train_config.seed + 3,
    )

    phase3_loader = build_dataloader(
        dataset=phase3_mixer,
        batch_size=train_config.micro_batch_size_per_gpu,
        num_workers=train_config.num_dataloader_workers,
        prefetch_factor=train_config.prefetch_factor,
        shuffle=True,
        seed=train_config.seed + 3,
    )
    logger.info(f"Phase-3 loader ready: {len(phase3_mixer):,} sequences")

    # ── Optimizer — differential parameter groups ─────────────────────────────
    logger.info("Building AdamW optimizer with parameter-group differentiation...")
    from training.optimizer import build_optimizer

    optimizer = build_optimizer(
        model=model,
        lr_max=train_config.lr_max,
        beta1=train_config.beta1,
        beta2=train_config.beta2,
        eps=train_config.eps,
        weight_decay=train_config.weight_decay,
        embedding_lr_mult=train_config.embedding_lr_mult,
        ssm_lr_mult=train_config.ssm_lr_mult,
        router_lr_mult=train_config.router_lr_mult,
        use_fused=True,   # CUDA-fused AdamW — 30-50% faster on H100
    )

    # Per-group learning-rate multipliers for the scheduler (mirrors optimizer groups)
    lr_group_multipliers = {
        "main":       1.0,
        "no_decay":   1.0,
        "embedding":  train_config.embedding_lr_mult,
        "ssm_state":  train_config.ssm_lr_mult,
        "moe_router": train_config.router_lr_mult,
    }

    # ── LR Scheduler — cosine with linear warmup (token-based) ───────────────
    from training.scheduler import CosineWarmupScheduler

    # Effective total tokens spans phases 1+2+3.
    # Warmup covers all of Phase 1 (5B tokens), then cosine decays across Phase 2.
    # Phase 3 uses a constant LR override inside PretrainEngine.
    lr_scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        lr_max=train_config.lr_max,
        lr_min=train_config.lr_min,
        warmup_tokens=train_config.lr_warmup_tokens,   # 5B (full Phase 1)
        total_tokens=train_config.phase1_tokens + train_config.phase2_tokens,
        group_multipliers=lr_group_multipliers,
    )

    # Fast-forward the scheduler to the resume point so the LR is correct.
    if start_tokens > 0:
        logger.info(f"Fast-forwarding LR scheduler to {start_tokens / 1e9:.2f}B tokens...")
        lr_scheduler.step(start_tokens)
        logger.info(f"LR after fast-forward: {lr_scheduler.get_lr():.2e}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if is_main and train_config.wandb_project:
        try:
            import wandb
            run_name = (
                f"forge3b-pretrain-"
                f"{datetime.datetime.now().strftime('%m%d_%H%M')}"
            )
            # Resume an existing run if we are continuing from a checkpoint
            resume_wandb = "allow" if args.resume_from else None

            wandb_run = wandb.init(
                project=train_config.wandb_project,
                entity=train_config.wandb_entity,
                name=run_name,
                resume=resume_wandb,
                config={
                    # Model
                    "model/n_params_total":      n_params_total,
                    "model/n_params_active":     1_174_000_000,
                    "model/d_model":             model_config.d_model,
                    "model/n_layers":            model_config.n_layers,
                    "model/vocab_size":          model_config.vocab_size,
                    "model/n_arg_layers":        model_config.n_layers - len(model_config.mha_layer_indices),
                    "model/n_mha_layers":        len(model_config.mha_layer_indices),
                    "model/arg_d_state":         model_config.arg_d_state,
                    "model/arg_local_window":    model_config.arg_local_window,
                    "model/hse_n_domains":       model_config.hse_n_domains,
                    "model/hse_experts_total":   model_config.hse_n_domains * model_config.hse_n_experts_per_domain,
                    "model/hse_top_k":           model_config.hse_top_k,
                    "model/norm_type":           model_config.norm_type,
                    # Training
                    "train/phase1_tokens_b":     train_config.phase1_tokens / 1e9,
                    "train/phase2_tokens_b":     train_config.phase2_tokens / 1e9,
                    "train/phase3_tokens_b":     train_config.phase3_tokens / 1e9,
                    "train/total_tokens_b":      train_config.total_tokens / 1e9,
                    "train/lr_max":              train_config.lr_max,
                    "train/lr_min":              train_config.lr_min,
                    "train/weight_decay":        train_config.weight_decay,
                    "train/batch_tokens_p2":     train_config.phase2_global_batch_tokens,
                    "train/micro_batch_per_gpu": train_config.micro_batch_size_per_gpu,
                    "train/grad_clip":           train_config.grad_clip,
                    "train/bf16":                train_config.bf16,
                    # Infrastructure
                    "infra/world_size":          world_size,
                    "infra/torch_compile":       train_config.torch_compile,
                    "infra/deepspeed_config":    train_config.deepspeed_config,
                    "infra/resume_from":         args.resume_from or "none",
                    "infra/start_tokens_b":      start_tokens / 1e9,
                },
                tags=["forge-3b", "pretraining", f"gpus={world_size}"],
            )
            logger.info(f"WandB run initialized: {wandb_run.url}")
        except ImportError:
            logger.warning("WandB not installed — training without run tracking. "
                           "Install with: pip install wandb")
        except Exception as exc:
            logger.warning(f"WandB init failed ({exc}) — continuing without WandB")

    # Broadcast WandB run ID to all ranks for shared logging (no-op on non-main)
    if world_size > 1:
        import torch.distributed as _dist
        _dist.barrier()

    # ── Pre-flight Memory Check ────────────────────────────────────────────────
    if torch.cuda.is_available():
        from training.gpu_optimizer import GPUMemoryMonitor
        _mem = GPUMemoryMonitor(device)
        _mem.log_memory(step=0, prefix="pre-training")
        if _mem.check_oom_risk():
            logger.error(
                "OOM risk detected before training even starts. "
                "Reduce micro_batch_size_per_gpu or enable more aggressive "
                "gradient checkpointing."
            )
            raise MemoryError("Pre-flight OOM check failed.")

    # ── PretrainEngine ────────────────────────────────────────────────────────
    logger.info("Initializing PretrainEngine...")
    from training.pretrain_engine import PretrainEngine

    engine = PretrainEngine(
        model=model,
        tokenizer=tokenizer,
        train_config=train_config,
        model_config=model_config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        wandb_run=wandb_run,
    )

    # Carry over the already-processed token count so budget/throughput tracking
    # is correct when resuming from a checkpoint.
    if start_tokens > 0:
        engine._tokens_processed = start_tokens
        logger.info(f"Engine token counter initialised to {start_tokens / 1e9:.2f}B")

    # ── Training ──────────────────────────────────────────────────────────────
    if is_main:
        logger.info("=" * 70)
        logger.info("FORGE-3B PRETRAINING — FULL THREE-PHASE RUN")
        logger.info(f"  Phase 1 : {train_config.phase1_tokens / 1e9:.0f}B tokens | "
                    f"seq={train_config.phase1_seq_len}")
        logger.info(f"  Phase 2 : {train_config.phase2_tokens / 1e9:.0f}B tokens | "
                    f"seq={train_config.phase2_seq_len}")
        logger.info(f"  Phase 3 : {train_config.phase3_tokens / 1e9:.0f}B tokens | "
                    f"seq={train_config.phase3_seq_len}")
        logger.info(f"  Total   : {train_config.total_tokens / 1e9:.0f}B tokens")
        logger.info(f"  Budget  : $450.00  |  Rate: $63.17/hr (16× H100 SXM)")
        logger.info("=" * 70)

    try:
        engine.train(
            phase1_dataloader=phase1_loader,
            phase2_dataloader=phase2_loader,
            phase3_dataloader=phase3_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.info("Saving emergency checkpoint before exit...")
        engine._save_checkpoint(
            phase=0,
            step=engine._global_step,
            tokens=engine._tokens_processed,
        )
        logger.info("Emergency checkpoint saved. Exiting.")
        raise
    except Exception as exc:
        logger.exception(f"Training crashed: {exc}")
        logger.info("Attempting emergency checkpoint save...")
        try:
            engine._save_checkpoint(
                phase=0,
                step=engine._global_step,
                tokens=engine._tokens_processed,
            )
        except Exception as save_exc:
            logger.error(f"Emergency checkpoint also failed: {save_exc}")
        raise

    # ── Final Export ─────────────────────────────────────────────────────────
    if is_main:
        final_dir = Path(train_config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting consolidated BF16 model to {final_dir}...")
        # DeepSpeed ZeRO-3 weights are gathered inside PretrainEngine._save_checkpoint.
        # Here we convert the gathered FP32 master weights back to BF16 for efficient
        # distribution and downstream use.
        _bf16_path = str(final_dir / "model_bf16.pt")
        try:
            # If the DeepSpeed consolidated weights already exist, convert them.
            _ds_mp_path = str(final_dir / "mp_rank_00_model_states.pt")
            if Path(_ds_mp_path).exists():
                _full_sd = torch.load(_ds_mp_path, map_location="cpu")
                _model_sd = _full_sd.get("module", _full_sd)
                _bf16_sd  = {k: v.to(torch.bfloat16) for k, v in _model_sd.items()}
                torch.save(_bf16_sd, _bf16_path)
                logger.info(f"BF16 model saved: {_bf16_path} "
                            f"({Path(_bf16_path).stat().st_size / 1e9:.2f} GB)")
            else:
                # Fallback: save current in-memory model (may already be BF16)
                _bf16_sd = {k: v.to(torch.bfloat16)
                            for k, v in model.state_dict().items()}
                torch.save(_bf16_sd, _bf16_path)
                logger.info(f"BF16 model saved (fallback): {_bf16_path}")
        except Exception as export_exc:
            logger.error(f"BF16 export failed: {export_exc}")

        # Save final tokenizer and model config alongside weights
        tokenizer.save_pretrained(str(final_dir))
        model_config.to_json(str(final_dir / "model_config.json"))
        logger.info("Tokenizer and model config saved alongside BF16 weights")
        
        # Upload final model to Hugging Face Hub in the background
        from training.hub_uploader import upload_folder_async
        upload_folder_async(str(final_dir), repo_name="forge-3b-pretrain", folder_in_repo="final")

        # Summary banner
        import json as _json
        _latest_meta = list(Path(train_config.output_dir).glob("phase*_step*/checkpoint_meta.json"))
        if _latest_meta:
            _latest_meta.sort(key=lambda p: p.stat().st_mtime)
            with open(str(_latest_meta[-1])) as _f:
                _meta_data = _json.load(_f)
            logger.info("=" * 70)
            logger.info("PRETRAINING COMPLETE")
            logger.info(f"  Tokens processed : {_meta_data.get('tokens_processed', 0) / 1e9:.1f}B")
            logger.info(f"  Total cost       : ${_meta_data.get('cost_usd', 0):.2f}")
            logger.info(f"  Budget remaining : ${_meta_data.get('budget_remaining_usd', 0):.2f}")
            logger.info(f"  Final checkpoint : {str(_latest_meta[-1].parent)}")
            logger.info("=" * 70)

        if wandb_run is not None:
            wandb_run.summary.update({
                "pretraining/completed": True,
                "pretraining/tokens_b": engine._tokens_processed / 1e9,
                "pretraining/total_steps": engine._global_step,
            })
            wandb_run.finish()
            logger.info("WandB run finalised")

    # Ensure all ranks finish cleanly before the process exits
    if world_size > 1:
        import torch.distributed as _dist
        _dist.barrier()
        _dist.destroy_process_group()

    logger.info(f"[rank={rank}] Exiting cleanly.")


if __name__ == "__main__":
    main()