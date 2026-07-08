#!/usr/bin/env python3
"""
FORGE-3B Direct Preference Optimization (DPO) Entry Point.

Usage (16× H100 on RunPod):
    deepspeed --num_gpus=16 run_dpo.py \\
        --base_model  /workspace/checkpoints/forge_3b_sft/final \\
        --data_path   /workspace/data/dpo/preferences.jsonl \\
        --output_dir  /workspace/checkpoints/forge_3b_dpo \\
        --wandb_project forge_3b_dpo

Single GPU:
    python run_dpo.py \\
        --base_model ./checkpoints/forge_3b_sft/final \\
        --data_path  ./data/dpo/preferences.jsonl \\
        --output_dir ./checkpoints/forge_3b_dpo
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_dpo")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FORGE-3B DPO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--base_model",   required=True,
                   help="SFT model dir (contains model_bf16.pt + tokenizer/)")
    p.add_argument("--data_path",    required=True,
                   help="JSONL with preference pairs (prompt/chosen/rejected fields)")
    p.add_argument("--output_dir",   default="./checkpoints/forge_3b_dpo")
    p.add_argument("--model_config", default=None)
    p.add_argument("--resume_from",  default=None)

    # DPO hyperparams
    p.add_argument("--beta",        type=float, default=0.1,
                   help="KL penalty coefficient")
    p.add_argument("--loss_type",   default="dpo",
                   choices=["dpo", "ipo", "cdpo"],
                   help="DPO variant")
    p.add_argument("--n_epochs",    type=int,   default=1)
    p.add_argument("--batch_pairs", type=int,   default=16,
                   help="Preference pairs per GPU per step")
    p.add_argument("--ga_steps",    type=int,   default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--seq_len",     type=int,   default=4096)

    # Optimizer
    p.add_argument("--lr",          type=float, default=5e-7)
    p.add_argument("--grad_clip",   type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # Tokenizer
    p.add_argument("--tokenizer_profile", default="standard",
                   choices=["standard", "lite"])

    # Infrastructure
    p.add_argument("--deepspeed_config", default="./configs/ds_zero3_sft.json")
    p.add_argument("--bf16",             action="store_true", default=True)
    p.add_argument("--save_every_steps", type=int, default=100)
    p.add_argument("--seed",             type=int, default=42)

    # Logging
    p.add_argument("--wandb_project", default="forge_3b_dpo")
    p.add_argument("--wandb_entity",  default=None)

    return p


# ── Distributed ───────────────────────────────────────────────────────────────
def _setup_distributed():
    rank       = int(os.environ.get("RANK",       os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device
# ── Weight loading helper ─────────────────────────────────────────────────────
def _load_model_weights(model, base_path: Path, bf16: bool, logger, is_zero3: bool = False):
    """Try multiple checkpoint filename conventions, load with strict=False."""
    for filename in ("model_bf16.pt", "model.pt", "pytorch_model.bin"):
        weight_path = base_path / filename
        if weight_path.exists():
            state_dict = torch.load(str(weight_path), map_location="cpu")
            if bf16:
                state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            
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
            logger.info(f"Weights loaded from {weight_path}")
            return
    raise FileNotFoundError(
        f"No model weights found in {base_path}. "
        "Expected: model_bf16.pt | model.pt | pytorch_model.bin"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # ── Distributed ───────────────────────────────────────────────────────────
    rank, world_size, local_rank, device = _setup_distributed()
    is_main = (rank == 0)

    if not is_main:
        logger.setLevel(logging.WARNING)

    torch.manual_seed(args.seed + rank)

    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"FORGE-3B DPO | beta={args.beta} | loss={args.loss_type} | "
            f"world_size={world_size} | device={device}"
        )

    # ── Configs ───────────────────────────────────────────────────────────────
    from config import ForgeModelConfig, DPOConfig

    model_config_path = args.model_config or str(
        Path(args.base_model) / "model_config.json"
    )
    if Path(model_config_path).exists():
        model_config = ForgeModelConfig.from_json(model_config_path)
        logger.info(f"Loaded model config: {model_config_path}")
    else:
        model_config = ForgeModelConfig()
        logger.warning(f"model_config.json not found, using defaults")

    dpo_config = DPOConfig(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        data_path=args.data_path,
        beta=args.beta,
        loss_type=args.loss_type,
        batch_size_pairs=args.batch_pairs,
        gradient_accumulation_steps=args.ga_steps,
        seq_len=args.seq_len,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        save_every_n_steps=args.save_every_steps,
        wandb_project=args.wandb_project,
        deepspeed_config=args.deepspeed_config,
        num_gpus=world_size,
        bf16=args.bf16,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
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
        logger.warning(f"No tokenizer found at {tok_dir} — initialised fresh")

    model_config.vocab_size = tokenizer.vocab_size

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
                    ds_config["gradient_accumulation_steps"] = dpo_config.gradient_accumulation_steps
                if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
                    # For DPO, batch_pairs is equivalent to micro_batch per GPU
                    ds_config["train_micro_batch_size_per_gpu"] = dpo_config.batch_size_pairs
        except Exception as e:
            logger.warning(f"Failed to check DeepSpeed config stage: {e}")

    # ── Policy Model ──────────────────────────────────────────────────────────
    logger.info("Building FORGE-3B policy model...")
    from model.forge_model import build_forge_3b

    if is_zero3 and ds_config is not None:
        import deepspeed
        logger.info("ZeRO-3 detected — wrapping policy model initialization in deepspeed.zero.Init()")
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            policy_model = build_forge_3b(model_config)
    else:
        policy_model = build_forge_3b(model_config)
        policy_model = policy_model.to(device)
    _load_model_weights(policy_model, Path(args.base_model), args.bf16, logger, is_zero3=is_zero3)

    # Resume from a prior DPO checkpoint if requested
    if args.resume_from:
        resume_path = Path(args.resume_from) / "model.pt"
        if resume_path.exists():
            sd = torch.load(str(resume_path), map_location="cpu")
            if is_zero3:
                from deepspeed.zero import GatheredParameters
                with GatheredParameters(list(policy_model.parameters()), modifier_rank=0):
                    policy_model.load_state_dict(sd, strict=False)
            else:
                policy_model.load_state_dict(sd, strict=False)
            logger.info(f"DPO resume weights loaded from {resume_path}")

    # ── Reference Model (frozen copy of the SFT model) ────────────────────────
    # The reference model is kept in BF16 and set to eval() — never backpropagated.
    # On 16× H100 this fits fine alongside the policy with ZeRO-3 on the policy.
    logger.info("Building frozen reference model...")
    if is_zero3 and ds_config is not None:
        logger.info("ZeRO-3 detected — wrapping reference model initialization in deepspeed.zero.Init()")
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            ref_model = build_forge_3b(model_config)
    else:
        ref_model = build_forge_3b(model_config)
        ref_model = ref_model.to(device)
    _load_model_weights(ref_model, Path(args.base_model), args.bf16, logger, is_zero3=is_zero3)

    # Hard-freeze the reference model — zero optimizer memory
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()

    n_policy_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in policy_model.parameters())
    logger.info(
        f"Policy: {n_policy_params / 1e9:.3f}B params | "
        f"Reference: frozen BF16"
    )

    if model_config.use_gradient_checkpointing:
        policy_model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on policy")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    
    # If data_path is a directory, merge all .jsonl files into one
    if data_path.is_dir():
        merged_path = data_path / "_merged_dpo.jsonl"
        if not merged_path.exists():
            logger.info(f"Merging DPO JSONL files from {data_path}...")
            jsonl_files = sorted(data_path.glob("**/*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(f"No .jsonl files found in {data_path}")
            with open(merged_path, "w") as out:
                for jf in jsonl_files:
                    with open(jf) as inp:
                        for line in inp:
                            if line.strip():
                                out.write(line if line.endswith("\n") else line + "\n")
            logger.info(f"Merged {len(jsonl_files)} files → {merged_path}")
        actual_data_path = str(merged_path)
    else:
        if not data_path.exists():
            raise FileNotFoundError(
                f"DPO data not found: {data_path}\n"
                "Pass a .jsonl file or a directory containing .jsonl files."
            )
        actual_data_path = str(data_path)
    
    # Sniff the first line to detect format
    with open(actual_data_path) as f:
        first_line = f.readline().strip()
    first_item = json.loads(first_line)
    is_pretokenized = ("prompt_ids" in first_item or "chosen_full_ids" in first_item)
    
    if is_pretokenized:
        logger.info("Detected pre-tokenized DPO data — using PreTokenizedPreferenceDataset")
        from training.dpo_engine import PreTokenizedPreferenceDataset
        dataset = PreTokenizedPreferenceDataset(
            data_path=actual_data_path,
            seq_len=args.seq_len,
        )
    else:
        logger.info("Detected raw text DPO data — using PreferenceDataset (live tokenization)")
        from training.dpo_engine import PreferenceDataset
        dataset = PreferenceDataset(
            data_path=actual_data_path,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
        )
    logger.info(f"Preference dataset: {len(dataset):,} pairs")

    from tokenizer.data_collator import DPOCollator
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=dpo_config.batch_size_pairs,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        drop_last=True,
        collate_fn=DPOCollator(
            pad_token_id=model_config.pad_token_id,
            max_length=args.seq_len,
        ),
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    logger.info("Building AdamW optimizer for DPO...")
    from training.optimizer import build_optimizer

    optimizer = build_optimizer(
        model=policy_model,
        lr_max=dpo_config.lr,
        beta1=0.9,
        beta2=0.99,         # higher β₂ for DPO stability
        eps=1e-8,
        weight_decay=dpo_config.weight_decay,
        embedding_lr_mult=1.0,
        ssm_lr_mult=1.0,
        router_lr_mult=1.0,
        use_fused=True,
    )

    # ── LR Scheduler (constant for DPO — very low LR to prevent collapse) ────
    from training.scheduler import ConstantLRScheduler

    lr_scheduler = ConstantLRScheduler(optimizer=optimizer, lr=dpo_config.lr)

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if is_main and dpo_config.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=dpo_config.wandb_project,
                entity=args.wandb_entity,
                name=f"forge3b-dpo-{datetime.datetime.now().strftime('%m%d_%H%M')}",
                resume="allow" if args.resume_from else None,
                config={
                    "dpo/beta":          dpo_config.beta,
                    "dpo/loss_type":     dpo_config.loss_type,
                    "dpo/n_pairs":       len(dataset),
                    "dpo/n_epochs":      dpo_config.n_epochs,
                    "dpo/lr":            dpo_config.lr,
                    "dpo/seq_len":       dpo_config.seq_len,
                    "model/base":        args.base_model,
                    "infra/world_size":  world_size,
                    "infra/bf16":        dpo_config.bf16,
                },
                tags=["forge-3b", "dpo"],
            )
            logger.info(f"WandB run: {wandb_run.url}")
        except ImportError:
            logger.warning("WandB not installed")
        except Exception as exc:
            logger.warning(f"WandB init failed: {exc}")

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    # ── DPO Engine ────────────────────────────────────────────────────────────
    logger.info("Initializing DPOEngine...")
    from training.dpo_engine import DPOEngine

    engine = DPOEngine(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=dpo_config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        wandb_run=wandb_run,
    )

    if is_main:
        logger.info("=" * 60)
        logger.info("FORGE-3B DIRECT PREFERENCE OPTIMIZATION")
        logger.info(f"  Beta      : {dpo_config.beta}")
        logger.info(f"  Loss type : {dpo_config.loss_type}")
        logger.info(f"  Pairs     : {len(dataset):,}")
        logger.info(f"  Epochs    : {dpo_config.n_epochs}")
        logger.info(f"  LR        : {dpo_config.lr:.1e} (constant)")
        logger.info("=" * 60)

    try:
        engine.train(
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=lr_scheduler,
        )
    except KeyboardInterrupt:
        logger.warning("DPO interrupted — saving emergency checkpoint...")
        ckpt_dir = Path(dpo_config.output_dir) / "dpo_interrupt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(policy_model.state_dict(), str(ckpt_dir / "model.pt"))
        logger.info(f"Emergency checkpoint: {ckpt_dir}")
        raise
    except Exception as exc:
        logger.exception(f"DPO crashed: {exc}")
        raise

    # ── Final Export ──────────────────────────────────────────────────────────
    if is_main:
        final_dir = Path(dpo_config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        bf16_sd = {k: v.to(torch.bfloat16) for k, v in policy_model.state_dict().items()}
        torch.save(bf16_sd, str(final_dir / "model_bf16.pt"))
        tokenizer.save_pretrained(str(final_dir))
        model_config.to_json(str(final_dir / "model_config.json"))
        logger.info(f"DPO complete. Final policy model: {final_dir}")

        if wandb_run is not None:
            wandb_run.finish()

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    logger.info(f"[rank={rank}] DPO run complete.")


if __name__ == "__main__":
    main()
