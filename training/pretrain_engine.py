"""
FORGE-3B Pre-Training Engine.

Full three-phase pretraining with:
- DeepSpeed ZeRO-3 for memory efficiency
- BF16 mixed precision
- Gradient checkpointing
- torch.compile for kernel fusion
- Automatic checkpoint management
- Comprehensive logging (wandb + terminal)
- MoE Gumbel temperature annealing
- Budget tracking
"""

from __future__ import annotations
import os
import gc
import sys
import json
import math
import time
import shutil
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from training.hub_uploader import upload_folder_async


class PretrainEngine:
    """
    Production-grade pretraining engine for FORGE-3B.
    
    Manages the full three-phase training lifecycle:
    Phase 1: Vocabulary warmup   (5B tokens,  seq=512)
    Phase 2: Core pretraining    (43B tokens, seq=2048)
    Phase 3: Context extension   (2B tokens,  seq=4096)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_config,
        model_config,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        wandb_run=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = train_config
        self.model_config = model_config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = (rank == 0)
        self.wandb_run = wandb_run
        
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Training state
        self._global_step = 0
        self._tokens_processed = 0
        self._start_time = time.time()
        
        # Budget tracking (USD)
        self._hourly_cost = 63.17   # 16× H100 SXM community cloud
        self._budget = 450.0
        
        # Memory monitor
        from training.gpu_optimizer import GPUMemoryMonitor, ThroughputMeter
        self.memory_monitor = GPUMemoryMonitor(self.device, oom_threshold_gb=8.0)
        self.throughput_meter = ThroughputMeter(
            model_flops_per_token=6 * 1_174_000_000,  # 6 × 1.174B active
            device=self.device,
            world_size=world_size,
        )
        
        # GradScaler for FP16 (not needed for BF16, but keep for compatibility)
        self.scaler = None  # BF16 doesn't need gradient scaling
    
    def _setup_deepspeed(self, model, optimizer, lr_scheduler):
        """Initialize DeepSpeed ZeRO-3 engine."""
        try:
            import deepspeed
            
            ds_config = json.load(open(self.config.deepspeed_config))
            
            # Inject dynamic values
            ds_config["train_micro_batch_size_per_gpu"] = self.config.micro_batch_size_per_gpu
            # Inject GA steps — Phase 2 default; overridden per-phase in run_phase()
            ds_config["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps_phase2
            
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=ds_config,
            )
            
            logger.info("DeepSpeed ZeRO-3 initialized")
            return model_engine, optimizer
        
        except ImportError:
            logger.warning("DeepSpeed not available — using standard DDP")
            return model, optimizer
    
    def _save_checkpoint(self, phase: int, step: int, tokens: int):
        """Save model checkpoint with metadata."""
        if not self.is_main:
            return
        
        ckpt_dir = Path(self.config.output_dir) / f"phase{phase}_step{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        try:
            # DeepSpeed save
            if hasattr(self.model_engine, "save_checkpoint"):
                self.model_engine.save_checkpoint(str(ckpt_dir))
            else:
                torch.save(
                    self.model.state_dict(),
                    str(ckpt_dir / "model.safetensors"),
                )
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            return
        
        # Save metadata
        meta = {
            "phase": phase,
            "step": step,
            "tokens_processed": tokens,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "cost_usd": (time.time() - self._start_time) / 3600 * self._hourly_cost,
            "budget_remaining_usd": self._budget - 
                                   (time.time() - self._start_time) / 3600 * self._hourly_cost,
        }
        with open(str(ckpt_dir / "checkpoint_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Checkpoint saved: {ckpt_dir} "
                    f"(${meta['cost_usd']:.2f} spent, ${meta['budget_remaining_usd']:.2f} remaining)")
        
        # Upload checkpoint to Hugging Face Hub in the background
        upload_folder_async(str(ckpt_dir), repo_name="forge-3b-pretrain", folder_in_repo=ckpt_dir.name)
        
        # Rotate old checkpoints
        self._rotate_checkpoints()
    
    def _rotate_checkpoints(self):
        """Keep only the last N checkpoints to save disk space."""
        ckpt_base = Path(self.config.output_dir)
        checkpoints = sorted(ckpt_base.glob("phase*_step*"), 
                             key=lambda p: p.stat().st_mtime)
        
        n_keep = self.config.keep_last_n_checkpoints
        if len(checkpoints) > n_keep:
            for old_ckpt in checkpoints[:-n_keep]:
                shutil.rmtree(old_ckpt, ignore_errors=True)
                logger.debug(f"Removed old checkpoint: {old_ckpt}")
    
    def _log_step(
        self,
        step: int,
        loss: float,
        aux_loss: float,
        lr: float,
        tokens: int,
        phase: int,
        extra: Optional[Dict] = None,
    ):
        """Log training metrics to terminal and wandb."""
        elapsed_h = (time.time() - self._start_time) / 3600
        cost_usd = elapsed_h * self._hourly_cost
        
        throughput_stats = self.throughput_meter.get_stats()
        tok_per_sec = throughput_stats.get("tokens_per_sec", 0)
        mfu = throughput_stats.get("mfu", 0)
        
        if self.is_main and step % self.config.log_every_n_steps == 0:
            log_msg = (
                f"Phase{phase} | Step {step:,} | "
                f"Tokens {tokens/1e9:.2f}B | "
                f"Loss {loss:.4f} | "
                f"AuxLoss {aux_loss:.5f} | "
                f"LR {lr:.2e} | "
                f"Tok/s {tok_per_sec:,.0f} | "
                f"MFU {mfu*100:.1f}% | "
                f"Cost ${cost_usd:.2f}/${self._budget:.0f}"
            )
            logger.info(log_msg)
            
            if self.wandb_run is not None:
                wandb_metrics = {
                    f"phase{phase}/loss": loss,
                    f"phase{phase}/aux_loss": aux_loss,
                    f"phase{phase}/lr": lr,
                    f"phase{phase}/tokens_b": tokens / 1e9,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/cost_usd": cost_usd,
                    "train/budget_remaining": self._budget - cost_usd,
                    "hardware/gpu_mem_gb": self.memory_monitor.snapshot()["allocated_gb"],
                }
                if extra:
                    wandb_metrics.update(extra)
                self.wandb_run.log(wandb_metrics, step=step)
    
    def _update_moe_temperature(self, progress: float):
        """Anneal Gumbel temperature in all HSE layers."""
        from model.hse_layer import HSELayer
        for layer in self.model.layers:
            if isinstance(layer.ffn, HSELayer):
                layer.ffn.update_gumbel_tau(progress)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE RUNNERS
    # ─────────────────────────────────────────────────────────────────────────
    
    def run_phase(
        self,
        phase: int,
        dataloader: DataLoader,
        optimizer,
        scheduler,
        target_tokens: int,
        seq_len: int,
        gradient_accumulation_steps: int,
        model_engine=None,
    ) -> int:
        """
        Run one training phase. Returns total tokens processed.
        
        Uses gradient accumulation to achieve large effective batch sizes
        within the GPU memory budget.
        """
        from training.gpu_optimizer import bf16_autocast, clip_grad_norm_and_log
        
        model = model_engine if model_engine is not None else self.model
        model.train()
        
        if hasattr(model, "set_gradient_accumulation_steps"):
            model.set_gradient_accumulation_steps(gradient_accumulation_steps)
        
        tokens_this_phase = 0
        step = 0
        accum_loss = 0.0
        accum_aux = 0.0
        
        data_iter = iter(dataloader)
        
        while tokens_this_phase < target_tokens:
            if self.is_main:
                logger.info(f"Starting global step {self._global_step + 1} (accumulating {gradient_accumulation_steps} micro-batches)...")
            optimizer.zero_grad(set_to_none=True)  # more efficient than .zero_grad()
            
            step_loss = 0.0
            step_aux = 0.0
            
            self.throughput_meter.start_step()
            
            for accum_step in range(gradient_accumulation_steps):
                # Fetch batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch.get("labels", input_ids.clone())
                labels = labels.to(self.device, non_blocking=True)
                
                n_tokens = (labels != -100).sum().item()
                
                # Gradient sync only on last accumulation step for DDP (DeepSpeed handles this internally)
                is_deepspeed = hasattr(model, "backward")
                sync_context = (
                    model.no_sync() 
                    if hasattr(model, "no_sync") and not is_deepspeed and accum_step < gradient_accumulation_steps - 1
                    else contextlib.nullcontext()
                )
                
                with sync_context:
                    with bf16_autocast(enabled=self.config.bf16):
                        outputs = model(
                            input_ids=input_ids,
                            labels=labels,
                            return_aux_loss=True,
                        )
                        loss = outputs["loss"] / gradient_accumulation_steps
                
                # Backward
                if hasattr(model, "backward"):
                    model.backward(loss)  # DeepSpeed
                    model.step()          # DeepSpeed handles accumulation and weight updates internally!
                else:
                    loss.backward()
                
                step_loss += loss.item() * gradient_accumulation_steps
                aux = outputs.get("aux_loss")
                if aux is not None:
                    step_aux += float(aux.detach()) / gradient_accumulation_steps
            
            # Gradient clipping and optimizer step
            if hasattr(model, "step"):
                grad_norm = model.get_global_grad_norm() if hasattr(model, "get_global_grad_norm") else 0.0
            else:
                grad_norm = clip_grad_norm_and_log(model.parameters(), self.config.grad_clip)
                optimizer.step()
            
            # LR schedule update
            batch_tokens = self.config.micro_batch_size_per_gpu * seq_len * self.world_size * gradient_accumulation_steps
            scheduler.step(batch_tokens)
            
            # Throughput tracking
            self.throughput_meter.end_step(batch_tokens)
            
            tokens_this_phase += batch_tokens
            self._tokens_processed += batch_tokens
            self._global_step += 1
            step += 1
            
            # MoE temperature annealing
            total_prog = self._tokens_processed / self.model_config.hse_gumbel_tau_warmup_tokens
            self._update_moe_temperature(min(1.0, total_prog))
            
            # Logging
            self._log_step(
                step=self._global_step,
                loss=step_loss,
                aux_loss=step_aux,
                lr=scheduler.get_lr(),
                tokens=self._tokens_processed,
                phase=phase,
                extra={"train/grad_norm": grad_norm},
            )
            
            # Checkpointing
            if self._tokens_processed % self.config.save_every_n_tokens < batch_tokens:
                self._save_checkpoint(phase, self._global_step, self._tokens_processed)
            
            # Budget check
            elapsed_h = (time.time() - self._start_time) / 3600
            cost = elapsed_h * self._hourly_cost
            if cost > self._budget * 0.95:  # 95% budget warning
                logger.warning(f"⚠️  Budget at 95%: ${cost:.2f} of ${self._budget:.2f}")
            if cost > self._budget:
                logger.error(f"❌ Budget exceeded: ${cost:.2f} > ${self._budget:.2f}. Stopping!")
                self._save_checkpoint(phase, self._global_step, self._tokens_processed)
                sys.exit(1)
        
        return tokens_this_phase
    
    def train(
        self,
        phase1_dataloader: DataLoader,
        phase2_dataloader: DataLoader,
        phase3_dataloader: DataLoader,
        optimizer,
        lr_scheduler,
    ):
        """
        Full three-phase pretraining entrypoint.
        """
        from training.gpu_optimizer import compile_model, warmup_gpu
        
        # ── Setup ─────────────────────────────────────────────────────────────
        warmup_gpu(self.device)
        
        # torch.compile
        if self.config.torch_compile:
            self.model = compile_model(
                self.model,
                mode=self.config.compile_mode,
                fullgraph=False,  # False for MoE (dynamic dispatch)
                dynamic=True,     # Handle variable batch sizes
            )
        
        # DeepSpeed
        self.model_engine, optimizer = self._setup_deepspeed(self.model, optimizer, lr_scheduler)
        
        logger.info("=" * 70)
        logger.info("FORGE-3B PRETRAINING STARTED")
        logger.info(f"  Target: {self.config.total_tokens/1e9:.0f}B tokens")
        logger.info(f"  Budget: ${self._budget:.0f}")
        logger.info(f"  Cost rate: ${self._hourly_cost:.2f}/hr")
        logger.info(f"  Max training hours: {self._budget/self._hourly_cost:.1f}h")
        logger.info("=" * 70)
        
        # ── Phase 1: Vocabulary Warmup ────────────────────────────────────────
        logger.info(f"\n{'='*50}")
        logger.info("PHASE 1: Vocabulary Warmup (5B tokens, seq=512)")
        
        # Phase 1 uses smaller context — update local window in ARG layers
        self._set_local_window(64)
        
        ga_steps_p1 = max(1, self.config.phase1_global_batch_tokens // (
            self.config.micro_batch_size_per_gpu * 512 * self.world_size
        ))
        
        self.run_phase(
            phase=1,
            dataloader=phase1_dataloader,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            target_tokens=self.config.phase1_tokens,
            seq_len=512,
            gradient_accumulation_steps=ga_steps_p1,
            model_engine=self.model_engine,
        )
        self._save_checkpoint(1, self._global_step, self._tokens_processed)
        
        # ── Phase 2: Core Pre-Training ─────────────────────────────────────
        logger.info(f"\n{'='*50}")
        logger.info("PHASE 2: Core Pre-Training (43B tokens, seq=2048)")
        
        ga_steps_p2 = self.config.gradient_accumulation_steps_phase2
        
        self.run_phase(
            phase=2,
            dataloader=phase2_dataloader,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            target_tokens=self.config.phase2_tokens,
            seq_len=2048,
            gradient_accumulation_steps=ga_steps_p2,
            model_engine=self.model_engine,
        )
        self._save_checkpoint(2, self._global_step, self._tokens_processed)
        
        # ── Phase 3: Context Extension ────────────────────────────────────
        logger.info(f"\n{'='*50}")
        logger.info("PHASE 3: Context Extension (2B tokens, seq=4096)")
        
        # Enable YaRN RoPE scaling for context extension
        self._enable_yarn_scaling(factor=2.0)
        # Extend ARG local window
        self._set_local_window(128)
        
        ga_steps_p3 = max(1, self.config.phase3_global_batch_tokens // (
            self.config.micro_batch_size_per_gpu * 4096 * self.world_size
        ))
        
        # Use lower LR for context extension
        lr_const_scheduler = ConstantLRScheduler(optimizer, lr=3e-5)
        
        self.run_phase(
            phase=3,
            dataloader=phase3_dataloader,
            optimizer=optimizer,
            scheduler=lr_const_scheduler,
            target_tokens=self.config.phase3_tokens,
            seq_len=4096,
            gradient_accumulation_steps=ga_steps_p3,
            model_engine=self.model_engine,
        )
        
        # ── Final checkpoint ───────────────────────────────────────────────
        self._save_checkpoint(3, self._global_step, self._tokens_processed)
        
        # Copy to 'final'
        final_dir = Path(self.config.output_dir) / "final"
        if self.is_main:
            shutil.copytree(
                Path(self.config.output_dir) / f"phase3_step{self._global_step}",
                final_dir, 
                dirs_exist_ok=True,
            )
        
        elapsed_h = (time.time() - self._start_time) / 3600
        total_cost = elapsed_h * self._hourly_cost
        
        logger.info(f"\n{'='*70}")
        logger.info("PRETRAINING COMPLETE!")
        logger.info(f"  Tokens: {self._tokens_processed/1e9:.1f}B")
        logger.info(f"  Steps: {self._global_step:,}")
        logger.info(f"  Time: {elapsed_h:.1f}h")
        logger.info(f"  Cost: ${total_cost:.2f} of ${self._budget:.2f} budget")
        logger.info(f"  Remaining budget: ${self._budget - total_cost:.2f}")
        logger.info(f"  Tokens/sec: {self._tokens_processed / (elapsed_h * 3600):,.0f}")
        logger.info(f"{'='*70}\n")
    
    def _set_local_window(self, window_size: int):
        """Update ARG local attention window size."""
        from model.arg_layer import ARGLayer
        for layer in self.model.layers:
            if isinstance(layer.seq_mixer, ARGLayer):
                layer.seq_mixer.local_window = window_size
                # Rebuild position cache
                layer.seq_mixer.local_rope._cache_seq_len = 0
                layer.seq_mixer.local_rope._build_cache(window_size * 4, self.device)
        logger.info(f"ARG local window updated to {window_size}")
    
    def _enable_yarn_scaling(self, factor: float = 2.0):
        """Enable YaRN RoPE scaling in MHA layers for context extension."""
        from model.mha_layer import GlobalMHALayer
        from model.rotary_embedding import RotaryEmbedding
        
        for layer in self.model.layers:
            if isinstance(layer.seq_mixer, GlobalMHALayer):
                # Replace rope with YaRN-scaled version
                layer.seq_mixer.rope = RotaryEmbedding(
                    head_dim=self.model_config.mha_head_dim,
                    max_seq_len=self.model_config.max_seq_len * 2,
                    base=self.model_config.rope_theta,
                    scaling_type="yarn",
                    scaling_factor=factor,
                ).to(self.device)
        logger.info(f"YaRN RoPE scaling enabled: factor={factor}")


import contextlib
from training.scheduler import ConstantLRScheduler