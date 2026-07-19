"""
FORGE-3B Pre-Training Engine — Plain PyTorch DDP, no DeepSpeed.

Full three-phase pretraining with:
- Standard PyTorch DDP (works with torchrun for single or N GPUs)
- BF16 mixed precision
- Gradient checkpointing
- torch.compile for kernel fusion
- Automatic checkpoint management
- Comprehensive logging (wandb + terminal)
- MoE Gumbel temperature annealing
- Budget tracking
"""

from __future__ import annotations
import contextlib
import os
import gc
import sys
import json
import time
import shutil
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from training.hub_uploader import upload_folder_async
from training.scheduler import ConstantLRScheduler


class PretrainEngine:
    """
    Production-grade pretraining engine for FORGE-3B.
    Uses plain PyTorch DDP — no DeepSpeed dependency.

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
        self.tokenizer = tokenizer
        self.config = train_config
        self.model_config = model_config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = (rank == 0)
        self.wandb_run = wandb_run
        self.device = torch.device(f"cuda:{local_rank}")

        # raw_model: direct reference to nn.Module for layer-level access
        # (window updates, YaRN, state_dict saves, etc.)
        self.raw_model = model
        # self.model will be set to DDP-wrapped version in _wrap_ddp()
        self.model = model

        self._global_step = 0
        self._tokens_processed = 0
        self._start_time = time.time()

        # Budget tracking (USD)
        # Fix: Dynamic hourly cost based on RunPod Community Cloud 1x H100 SXM rate ($3.29/hr)
        self._hourly_cost = 3.29 * world_size
        self._budget = 450.0

        from training.gpu_optimizer import GPUMemoryMonitor, ThroughputMeter
        self.memory_monitor = GPUMemoryMonitor(self.device, oom_threshold_gb=8.0)
        self.throughput_meter = ThroughputMeter(
            model_flops_per_token=6 * 1_174_000_000,  # 6 × 1.174B active params
            device=self.device,
            world_size=world_size,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DDP SETUP
    # ─────────────────────────────────────────────────────────────────────────

    def _wrap_ddp(self):
        """Wrap model in DDP if world_size > 1. No-op on single GPU."""
        if self.world_size > 1:
            self.model = DDP(
                self.raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # set True if MoE routing leaves experts unused
                gradient_as_bucket_view=True,  # saves memory by aliasing gradient buckets
            )
            logger.info(f"Wrapped model in DDP (world_size={self.world_size})")
        else:
            self.model = self.raw_model
            logger.info("Single GPU — no DDP wrapping")

    # ─────────────────────────────────────────────────────────────────────────
    # CHECKPOINTING
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, phase: int, step: int, tokens: int):
        """Save model checkpoint with metadata. Only main rank writes."""
        if not self.is_main:
            return

        ckpt_dir = Path(self.config.output_dir) / f"phase{phase}_step{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(self.raw_model.state_dict(), str(ckpt_dir / "model.safetensors"))
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            return

        elapsed_h = (time.time() - self._start_time) / 3600
        cost_usd = elapsed_h * self._hourly_cost
        meta = {
            "phase": phase,
            "step": step,
            "tokens_processed": tokens,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "cost_usd": cost_usd,
            "budget_remaining_usd": self._budget - cost_usd,
        }
        with open(str(ckpt_dir / "checkpoint_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"Checkpoint saved: {ckpt_dir} "
            f"(${meta['cost_usd']:.2f} spent, ${meta['budget_remaining_usd']:.2f} remaining)"
        )
        upload_folder_async(str(ckpt_dir), repo_name="forge-3b-pretrain", folder_in_repo=ckpt_dir.name)
        self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        """Keep only the last N checkpoints to save disk space."""
        ckpt_base = Path(self.config.output_dir)
        checkpoints = sorted(
            ckpt_base.glob("phase*_step*"),
            key=lambda p: p.stat().st_mtime,
        )
        n_keep = self.config.keep_last_n_checkpoints
        if len(checkpoints) > n_keep:
            for old in checkpoints[:-n_keep]:
                shutil.rmtree(old, ignore_errors=True)
                logger.debug(f"Removed old checkpoint: {old}")

    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────────────────

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

        stats = self.throughput_meter.get_stats()
        tok_per_sec = stats.get("tokens_per_sec", 0)
        mfu = stats.get("mfu", 0)

        if self.is_main and step % self.config.log_every_n_steps == 0:
            logger.info(
                f"Phase{phase} | Step {step:,} | Tokens {tokens/1e9:.2f}B | "
                f"Loss {loss:.4f} | AuxLoss {aux_loss:.5f} | LR {lr:.2e} | "
                f"Tok/s {tok_per_sec:,.0f} | MFU {mfu*100:.1f}% | "
                f"Cost ${cost_usd:.2f}/${self._budget:.0f}"
            )
            if self.wandb_run is not None:
                metrics = {
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
                    metrics.update(extra)
                self.wandb_run.log(metrics, step=step)

    # ─────────────────────────────────────────────────────────────────────────
    # MoE TEMPERATURE ANNEALING
    # ─────────────────────────────────────────────────────────────────────────

    def _update_moe_temperature(self, progress: float):
        """Anneal Gumbel temperature in all HSE layers."""
        from model.hse_layer import HSELayer
        for layer in self.raw_model.layers:
            if isinstance(layer.ffn, HSELayer):
                layer.ffn.update_gumbel_tau(progress)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE RUNNER
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
    ) -> int:
        """
        Run one training phase. Returns total tokens processed this phase.

        Uses gradient accumulation with DDP no_sync() to achieve large effective
        batch sizes within the GPU memory budget.
        """
        from training.gpu_optimizer import bf16_autocast, clip_grad_norm_and_log

        self.model.train()
        tokens_this_phase = 0
        data_iter = iter(dataloader)

        while tokens_this_phase < target_tokens:
            if self.is_main:
                logger.info(
                    f"Starting global step {self._global_step + 1} "
                    f"(accumulating {gradient_accumulation_steps} micro-batches)..."
                )

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_aux = 0.0
            self.throughput_meter.start_step()
            _micro_step_start = time.perf_counter()
            _tokens_per_micro = self.config.micro_batch_per_gpu * context_length * self.world_size

            for accum_step in range(gradient_accumulation_steps):
                if self.is_main and (
                    accum_step % 100 == 0 or accum_step == gradient_accumulation_steps - 1
                ):
                    elapsed = time.perf_counter() - _micro_step_start
                    if accum_step > 0 and elapsed > 0:
                        live_tok_sec = (accum_step * _tokens_per_micro) / elapsed
                        live_tflops = self.throughput_meter.model_flops_per_token * live_tok_sec / 1e12
                        live_mfu = live_tflops / (self.throughput_meter.peak_tflops * self.world_size)
                        logger.info(
                            f"  [Micro-step {accum_step}/{gradient_accumulation_steps}] "
                            f"Forward/backward... | Live Tok/s: {live_tok_sec:,.0f} | MFU: {live_mfu*100:.1f}%"
                        )
                    else:
                        logger.info(
                            f"  [Micro-step {accum_step}/{gradient_accumulation_steps}] "
                            f"Forward/backward pass..."
                        )

                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch.get("labels", input_ids.clone()).to(self.device, non_blocking=True)

                # Only sync gradients on the last micro-step (DDP optimization — avoids
                # all-reduce on every step, collects it only at the final accumulation step)
                is_last_accum = accum_step == gradient_accumulation_steps - 1
                sync_ctx = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) and not is_last_accum
                    else contextlib.nullcontext()
                )

                with sync_ctx:
                    with bf16_autocast(enabled=self.config.bf16):
                        outputs = self.model(
                            input_ids=input_ids,
                            labels=labels,
                            return_aux_loss=True,
                        )
                        loss = outputs["loss"] / gradient_accumulation_steps
                    loss.backward()

                step_loss += loss.item()
                aux = outputs.get("aux_loss")
                if aux is not None:
                    step_aux += float(aux.detach()) / gradient_accumulation_steps

            # Clip gradients and step
            grad_norm = clip_grad_norm_and_log(self.model.parameters(), self.config.grad_clip)
            optimizer.step()

            # Token accounting
            batch_tokens = (
                self.config.micro_batch_size_per_gpu
                * seq_len
                * self.world_size
                * gradient_accumulation_steps
            )
            scheduler.step(batch_tokens)
            self.throughput_meter.end_step(batch_tokens)

            tokens_this_phase += batch_tokens
            self._tokens_processed += batch_tokens
            self._global_step += 1

            # MoE temperature annealing
            total_prog = self._tokens_processed / self.model_config.hse_gumbel_tau_warmup_tokens
            self._update_moe_temperature(min(1.0, total_prog))

            self._log_step(
                self._global_step, step_loss, step_aux,
                scheduler.get_lr(), self._tokens_processed, phase,
                extra={"train/grad_norm": grad_norm},
            )

            # Periodic checkpointing
            if self._tokens_processed % self.config.save_every_n_tokens < batch_tokens:
                self._save_checkpoint(phase, self._global_step, self._tokens_processed)

            # Budget guard
            elapsed_h = (time.time() - self._start_time) / 3600
            cost = elapsed_h * self._hourly_cost
            if cost > self._budget * 0.95:
                logger.warning(f"⚠️  Budget at 95%: ${cost:.2f} of ${self._budget:.2f}")
            if cost > self._budget:
                logger.error(f"❌ Budget exceeded: ${cost:.2f} > ${self._budget:.2f}. Stopping!")
                self._save_checkpoint(phase, self._global_step, self._tokens_processed)
                sys.exit(1)

        return tokens_this_phase

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRAIN ENTRYPOINT
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        phase1_dataloader: DataLoader,
        phase2_dataloader: DataLoader,
        phase3_dataloader: DataLoader,
        optimizer,
        lr_scheduler,
    ):
        """Full three-phase pretraining entrypoint."""
        from training.gpu_optimizer import warmup_gpu

        warmup_gpu(self.device)

        # torch.compile must run BEFORE DDP wrapping to avoid Dynamo graph breaks
        if self.config.torch_compile:
            from training.gpu_optimizer import compile_forge_layers
            compile_forge_layers(self.raw_model, mode=self.config.compile_mode, dynamic=True)

        # Wrap in DDP after compile, after model is already on device
        self._wrap_ddp()

        logger.info("=" * 70)
        logger.info("FORGE-3B PRETRAINING STARTED (plain PyTorch DDP, no DeepSpeed)")
        logger.info(f"  Target: {self.config.total_tokens/1e9:.0f}B tokens")
        logger.info(f"  Budget: ${self._budget:.0f}  |  Cost rate: ${self._hourly_cost:.2f}/hr")
        logger.info(f"  Max training hours: {self._budget/self._hourly_cost:.1f}h")
        logger.info(f"  World size: {self.world_size}")
        logger.info("=" * 70)

        # ── Phase 1: Vocabulary Warmup ────────────────────────────────────────
        logger.info(f"\n{'='*50}")
        logger.info("PHASE 1: Vocabulary Warmup (5B tokens, seq=512)")
        self._set_local_window(64)
        ga_steps_p1 = max(1, self.config.phase1_global_batch_tokens // (
            self.config.micro_batch_size_per_gpu * 512 * self.world_size
        ))
        self.run_phase(
            phase=1, dataloader=phase1_dataloader, optimizer=optimizer,
            scheduler=lr_scheduler, target_tokens=self.config.phase1_tokens,
            seq_len=512, gradient_accumulation_steps=ga_steps_p1,
        )
        self._save_checkpoint(1, self._global_step, self._tokens_processed)

        # ── Phase 2: Core Pre-Training ────────────────────────────────────────
        logger.info(f"\n{'='*50}")
        logger.info("PHASE 2: Core Pre-Training (43B tokens, seq=2048)")
        ga_steps_p2 = self.config.gradient_accumulation_steps_phase2
        self.run_phase(
            phase=2, dataloader=phase2_dataloader, optimizer=optimizer,
            scheduler=lr_scheduler, target_tokens=self.config.phase2_tokens,
            seq_len=2048, gradient_accumulation_steps=ga_steps_p2,
        )
        self._save_checkpoint(2, self._global_step, self._tokens_processed)

        # ── Phase 3: Context Extension ────────────────────────────────────────
        logger.info(f"\n{'='*50}")
        logger.info("PHASE 3: Context Extension (2B tokens, seq=4096)")
        self._enable_yarn_scaling(factor=2.0)
        self._set_local_window(128)
        ga_steps_p3 = max(1, self.config.phase3_global_batch_tokens // (
            self.config.micro_batch_size_per_gpu * 4096 * self.world_size
        ))
        lr_const_scheduler = ConstantLRScheduler(optimizer, lr=3e-5)
        self.run_phase(
            phase=3, dataloader=phase3_dataloader, optimizer=optimizer,
            scheduler=lr_const_scheduler, target_tokens=self.config.phase3_tokens,
            seq_len=4096, gradient_accumulation_steps=ga_steps_p3,
        )
        self._save_checkpoint(3, self._global_step, self._tokens_processed)

        # ── Final copy ────────────────────────────────────────────────────────
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
        logger.info(
            f"  Tokens: {self._tokens_processed/1e9:.1f}B | Steps: {self._global_step:,} | "
            f"Time: {elapsed_h:.1f}h | Cost: ${total_cost:.2f} of ${self._budget:.2f}"
        )
        logger.info(f"{'='*70}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER CONFIGURATION HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _set_local_window(self, window_size: int):
        """Update ARG local attention window size."""
        from model.arg_layer import ARGLayer
        for layer in self.raw_model.layers:
            if isinstance(layer.seq_mixer, ARGLayer):
                layer.seq_mixer.local_window = window_size
        logger.info(f"ARG local window updated to {window_size}")

    def _enable_yarn_scaling(self, factor: float = 2.0):
        """Enable YaRN RoPE scaling in MHA layers for context extension."""
        from model.mha_layer import GlobalMHALayer
        from model.rotary_embedding import RotaryEmbedding
        for layer in self.raw_model.layers:
            if isinstance(layer.seq_mixer, GlobalMHALayer):
                layer.seq_mixer.rope = RotaryEmbedding(
                    head_dim=self.model_config.mha_head_dim,
                    max_seq_len=self.model_config.max_seq_len * 2,
                    base=self.model_config.rope_theta,
                    scaling_type="yarn",
                    scaling_factor=factor,
                ).to(self.device)
        logger.info(f"YaRN RoPE scaling enabled: factor={factor}")