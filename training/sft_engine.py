"""
FORGE-3B Supervised Fine-Tuning Engine.

SFT features:
- Chat template encoding with per-token loss masking
- Loss only on assistant tokens (default)
- Cosine LR from pretrained LR to min
- Full BF16 + ZeRO-3
- Packing of short conversations for GPU efficiency
"""

from __future__ import annotations
import os
import json
import time
import math
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

from training.hub_uploader import upload_folder_async

import numpy as np


class PackedSFTDataset(Dataset):
    """
    Dataset for pre-tokenized SFT data stored as .npz shards.
    
    Expected format (from Phase-Technologies/forge-3b-sft-data):
        Each .npz file contains:
          - input_ids:  (N, seq_len) uint32  — tokenized sequences
          - loss_mask:  (N, seq_len) uint8   — 1 = compute loss, 0 = ignore
    
    Converts loss_mask to labels: where mask=1 → token id, where mask=0 → -100.
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4096,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        data_path = Path(data_dir)
        
        # Find all .npz shard files recursively
        npz_files = sorted(data_path.glob("**/*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        
        # Load all shards (memory-mapped where possible)
        all_ids = []
        all_masks = []
        for f in npz_files:
            try:
                data = np.load(str(f))
                ids = data["input_ids"]    # (N, seq_len) uint32
                mask = data["loss_mask"]   # (N, seq_len) uint8
                
                if ids.shape[1] != seq_len:
                    logger.warning(f"Shard {f} has seq_len={ids.shape[1]}, expected {seq_len}. Skipping.")
                    continue
                
                all_ids.append(ids)
                all_masks.append(mask)
                logger.info(f"  Loaded SFT shard {f.name}: {len(ids):,} samples")
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        if not all_ids:
            raise RuntimeError(f"No valid .npz shards loaded from {data_dir}")
        
        self.input_ids = np.concatenate(all_ids, axis=0)    # (total_N, seq_len)
        self.loss_mask = np.concatenate(all_masks, axis=0)   # (total_N, seq_len)
        
        if max_samples and max_samples < len(self.input_ids):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.input_ids), max_samples, replace=False)
            self.input_ids = self.input_ids[indices]
            self.loss_mask = self.loss_mask[indices]
        
        logger.info(
            f"PackedSFTDataset: {len(self.input_ids):,} samples × {seq_len} tokens, "
            f"loss coverage: {self.loss_mask.mean():.1%}"
        )
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.input_ids[idx].astype(np.int64)
        mask = self.loss_mask[idx]
        
        input_ids = torch.from_numpy(ids)
        # Labels: token id where mask=1, -100 where mask=0
        labels = input_ids.clone()
        labels[mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class SFTDataset(Dataset):
    """
    SFT dataset that formats conversations with FORGE chat template.
    Supports packing short conversations into seq_len windows.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int = 4096,
        loss_on_prompt: bool = False,
        pack_sequences: bool = True,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.loss_on_prompt = loss_on_prompt
        self.pack_sequences = pack_sequences
        
        # Load conversations
        self.conversations = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.conversations.append(json.loads(line))
        
        # Pre-tokenize and pack
        if pack_sequences:
            self.samples = self._pack_conversations()
        else:
            self.samples = [
                self.tokenizer.encode_chat_with_labels(
                    conv["messages"],
                    loss_on_prompt=loss_on_prompt,
                )
                for conv in self.conversations
            ]
        
        logger.info(f"SFTDataset: {len(self.conversations)} conversations → "
                    f"{len(self.samples)} packed samples")
    
    def _pack_conversations(self) -> List[Dict[str, torch.Tensor]]:
        """Pack multiple short conversations into seq_len windows."""
        packed = []
        current_ids = []
        current_labels = []
        
        for conv in self.conversations:
            sample = self.tokenizer.encode_chat_with_labels(
                conv.get("messages", conv.get("conversation", [])),
                loss_on_prompt=self.loss_on_prompt,
            )
            ids = sample["input_ids"].tolist()
            labels = sample["labels"].tolist()
            
            # If adding this conversation would exceed seq_len, flush
            if len(current_ids) + len(ids) > self.seq_len and current_ids:
                # Pad to seq_len
                pad_len = self.seq_len - len(current_ids)
                current_ids.extend([0] * pad_len)
                current_labels.extend([-100] * pad_len)
                
                packed.append({
                    "input_ids": torch.tensor(current_ids[:self.seq_len], dtype=torch.long),
                    "labels": torch.tensor(current_labels[:self.seq_len], dtype=torch.long),
                })
                current_ids, current_labels = [], []
            
            current_ids.extend(ids)
            current_labels.extend(labels)
        
        # Flush remainder
        if current_ids:
            pad_len = self.seq_len - len(current_ids) % self.seq_len
            if pad_len < self.seq_len:
                current_ids.extend([0] * pad_len)
                current_labels.extend([-100] * pad_len)
            
            for i in range(0, len(current_ids) - self.seq_len + 1, self.seq_len):
                packed.append({
                    "input_ids": torch.tensor(current_ids[i:i+self.seq_len], dtype=torch.long),
                    "labels": torch.tensor(current_labels[i:i+self.seq_len], dtype=torch.long),
                })
        
        return packed
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class SFTEngine:
    """Supervised Fine-Tuning engine for FORGE-3B."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        wandb_run=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = (rank == 0)
        self.device = torch.device(f"cuda:{local_rank}")
        self.wandb_run = wandb_run
        
        self._global_step = 0
        self._start_time = time.time()
        self._hourly_cost = 63.17
    
    def train(
        self,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ):
        """Run SFT training loop."""
        from training.gpu_optimizer import bf16_autocast, clip_grad_norm_and_log
        
        self.model.train()
        
        target_tokens = self.config.total_tokens
        tokens_processed = 0
        
        # GA steps: for SFT with 256K batch
        ga_steps = max(1, self.config.global_batch_tokens // (
            self.config.micro_batch_size_per_gpu * self.config.seq_len * self.world_size
        ))
        
        logger.info(f"SFT: target={target_tokens/1e9:.1f}B tokens, "
                    f"GA steps={ga_steps}, seq_len={self.config.seq_len}")
        
        data_iter = iter(train_dataloader)
        
        while tokens_processed < target_tokens:
            if self.is_main:
                logger.info(f"Starting SFT Step {self._global_step + 1} (accumulating {ga_steps} micro-batches)...")
            optimizer.zero_grad(set_to_none=True)
            
            accum_loss = 0.0
            
            for accum_step in range(ga_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_dataloader)
                    batch = next(data_iter)
                
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                
                with bf16_autocast(enabled=self.config.bf16):
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        return_aux_loss=False,  # no MoE loss during SFT
                    )
                    loss = outputs["loss"] / ga_steps
                
                loss.backward()
                accum_loss += loss.item() * ga_steps
            
            grad_norm = clip_grad_norm_and_log(self.model.parameters(), self.config.grad_clip)
            optimizer.step()
            
            batch_tokens = self.config.micro_batch_size_per_gpu * self.config.seq_len * \
                           self.world_size * ga_steps
            scheduler.step(batch_tokens)
            tokens_processed += batch_tokens
            self._global_step += 1
            
            if self.is_main and self._global_step % 10 == 0:
                elapsed_h = (time.time() - self._start_time) / 3600
                cost = elapsed_h * self._hourly_cost
                logger.info(
                    f"SFT Step {self._global_step} | "
                    f"Loss {accum_loss:.4f} | "
                    f"LR {scheduler.get_lr():.2e} | "
                    f"Cost ${cost:.2f}"
                )
                if self.wandb_run:
                    self.wandb_run.log({
                        "sft/loss": accum_loss,
                        "sft/lr": scheduler.get_lr(),
                        "sft/grad_norm": grad_norm,
                    }, step=self._global_step)
            
            if self._global_step % self.config.save_every_n_steps == 0 and self.is_main:
                ckpt_dir = Path(self.config.output_dir) / f"sft_step{self._global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(ckpt_dir / "model.pt"))
                logger.info(f"SFT checkpoint saved: {ckpt_dir}")
                # Upload checkpoint
                upload_folder_async(str(ckpt_dir), repo_name="forge-3b-sft", folder_in_repo=ckpt_dir.name)
        
        # Save final
        if self.is_main:
            final_dir = Path(self.config.output_dir) / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), str(final_dir / "model.pt"))
            self.tokenizer.save_pretrained(str(final_dir))
            # Upload final model
            upload_folder_async(str(final_dir), repo_name="forge-3b-sft", folder_in_repo="final")
        
        elapsed_h = (time.time() - self._start_time) / 3600
        logger.info(f"SFT complete: {elapsed_h:.1f}h, ${elapsed_h*self._hourly_cost:.2f}")