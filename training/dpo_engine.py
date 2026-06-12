"""
DPO (Direct Preference Optimization) Training Engine for FORGE-3B.

Implements DPO with:
- Reference model frozen in BF16 (no optimizer states)
- Policy model trained with ZeRO-3
- Both models on same GPUs (reference sharded via ZeRO-3 inference mode)
- IPO and cDPO variants supported
- Reward logging for preference analysis
"""

from __future__ import annotations
import json
import time
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """
    Dataset for DPO training.
    Each sample: chosen conversation + rejected conversation.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        self.samples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.samples.append(item)
        
        logger.info(f"PreferenceDataset: {len(self.samples)} preference pairs")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        
        prompt_msgs = item.get("prompt", item.get("messages_prompt", []))
        chosen_msgs = item.get("chosen", [])
        rejected_msgs = item.get("rejected", [])
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode_chat(
            prompt_msgs, add_generation_prompt=True, return_as_tensor=False
        )
        
        # Encode chosen completion
        chosen_ids = self.tokenizer.encode(
            chosen_msgs if isinstance(chosen_msgs, str) else chosen_msgs[-1].get("content", ""),
            add_bos=False, add_eos=True, truncate=True,
        )
        
        # Encode rejected completion
        rejected_ids = self.tokenizer.encode(
            rejected_msgs if isinstance(rejected_msgs, str) else rejected_msgs[-1].get("content", ""),
            add_bos=False, add_eos=True, truncate=True,
        )
        
        max_len = self.seq_len
        
        def _make_input_labels(prompt: List[int], completion: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
            ids = (prompt + completion)[:max_len]
            labels = ([-100] * len(prompt) + completion)[:max_len]
            # Pad to max_len
            pad_len = max_len - len(ids)
            ids = ids + [0] * pad_len
            labels = labels + [-100] * pad_len
            return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        
        chosen_input, chosen_labels = _make_input_labels(prompt_ids, chosen_ids)
        rejected_input, rejected_labels = _make_input_labels(prompt_ids, rejected_ids)
        
        return {
            "chosen_input_ids": chosen_input,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input,
            "rejected_labels": rejected_labels,
        }


def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # (B,)
    policy_rejected_logps: torch.Tensor,  # (B,)
    ref_chosen_logps: torch.Tensor,       # (B,)
    ref_rejected_logps: torch.Tensor,     # (B,)
    beta: float = 0.1,
    loss_type: str = "dpo",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DPO loss computation.
    
    DPO:  L = -E[log σ(β(log π_θ(y_w|x) - log π_ref(y_w|x)) - 
                        β(log π_θ(y_l|x) - log π_ref(y_l|x)))]
    
    Returns: (loss, chosen_rewards, rejected_rewards)
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    
    if loss_type == "dpo":
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(beta * logits).mean()
    
    elif loss_type == "ipo":
        # Identity Preference Optimization (more stable than DPO)
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        loss = (pi_logratios - ref_logratios - 1.0 / (2.0 * beta)).pow(2).mean()
    
    elif loss_type == "cdpo":
        # Conservative DPO with label smoothing
        eps = 0.1
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        loss = -(1 - eps) * F.logsigmoid(beta * logits).mean() - \
                eps * F.logsigmoid(-beta * logits).mean()
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss, chosen_rewards, rejected_rewards


def get_log_probs(
    logits: torch.Tensor,   # (B, T, vocab_size)
    labels: torch.Tensor,   # (B, T) with -100 for ignored positions
) -> torch.Tensor:
    """Compute per-sequence sum of log-probabilities for non-masked tokens."""
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].float().contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()               # (B, T-1)
    
    # Log-softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)          # (B, T-1, V)
    
    # Gather log-probs at label positions
    mask = (shift_labels != -100)                             # (B, T-1)
    
    # Replace -100 with 0 for gather (will be masked out)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    
    selected_log_probs = log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)                                             # (B, T-1)
    
    # Sum over non-masked positions, normalize by sequence length
    n_tokens = mask.float().sum(-1).clamp(min=1.0)           # (B,)
    seq_log_probs = (selected_log_probs * mask.float()).sum(-1) / n_tokens
    
    return seq_log_probs   # (B,)


class DPOEngine:
    """DPO training engine with frozen reference model."""
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        config,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        wandb_run=None,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = (rank == 0)
        self.device = torch.device(f"cuda:{local_rank}")
        self.wandb_run = wandb_run
        
        # Freeze reference model completely
        for param in self.ref.parameters():
            param.requires_grad_(False)
        self.ref.eval()
        
        self._global_step = 0
        self._start_time = time.time()
        self._hourly_cost = 6.58
        
        logger.info(f"DPO Engine: beta={config.beta}, loss={config.loss_type}, "
                    f"n_pairs={config.n_preference_pairs}")
    
    def train(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ):
        """Run DPO training."""
        from training.gpu_optimizer import bf16_autocast, clip_grad_norm_and_log
        
        self.policy.train()
        
        all_rewards_chosen = []
        all_rewards_rejected = []
        all_accuracies = []
        
        for epoch in range(self.config.n_epochs):
            for batch in dataloader:
                
                chosen_input = batch["chosen_input_ids"].to(self.device, non_blocking=True)
                chosen_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
                rejected_input = batch["rejected_input_ids"].to(self.device, non_blocking=True)
                rejected_labels = batch["rejected_labels"].to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                # ── Policy forward passes ──────────────────────────────────
                with bf16_autocast(enabled=self.config.bf16):
                    # Process chosen + rejected in one batch for efficiency
                    combined_input = torch.cat([chosen_input, rejected_input], dim=0)
                    combined_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
                    
                    policy_outputs = self.policy(
                        input_ids=combined_input,
                        labels=combined_labels,
                        return_aux_loss=False,
                    )
                    
                    B = chosen_input.shape[0]
                    policy_logits = policy_outputs["logits"]
                    
                    policy_chosen_logps = get_log_probs(policy_logits[:B], chosen_labels)
                    policy_rejected_logps = get_log_probs(policy_logits[B:], rejected_labels)
                
                # ── Reference model forward (no grad) ─────────────────────
                with torch.no_grad():
                    with bf16_autocast(enabled=self.config.bf16):
                        ref_outputs = self.ref(
                            input_ids=combined_input,
                            return_aux_loss=False,
                        )
                        ref_logits = ref_outputs["logits"]
                        ref_chosen_logps = get_log_probs(ref_logits[:B], chosen_labels)
                        ref_rejected_logps = get_log_probs(ref_logits[B:], rejected_labels)
                
                # ── DPO Loss ───────────────────────────────────────────────
                loss, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    beta=self.config.beta,
                    loss_type=self.config.loss_type,
                )
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (self._global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    grad_norm = clip_grad_norm_and_log(
                        self.policy.parameters(), self.config.grad_clip
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                
                # ── Metrics ────────────────────────────────────────────────
                accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
                margin = (chosen_rewards - rejected_rewards).mean().item()
                
                all_accuracies.append(accuracy)
                all_rewards_chosen.append(chosen_rewards.mean().item())
                all_rewards_rejected.append(rejected_rewards.mean().item())
                
                self._global_step += 1
                
                if self.is_main and self._global_step % 10 == 0:
                    elapsed_h = (time.time() - self._start_time) / 3600
                    cost = elapsed_h * self._hourly_cost
                    
                    avg_acc = sum(all_accuracies[-50:]) / max(1, len(all_accuracies[-50:]))
                    avg_margin = (
                        sum(all_rewards_chosen[-50:]) - sum(all_rewards_rejected[-50:])
                    ) / max(1, len(all_rewards_chosen[-50:]))
                    
                    logger.info(
                        f"DPO Step {self._global_step} | "
                        f"Loss {loss.item()*self.config.gradient_accumulation_steps:.4f} | "
                        f"Accuracy {avg_acc:.3f} | "
                        f"Margin {avg_margin:.4f} | "
                        f"Cost ${cost:.2f}"
                    )
                    
                    if self.wandb_run:
                        self.wandb_run.log({
                            "dpo/loss": loss.item() * self.config.gradient_accumulation_steps,
                            "dpo/accuracy": avg_acc,
                            "dpo/margin": avg_margin,
                            "dpo/chosen_reward": chosen_rewards.mean().item(),
                            "dpo/rejected_reward": rejected_rewards.mean().item(),
                        }, step=self._global_step)
                
                if self._global_step % self.config.save_every_n_steps == 0 and self.is_main:
                    ckpt_dir = Path(self.config.output_dir) / f"dpo_step{self._global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.policy.state_dict(), str(ckpt_dir / "model.pt"))
        
        # Save final
        if self.is_main:
            final_dir = Path(self.config.output_dir) / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.policy.state_dict(), str(final_dir / "model.pt"))
            self.tokenizer.save_pretrained(str(final_dir))
        
        elapsed_h = (time.time() - self._start_time) / 3600
        final_acc = sum(all_accuracies) / max(1, len(all_accuracies))
        logger.info(f"DPO complete: {elapsed_h:.1f}h, ${elapsed_h*self._hourly_cost:.2f}, "
                    f"final_accuracy={final_acc:.3f}")