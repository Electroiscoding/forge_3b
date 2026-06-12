"""Learning rate schedulers for FORGE training phases."""

from __future__ import annotations
import math
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """
    Cosine LR decay with linear warmup.
    Operates in terms of training tokens (not steps) for phase alignment.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float,
        lr_min: float,
        warmup_tokens: int,
        total_tokens: int,
        group_multipliers: Optional[dict] = None,
    ):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self.group_multipliers = group_multipliers or {}
        
        self._tokens_seen = 0
    
    def step(self, n_tokens: int):
        """Update LR after processing n_tokens tokens."""
        self._tokens_seen += n_tokens
        t = self._tokens_seen
        
        if t < self.warmup_tokens:
            # Linear warmup
            lr = self.lr_min + (self.lr_max - self.lr_min) * (t / self.warmup_tokens)
        else:
            # Cosine decay
            progress = (t - self.warmup_tokens) / max(1, self.total_tokens - self.warmup_tokens)
            progress = min(1.0, progress)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.lr_min + (self.lr_max - self.lr_min) * cosine_factor
        
        # Update each parameter group with its multiplier
        for group in self.optimizer.param_groups:
            name = group.get("name", "main")
            mult = self.group_multipliers.get(name, 1.0)
            group["lr"] = lr * mult
        
        return lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
    
    def state_dict(self) -> dict:
        return {"tokens_seen": self._tokens_seen, "lr_max": self.lr_max, 
                "lr_min": self.lr_min}
    
    def load_state_dict(self, state: dict):
        self._tokens_seen = state["tokens_seen"]


class ConstantLRScheduler:
    """Constant LR for DPO/fine-tuning phases."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, lr: float):
        self.optimizer = optimizer
        self.lr = lr
        for group in optimizer.param_groups:
            group["lr"] = lr
    
    def step(self, n_tokens: int = 0):
        return self.lr
    
    def get_lr(self) -> float:
        return self.lr
    
    def state_dict(self) -> dict:
        return {"lr": self.lr}
    
    def load_state_dict(self, state: dict):
        pass