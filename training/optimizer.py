"""
Optimizer factory with parameter group differentiation.
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    lr_max: float = 3e-4,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    embedding_lr_mult: float = 0.5,
    ssm_lr_mult: float = 0.3,
    router_lr_mult: float = 0.5,
    use_fused: bool = True,
    deepspeed_config_path: Optional[str] = None,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with parameter-group-specific learning rates.
    
    Groups:
    1. Embeddings             — lr × 0.5, no weight decay
    2. SSM state matrices     — lr × 0.3, no weight decay (sensitive params)
    3. MoE routers            — lr × 0.5, no weight decay
    4. Norms (scale/bias)     — lr × 1.0, no weight decay
    5. Everything else        — lr × 1.0, weight_decay applied
    """
    
    is_zero3 = any(hasattr(p, "ds_id") for p in model.parameters())
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)
    
    if is_zero3 and tie_word_embeddings:
        logger.info("DeepSpeed ZeRO-3 with weight tying detected — consolidating parameters into a single optimizer group to prevent duplicate bucket assertion crashes.")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [
            {
                "params": trainable_params,
                "lr": lr_max,
                "weight_decay": weight_decay,
                "name": "main",
            }
        ]
    else:
        # Categorize parameters
        embed_params = []
        ssm_params = []
        router_params = []
        no_decay_params = []    # norms
        decay_params = []       # main model weights
        
        no_decay_names = {"norm", "bias", "weight_g", "weight_b"}  # DGN names
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Embedding table
            if "embed_tokens" in name:
                embed_params.append(param)
            
            # SSM complex eigenvalue parameters (very sensitive)
            elif any(ssm_name in name for ssm_name in ["nu", "theta", ".A", ".B_log"]):
                ssm_params.append(param)
            
            # MoE router weights
            elif "tier1_router" in name or "tier2_router" in name:
                router_params.append(param)
            
            # Norm parameters (weight/bias) — no weight decay
            elif any(nd in name.split(".")[-1] for nd in ["weight", "bias"]) and \
                 any(norm_name in name for norm_name in ["norm", "dgn"]):
                no_decay_params.append(param)
            
            # Biases or 1D parameters (no weight decay)
            elif name.endswith(".bias") or (hasattr(param, "ds_shape") and len(param.ds_shape) <= 1) or (not hasattr(param, "ds_shape") and param.ndim <= 1):
                no_decay_params.append(param)
            
            # Everything else — weight decay
            else:
                decay_params.append(param)
        
        param_groups = [
            # Group 1: Main weights with decay
            {
                "params": decay_params,
                "lr": lr_max,
                "weight_decay": weight_decay,
                "name": "main",
            },
            # Group 2: Norms and biases — no decay
            {
                "params": no_decay_params,
                "lr": lr_max,
                "weight_decay": 0.0,
                "name": "no_decay",
            },
            # Group 3: Embedding table — lower LR, no decay
            {
                "params": embed_params,
                "lr": lr_max * embedding_lr_mult,
                "weight_decay": 0.0,
                "name": "embedding",
            },
            # Group 4: SSM eigenvalues — lower LR, no decay
            {
                "params": ssm_params,
                "lr": lr_max * ssm_lr_mult,
                "weight_decay": 0.0,
                "name": "ssm_state",
            },
            # Group 5: MoE routers — lower LR, no decay
            {
                "params": router_params,
                "lr": lr_max * router_lr_mult,
                "weight_decay": 0.0,
                "name": "moe_router",
            },
        ]
    
    # Filter empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    
    # Log parameter count per group
    for g in param_groups:
        n = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in g["params"])
        logger.info(f"Optimizer group '{g['name']}': {n/1e6:.1f}M params, "
                    f"lr={g['lr']:.2e}, wd={g['weight_decay']}")
    
    # Detect if CPU offloading is enabled in the DeepSpeed config
    offload_to_cpu = False
    if deepspeed_config_path:
        try:
            import json
            import os
            if os.path.exists(deepspeed_config_path):
                with open(deepspeed_config_path) as f:
                    ds_config = json.load(f)
                zero_opt = ds_config.get("zero_optimization", {})
                offload_opt = zero_opt.get("offload_optimizer", {})
                if offload_opt and offload_opt.get("device") == "cpu":
                    offload_to_cpu = True
        except Exception as e:
            logger.warning(f"Could not parse DeepSpeed config to check for optimizer offloading: {e}")

    # Instantiate optimizer based on CPU offload configuration
    if offload_to_cpu:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(
                param_groups,
                betas=(beta1, beta2),
                eps=eps,
            )
            logger.info("Using DeepSpeedCPUAdam for CPU offloading (high-performance AVX/C++)")
        except Exception as e:
            logger.warning(f"Failed to load DeepSpeedCPUAdam ({e}), falling back to standard PyTorch AdamW on CPU")
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(beta1, beta2),
                eps=eps,
            )
    else:
        # Try using 8-bit AdamW from bitsandbytes to keep optimizer memory within GPU limits (saves ~23GB)
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(
                param_groups,
                betas=(beta1, beta2),
                eps=eps,
            )
            logger.info("Using 8-bit AdamW (bitsandbytes) to fit states on GPU without OOM")
        except ImportError:
            # Fallback: Use fused Adam if available on GPU (torch 2.x — 30-50% faster than standard)
            try:
                if use_fused and torch.cuda.is_available():
                    optimizer = torch.optim.AdamW(
                        param_groups,
                        betas=(beta1, beta2),
                        eps=eps,
                        fused=True,   # CUDA-fused implementation
                    )
                    logger.info("Using fused AdamW (CUDA native)")
                else:
                    raise ImportError("fused not enabled")
            except (TypeError, ImportError):
                optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(beta1, beta2),
                    eps=eps,
                )
                logger.info("Using standard AdamW")
    
    return optimizer


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate (from first param group)."""
    return optimizer.param_groups[0]["lr"]