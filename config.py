"""
FORGE-3B: Master Configuration
All hyperparameters, paths, and GPU optimization settings in one place.
"""

from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# GPU GLOBAL OPTIMIZATION FLAGS
# These are applied at import time across the entire process.
# ─────────────────────────────────────────────────────────────────────────────

import torch

# Allow TF32 for matmuls and convolutions — massive speedup on Ampere/Hopper
# with negligible precision loss for language model training.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True          # auto-tune cuDNN kernels
torch.backends.cuda.enable_flash_sdp(True)     # Flash SDP in PyTorch 2.x
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)     # disable slow math SDP

# NCCL tuning for multi-GPU
os.environ.setdefault("NCCL_MIN_NCHANNELS", "4")
os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")
os.environ.setdefault("NCCL_SOCKET_NTHREADS", "4")
os.environ.setdefault("NCCL_IB_DISABLE", "0")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")

# CUDA memory allocator — reduce fragmentation for large model + optimizer states
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.8")

# Tokenizer parallelism — disable HF tokenizer parallelism (we use CRAYON)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForgeModelConfig:
    """Complete FORGE-3B architecture configuration."""
    
    # ── Core Dimensions ──────────────────────────────────────────────────────
    vocab_size: int = 206_464          # CRAYON standard profile (206,373) padded to ×128
    d_model: int = 2048
    n_layers: int = 36
    max_seq_len: int = 4096
    
    # ── Layer Distribution ────────────────────────────────────────────────────
    # Layer 0,1,2 = ARG; Layer 3 = MHA; repeat ×9
    mha_layer_indices: List[int] = field(default_factory=lambda: [3,7,11,15,19,23,27,31,35])
    # Dense FFN on odd layers, HSE on even layers
    dense_ffn_layer_indices: List[int] = field(default_factory=lambda: list(range(1,36,2)))
    hse_ffn_layer_indices: List[int] = field(default_factory=lambda: list(range(0,36,2)))
    
    # ── ARG (Adaptive Recurrent Gating) ──────────────────────────────────────
    arg_d_inner: int = 2048
    arg_d_state: int = 64
    arg_d_rank: int = 64               # dt projection rank
    arg_conv_kernel: int = 4
    arg_local_window: int = 64         # local attention window (phase 1+2)
    arg_local_n_heads: int = 8
    arg_local_n_kv_heads: int = 2
    arg_head_dim: int = 128
    arg_scalar_gate: bool = True       # scalar gate per token (vs d_model-dim)
    
    # ── MHA (Multi-Head Attention) ────────────────────────────────────────────
    mha_n_heads: int = 16
    mha_n_kv_heads: int = 4            # GQA: 4:1 ratio
    mha_head_dim: int = 128
    rope_theta: float = 500_000.0      # long-context RoPE base
    rope_scaling_type: Optional[str] = None  # None | "yarn"
    rope_scaling_factor: float = 1.0
    
    # ── Dense FFN (SwiGLU) ────────────────────────────────────────────────────
    dense_d_ff: int = 5504             # ≈ 2.69 × d_model, multiple of 128
    
    # ── HSE FFN (Hierarchical Sparse Expert) ──────────────────────────────────
    hse_n_domains: int = 4
    hse_n_experts_per_domain: int = 8  # total 32 experts
    hse_top_k: int = 2                 # active per token
    hse_d_ff_expert: int = 512
    hse_capacity_factor: float = 1.25
    hse_expert_dropout: float = 0.1
    hse_aux_loss_alpha: float = 0.01
    hse_gumbel_tau_init: float = 1.0
    hse_gumbel_tau_final: float = 0.1
    hse_gumbel_tau_warmup_tokens: int = 20_000_000_000  # anneal over 20B tokens
    
    # ── Normalization ─────────────────────────────────────────────────────────
    norm_type: str = "dgn"             # "dgn" or "rmsnorm"
    dgn_n_groups: int = 16
    norm_eps: float = 1e-6
    
    # ── Initialization ────────────────────────────────────────────────────────
    init_std: float = 0.02
    init_std_residual_scale: bool = True  # scale by 1/sqrt(2n_layers)
    
    # ── Misc ──────────────────────────────────────────────────────────────────
    dropout: float = 0.0               # 0.0 during pretraining
    tie_word_embeddings: bool = True
    use_cache: bool = True             # KV cache for inference
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # ── Compile/Optimization Flags ────────────────────────────────────────────
    use_flash_attention: bool = True
    use_triton_kernels: bool = True    # Triton fused ops
    use_torch_compile: bool = True
    compile_mode: str = "default" # "default"|"reduce-overhead"|"max-autotune"
    use_gradient_checkpointing: bool = True
    gradient_checkpointing_ratio: float = 0.5  # checkpoint every 2nd layer
    
    def n_params_total(self) -> int:
        """Approximate total parameter count."""
        embed = self.vocab_size * self.d_model
        n_arg = len([i for i in range(self.n_layers) if i not in self.mha_layer_indices])
        n_mha = len(self.mha_layer_indices)
        n_dense = len(self.dense_ffn_layer_indices)
        n_hse = len(self.hse_ffn_layer_indices)
        
        arg_per = (
            self.d_model * 2 * self.arg_d_inner +      # in_proj
            self.arg_d_inner * self.arg_conv_kernel +   # conv1d
            self.arg_d_inner * (self.arg_d_rank + 2 * self.arg_d_state) +  # x_proj
            self.arg_d_rank * self.arg_d_inner +        # dt_proj
            self.arg_d_state * 2 +                      # nu, theta
            self.arg_d_inner +                          # D
            self.arg_d_inner * self.d_model +           # out_proj
            self.arg_local_n_heads * self.arg_head_dim * self.d_model +  # Q
            self.arg_local_n_kv_heads * self.arg_head_dim * self.d_model * 2 +  # KV
            self.arg_local_n_heads * self.arg_head_dim * self.d_model +  # O
            self.d_model * self.arg_d_state             # CPB
        )
        
        mha_per = (
            self.d_model * self.mha_n_heads * self.mha_head_dim +      # Q
            self.d_model * self.mha_n_kv_heads * self.mha_head_dim * 2 +  # KV
            self.d_model * self.mha_n_heads * self.mha_head_dim         # O
        )
        
        dense_per = 3 * self.d_model * self.dense_d_ff
        
        n_experts_total = self.hse_n_domains * self.hse_n_experts_per_domain
        hse_per = n_experts_total * 3 * self.d_model * self.hse_d_ff_expert
        
        total = embed + n_arg * arg_per + n_mha * mha_per + n_dense * dense_per + n_hse * hse_per
        return total
    
    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "ForgeModelConfig":
        import inspect
        with open(path) as f:
            data = json.load(f)
        sig = inspect.signature(cls)
        filtered_data = {k: v for k, v in data.items() if k in sig.parameters}
        return cls(**filtered_data)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    """Phase-aware pretraining configuration."""
    
    output_dir: str = "./checkpoints/forge_3b_pretrain"
    data_dir: str = "Phase-Technologies/forge-3b-pretrain-data"
    resume_from_checkpoint: Optional[str] = None
    
    # ── Token Budget ──────────────────────────────────────────────────────────
    phase1_tokens: int = 5_000_000_000       # 5B
    phase2_tokens: int = 43_000_000_000      # 43B
    phase3_tokens: int = 2_000_000_000       # 2B
    total_tokens: int = 50_000_000_000       # 50B
    
    # ── Batch Config ──────────────────────────────────────────────────────────
    phase1_global_batch_tokens: int = 1_000_000
    phase2_global_batch_tokens: int = 2_000_000
    phase3_global_batch_tokens: int = 1_000_000
    micro_batch_size_per_gpu: int = 2        # sequences per GPU per step
    
    # ── Context Lengths ───────────────────────────────────────────────────────
    phase1_seq_len: int = 512
    phase2_seq_len: int = 2048
    phase3_seq_len: int = 4096
    
    # ── Learning Rate ─────────────────────────────────────────────────────────
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    lr_warmup_tokens: int = 5_000_000_000    # warm up over phase1
    lr_schedule: str = "cosine"
    
    # ── AdamW ─────────────────────────────────────────────────────────────────
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # ── Parameter Groups LR Multipliers ──────────────────────────────────────
    embedding_lr_mult: float = 0.5
    ssm_lr_mult: float = 0.3
    router_lr_mult: float = 0.5
    
    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_every_n_tokens: int = 2_000_000_000  # every 2B tokens
    keep_last_n_checkpoints: int = 5
    
    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    wandb_project: str = "forge_3b_pretrain"
    wandb_entity: Optional[str] = None
    
    # ── GPU/Distributed ───────────────────────────────────────────────────────
    num_gpus: int = 16
    bf16: bool = True
    fp8: bool = False                          # H100 FP8 (experimental)
    deepspeed_config: str = "./configs/ds_zero3.json"
    torch_compile: bool = True
    compile_mode: str = "default"
    
    # ── Data ──────────────────────────────────────────────────────────────────
    num_dataloader_workers: int = 4
    prefetch_factor: int = 4
    seed: int = 42
    
    @property
    def gradient_accumulation_steps_phase2(self) -> int:
        """Accumulation steps for phase 2 global batch."""
        tokens_per_gpu_step = self.micro_batch_size_per_gpu * self.phase2_seq_len
        tokens_per_step_all_gpus = tokens_per_gpu_step * self.num_gpus
        return self.phase2_global_batch_tokens // tokens_per_step_all_gpus


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration."""
    
    base_model_path: str = "./checkpoints/forge_3b_pretrain/final"
    output_dir: str = "./checkpoints/forge_3b_sft"
    data_dir: str = "Phase-Technologies/forge-3b-sft-data"
    
    # ── Training ──────────────────────────────────────────────────────────────
    total_tokens: int = 1_400_000_000
    global_batch_tokens: int = 256_000
    micro_batch_size_per_gpu: int = 1
    seq_len: int = 4096
    
    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_max: float = 1e-5
    lr_min: float = 1e-6
    lr_schedule: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 0.5
    
    epochs: int = 1
    loss_on_prompt: bool = False       # only compute loss on assistant turns
    
    save_every_n_steps: int = 200
    wandb_project: str = "forge_3b_sft"
    deepspeed_config: str = "./configs/ds_zero3_sft.json"
    num_gpus: int = 16
    bf16: bool = True
    torch_compile: bool = True
    compile_mode: str = "default"


@dataclass  
class DPOConfig:
    """Direct Preference Optimization configuration."""
    
    base_model_path: str = "./checkpoints/forge_3b_sft/final"
    output_dir: str = "./checkpoints/forge_3b_dpo"
    data_dir: str = "Phase-Technologies/forge-3b-dpo-data"
    
    # ── DPO ───────────────────────────────────────────────────────────────────
    beta: float = 0.1                  # KL penalty coefficient
    loss_type: str = "dpo"             # "dpo" | "ipo" | "cdpo"
    reference_free: bool = False
    
    # ── Training ──────────────────────────────────────────────────────────────
    n_preference_pairs: int = 200_000
    batch_size_pairs: int = 16         # preference pairs per GPU step
    gradient_accumulation_steps: int = 4
    seq_len: int = 4096
    
    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr: float = 5e-7
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 0.3
    n_epochs: int = 1
    
    save_every_n_steps: int = 100
    wandb_project: str = "forge_3b_dpo"
    deepspeed_config: str = "./configs/ds_zero3_sft.json"
    num_gpus: int = 16
    bf16: bool = True
    torch_compile: bool = True
    compile_mode: str = "default"


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrayonConfig:
    """CRAYON tokenizer configuration."""
    profile: str = "standard"          # "standard" (206k) | "lite" (50k)
    device: str = "cpu"                # "cpu" is faster for tokenization (23M vs 1.2M tok/s)
    pad_token: str = "<|pad|>"
    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"
    sep_token: str = "<|sep|>"
    sys_token: str = "<|sys|>"
    usr_token: str = "<|usr|>"
    asst_token: str = "<|asst|>"
    tool_token: str = "<|tool|>"
    tool_resp_token: str = "<|tool_resp|>"