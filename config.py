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
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True          # auto-tune cuDNN kernels
    try:
        torch.backends.cuda.enable_flash_sdp(True)     # Flash SDP in PyTorch 2.x
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)     # disable slow math SDP
    except Exception:
        pass

# NCCL tuning for 64x H100 GPU cluster (8x 8-GPU nodes via NVLink + 3.2 Tbps InfiniBand)
os.environ.setdefault("NCCL_MIN_NCHANNELS", "32")
os.environ.setdefault("NCCL_MAX_NCHANNELS", "64")
os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "8")
os.environ.setdefault("NCCL_SOCKET_NTHREADS", "8")
os.environ.setdefault("NCCL_BUFFSIZE", "16777216")      # 16MB ring buffer for high-bandwidth AllReduce
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "5")        # GPUDirect RDMA Level 5 for NVLink + InfiniBand
os.environ.setdefault("NCCL_CROSS_NIC", "1")            # Multi-NIC rail-optimized routing
os.environ.setdefault("NCCL_IB_DISABLE", "0")           # InfiniBand enabled
os.environ.setdefault("NCCL_P2P_DISABLE", "0")          # NVLink P2P enabled
os.environ.setdefault("NCCL_NVLS_ENABLE", "1")          # NVLink SHARP collective offload
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1") # Precise CUDA stream scheduling

# CUDA memory allocator — reduce fragmentation for large model + optimizer states
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8")

# Tokenizer parallelism — disable HF tokenizer parallelism (we use CRAYON)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForgeModelConfig:
    """Complete FORGE-1B architecture configuration. 991.6M total params (<= 1.0B)."""
    
    # ── Core Dimensions ──────────────────────────────────────────────────────
    vocab_size: int = 206_464          # CRAYON standard profile (206,373) padded to ×128
    d_model: int = 1280               # 1B: 1280 (<= 1.0B total params)
    n_layers: int = 24                # 1B: 24 layers (18 ARG + 6 MHA)
    max_seq_len: int = 4096
    
    # ── Layer Distribution ────────────────────────────────────────────────────
    # Layer 0,1,2 = ARG; Layer 3 = MHA; repeat ×6
    mha_layer_indices: List[int] = field(default_factory=lambda: [3,7,11,15,19,23])
    # Dense FFN on odd layers, HSE on even layers
    dense_ffn_layer_indices: List[int] = field(default_factory=lambda: list(range(1,24,2)))
    hse_ffn_layer_indices: List[int] = field(default_factory=lambda: list(range(0,24,2)))
    
    # ── ARG (Adaptive Recurrent Gating) ──────────────────────────────────────
    arg_d_inner: int = 1280           # matches d_model for 1B
    arg_d_state: int = 48
    arg_d_rank: int = 48               # dt projection rank
    arg_conv_kernel: int = 4
    arg_local_window: int = 64         # local attention window (phase 1+2)
    arg_local_n_heads: int = 8
    arg_local_n_kv_heads: int = 2
    arg_head_dim: int = 80            # 1280 / 16 heads
    arg_scalar_gate: bool = True       # scalar gate per token (vs d_model-dim)
    
    # ── MHA (Multi-Head Attention) ────────────────────────────────────────────
    mha_n_heads: int = 16
    mha_n_kv_heads: int = 4            # GQA: 4:1 ratio
    mha_head_dim: int = 80            # 1280 / 16 heads
    rope_theta: float = 500_000.0      # long-context RoPE base
    rope_scaling_type: Optional[str] = None  # None | "yarn"
    rope_scaling_factor: float = 1.0
    
    # ── Dense FFN (SwiGLU) ────────────────────────────────────────────────────
    dense_d_ff: int = 3200             # 2.5 × d_model, multiple of 128
    
    # ── HSE FFN (Hierarchical Sparse Expert) ──────────────────────────────────
    hse_n_domains: int = 4
    hse_n_experts_per_domain: int = 8  # total 32 experts
    hse_top_k: int = 2                 # active per token
    hse_d_ff_expert: int = 288        # scaled for <= 1.0B total params
    hse_capacity_factor: float = 1.25
    hse_expert_dropout: float = 0.1
    hse_aux_loss_alpha: float = 0.01
    hse_gumbel_tau_init: float = 1.0
    hse_gumbel_tau_final: float = 0.1
    hse_gumbel_tau_warmup_tokens: int = 10_000_000_000  # anneal over 10B tokens
    
    # ── Normalization ─────────────────────────────────────────────────────────
    norm_type: str = "dgn"             # "dgn" or "rmsnorm"
    dgn_n_groups: int = 16            # 1280 / 16 = 80 per group
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
    
    # ── Compile/Optimization Flags ────────────────────────────────────
    use_flash_attention: bool = True
    use_triton_kernels: bool = True    # Triton fused ops
    use_torch_compile: bool = True
    compile_mode: str = "max-autotune"  # max throughput on H100
    use_gradient_checkpointing: bool = False  # 1B fits in 80GB without checkpointing → faster
    gradient_checkpointing_ratio: float = 0.0
    
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
    """Phase-aware pretraining configuration for FORGE-1B.
    
    Budget math (1x H100 SXM @ $3.29/hr):
      - Phase 1 (vocab warmup):  2B tokens @ 80k tok/s  →  ~7h  → $23
      - Phase 2 (core pretrain): 16B tokens @ 80k tok/s → ~56h  → $184
      - Phase 3 (ctx extension):  2B tokens @ 60k tok/s →  ~9h  → $30
      PRETRAIN TOTAL: ~72h → ~$237
      SFT:  1.4B tokens → ~5h → $16
      DPO:  0.8B tokens → ~3h → $10
      GRAND TOTAL: ~$263 of $400 budget. $137 buffer.
    """
    
    output_dir: str = "./checkpoints/forge_1b_pretrain"
    data_dir: str = "Phase-Technologies/forge-3b-pretrain-data"  # same tokenized data
    resume_from_checkpoint: Optional[str] = None
    
    # ── Token Budget (Strictly locked to $400 Budget: $384 total cost @ $126.34/hr) ──
    # 11B tokens directly in Phase 2 at seq=2048 (Phase 1 & Phase 3 omitted)
    phase1_tokens: int = 0                   # Omitted (start directly on Phase 2 @ seq=2048)
    phase2_tokens: int = 11_000_000_000      # 11.0B core pretrain (seq=2048) -> ~2.8h ($357)
    phase3_tokens: int = 0                   # Omitted (pure seq=2048 pretraining)
    total_tokens: int = 11_000_000_000       # 11.0B total pretrain
    
    # ── Batch Config (Tuned for 32x H100 SXM — 1,048,576 tokens/step) ─────────
    # With 32 GPUs @ micro_batch=16 (seq=2048): 32 × 16 × 2048 = 1,048,576 tokens per pass!
    # Exactly 1 gradient accumulation step (zero idle wait, maximum NVLink saturation)
    phase1_global_batch_tokens: int = 1_048_576   # 1M tokens/step
    phase2_global_batch_tokens: int = 1_048_576   # 1M tokens/step (Phase 2)
    phase3_global_batch_tokens: int = 1_048_576   # 1M tokens/step (Phase 3)
    micro_batch_size_per_gpu: int = 16            # default fallback
    phase1_micro_batch_per_gpu: int = 64          # 64 seqs × 512 = 32,768 tokens/GPU
    phase2_micro_batch_per_gpu: int = 16          # 16 seqs × 2048 = 32,768 tokens/GPU (1 accum step across 32 GPUs)
    phase3_micro_batch_per_gpu: int = 8           # 8 seqs × 4096 = 32,768 tokens/GPU
    
    # ── Context Lengths ───────────────────────────────────────────────────────
    phase1_seq_len: int = 512
    phase2_seq_len: int = 2048
    phase3_seq_len: int = 2048
    
    # ── Learning Rate ─────────────────────────────────────────────────────────
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    lr_warmup_tokens: int = 500_000_000      # 500M tokens warmup directly in Phase 2
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
    save_every_n_tokens: int = 1_000_000_000  # checkpoint every 1B tokens
    keep_last_n_checkpoints: int = 5
    
    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps: int = 1           # log every step for full metrics visibility
    eval_every_n_steps: int = 200        # eval perplexity every 200 steps
    wandb_project: str = "forge_1b_pretrain"
    wandb_entity: Optional[str] = None
    
    # ── GPU/Distributed ───────────────────────────────────────────────────────
    num_gpus: int = 1                          # default 1x H100; set to 16 for 16x
    bf16: bool = True
    fp8: bool = False                          # H100 FP8 (experimental)
    deepspeed_config: str = "./configs/ds_zero3.json"
    torch_compile: bool = True
    compile_mode: str = "max-autotune"         # peak throughput
    
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
    """Supervised Fine-Tuning configuration for FORGE-1B."""
    
    base_model_path: str = "./checkpoints/forge_1b_pretrain/final"
    output_dir: str = "./checkpoints/forge_1b_sft"
    data_dir: str = "Phase-Technologies/forge-3b-sft-data"
    
    # ── Training ──────────────────────────────────────────────────────────────
    total_tokens: int = 1_400_000_000
    global_batch_tokens: int = 524_288  # 512K per step
    micro_batch_size_per_gpu: int = 4
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
    
    save_every_n_steps: int = 100
    wandb_project: str = "forge_1b_sft"
    deepspeed_config: str = "./configs/ds_zero3_sft.json"
    num_gpus: int = 1
    bf16: bool = True
    torch_compile: bool = True
    compile_mode: str = "max-autotune"


@dataclass  
class DPOConfig:
    """Direct Preference Optimization configuration for FORGE-1B."""
    
    base_model_path: str = "./checkpoints/forge_1b_sft/final"
    output_dir: str = "./checkpoints/forge_1b_dpo"
    data_dir: str = "Phase-Technologies/forge-3b-dpo-data"
    
    # ── DPO ───────────────────────────────────────────────────────────────────
    beta: float = 0.1                  # KL penalty coefficient
    loss_type: str = "dpo"             # "dpo" | "ipo" | "cdpo"
    reference_free: bool = False
    
    # ── Training ──────────────────────────────────────────────────────────────
    n_preference_pairs: int = 200_000
    batch_size_pairs: int = 8          # preference pairs per GPU step
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
    wandb_project: str = "forge_1b_dpo"
    deepspeed_config: str = "./configs/ds_zero3_sft.json"
    num_gpus: int = 1
    bf16: bool = True
    torch_compile: bool = True
    compile_mode: str = "max-autotune"


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