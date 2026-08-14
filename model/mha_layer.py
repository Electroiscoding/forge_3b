"""
Full Multi-Head Attention Layer for FORGE's global attention blocks.
Uses FlashAttention-3 when available, with GQA and RoPE.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dgn_norm import build_norm
from .rotary_embedding import RotaryEmbedding

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN = True
except ImportError:
    FLASH_ATTN = False


class GlobalMHALayer(nn.Module):
    """
    Global Multi-Head Attention with:
    - GQA (Grouped Query Attention) with n_kv_heads < n_heads
    - RoPE positional encoding (long-context base 500k)
    - FlashAttention-3 for maximum GPU utilization
    - KV cache support for efficient inference
    - Pre-norm with DGN
    """
    
    def __init__(
        self,
        d_model: int = 1280,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        head_dim: int = 80,
        rope_base: float = 500_000.0,
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: float = 1.0,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        norm_type: str = "dgn",
        dgn_n_groups: int = 16,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        layer_idx: int = 0,
    ):
        super().__init__()
        assert d_model == n_heads * head_dim, \
            f"d_model={d_model} must equal n_heads*head_dim={n_heads*head_dim}"
        assert n_heads % n_kv_heads == 0, \
            f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # GQA repetition factor
        self.dropout = dropout
        self.use_flash = use_flash_attention and FLASH_ATTN
        self.layer_idx = layer_idx
        self.scale = head_dim ** -0.5
        
        # Pre-norm
        self.norm = build_norm(norm_type, d_model, dgn_n_groups, norm_eps)
        
        # Q/K/V projections
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
        )
        
        # KV cache for inference
        self._kv_cache_k: Optional[torch.Tensor] = None
        self._kv_cache_v: Optional[torch.Tensor] = None
        self._kv_cache_len: int = 0
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        # Output projection: scale by 1/sqrt(2 * n_layers) for residual depth
        nn.init.normal_(self.o_proj.weight, std=0.02)
    
    def clear_kv_cache(self):
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_len = 0
    
    def forward(
        self,
        x: torch.Tensor,                           # (B, T, d_model)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) or (B, 1, T, T)
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        
        B, T, _ = x.shape
        residual = x
        x_norm = self.norm(x)
        
        # Project
        Q = self.q_proj(x_norm)   # (B, T, n_heads * head_dim)
        K = self.k_proj(x_norm)   # (B, T, n_kv_heads * head_dim)
        V = self.v_proj(x_norm)   # (B, T, n_kv_heads * head_dim)
        
        # Reshape to (B, H, T, D)
        Q = rearrange(Q, 'b t (h d) -> b h t d', d=self.head_dim)
        K = rearrange(K, 'b t (h d) -> b h t d', d=self.head_dim)
        V = rearrange(V, 'b t (h d) -> b h t d', d=self.head_dim)
        
        # RoPE
        Q, K = self.rope.apply_rotary(Q, K, position_ids)
        
        # KV cache concat (inference)
        if use_cache:
            if self._kv_cache_k is not None:
                K = torch.cat([self._kv_cache_k, K], dim=2)
                V = torch.cat([self._kv_cache_v, V], dim=2)
            self._kv_cache_k = K
            self._kv_cache_v = V
            self._kv_cache_len = K.shape[2]
        
        # GQA: expand K,V from n_kv_heads to n_heads
        if self.n_rep > 1:
            K = K.repeat_interleave(self.n_rep, dim=1)
            V = V.repeat_interleave(self.n_rep, dim=1)
        
        # Attention
        if self.use_flash:
            # FlashAttention: expects (B, T, H, D)
            Q_fa = rearrange(Q, 'b h t d -> b t h d').contiguous()
            K_fa = rearrange(K, 'b h t d -> b t h d').contiguous()
            V_fa = rearrange(V, 'b h t d -> b t h d').contiguous()
            
            out = flash_attn_func(
                Q_fa, K_fa, V_fa,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                softmax_scale=self.scale,
            )  # (B, T, H, D)
            out = rearrange(out, 'b t h d -> b t (h d)')
        else:
            # Standard PyTorch SDPA (fused, memory efficient)
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(attention_mask is None),
                scale=self.scale,
            )  # (B, H, T, D)
            out = rearrange(out, 'b h t d -> b t (h d)')
        
        out = self.o_proj(out)  # (B, T, d_model)
        
        return residual + out, (K, V) if use_cache else None