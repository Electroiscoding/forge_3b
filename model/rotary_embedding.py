"""
RoPE (Rotary Positional Embedding) with YaRN support.
GPU-optimized via cached cos/sin buffers and optional Triton kernel.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) with:
    - Long-context theta (500,000 base)
    - YaRN scaling for context extension
    - GPU-resident cos/sin cache (no recomputation per step)
    - BF16-native computation
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 500_000.0,
        scaling_type: Optional[str] = None,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        
        # Compute inverse frequencies
        inv_freq = self._build_inv_freq(head_dim, base, scaling_type, scaling_factor)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute and cache cos/sin for all positions
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_seq_len: int = 0
        
        # Build initial cache
        self._build_cache(max_seq_len, device)
    
    def _build_inv_freq(
        self,
        head_dim: int,
        base: float,
        scaling_type: Optional[str],
        scaling_factor: float,
    ) -> torch.Tensor:
        """Compute inverse frequencies with optional YaRN scaling."""
        dim = head_dim
        # Standard RoPE frequencies
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        if scaling_type == "yarn":
            # YaRN: scale only the low-frequency dimensions
            # High-frequency dims (short-range) are left unchanged
            # Low-frequency dims (long-range) are scaled down by 1/factor
            mscale = 0.1 * math.log(scaling_factor) + 1.0  # YaRN mscale
            
            # Threshold: dims where wavelength > original_max_len
            low_freq_factor = 1.0
            high_freq_factor = 4.0
            original_max_len = self.max_seq_len
            new_base = base * (
                (scaling_factor * original_max_len / (2 * math.pi)) -
                (original_max_len / (2 * math.pi))
            ) ** (dim / (dim - 2))
            
            # Apply NTK-by-parts
            d = torch.arange(0, dim, 2).float()
            wavelength = 2 * math.pi / freqs
            
            # Smooth interpolation between no-scale and full-scale
            scale = torch.where(
                wavelength < original_max_len / high_freq_factor,
                torch.ones_like(freqs),
                torch.where(
                    wavelength > original_max_len / low_freq_factor,
                    torch.full_like(freqs, 1.0 / scaling_factor),
                    (original_max_len / wavelength / low_freq_factor - 1.0) / 
                    (high_freq_factor / low_freq_factor - 1.0) *
                    (1.0 - 1.0 / scaling_factor) + 1.0 / scaling_factor
                )
            )
            freqs = freqs * scale
        
        elif scaling_type == "linear":
            freqs = freqs / scaling_factor
        
        return freqs
    
    @torch.no_grad()
    def _build_cache(self, seq_len: int, device=None):
        """Build/extend the cos/sin cache up to seq_len."""
        if seq_len <= self._cache_seq_len:
            return
        
        device = device or (self.inv_freq.device if hasattr(self, 'inv_freq') 
                            else torch.device('cpu'))
        
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        
        # Store in BF16 for fast loading
        self._cos_cache = emb.cos().to(torch.bfloat16)
        self._sin_cache = emb.sin().to(torch.bfloat16)
        self._cache_seq_len = seq_len
    
    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached cos/sin for positions 0..seq_len-1."""
        if seq_len > self._cache_seq_len:
            self._build_cache(seq_len * 2, device)  # extend with buffer
        
        cos = self._cos_cache[:seq_len].to(device)
        sin = self._sin_cache[:seq_len].to(device)
        return cos, sin  # each (seq_len, head_dim)
    
    def apply_rotary(
        self,
        q: torch.Tensor,   # (B, n_heads, T, head_dim)
        k: torch.Tensor,   # (B, n_kv_heads, T, head_dim)
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K."""
        T = q.shape[2]
        device = q.device
        
        cos, sin = self.get_cos_sin(T, device)  # (T, head_dim)
        
        if position_ids is not None:
            # Custom positions (for packed sequences or position bias)
            cos = cos[position_ids]   # (B, T, head_dim) or (T, head_dim)
            sin = sin[position_ids]
        
        # Reshape for broadcasting: (1, 1, T, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_out = _rotate_half_apply(q.float(), cos, sin).to(q.dtype)
        k_out = _rotate_half_apply(k.float(), cos, sin).to(k.dtype)
        
        return q_out, k_out
    
    def forward(self, q, k, position_ids=None):
        return self.apply_rotary(q, k, position_ids)


def _rotate_half_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary: x * cos + rotate_half(x) * sin"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin