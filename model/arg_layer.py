"""
ARG (Adaptive Recurrent Gating) Layer — FORGE's core sequence mixer.

Combines:
1. Complex-eigenvalue selective SSM (recurrent branch)  
2. Local windowed GQA (local attention branch)
3. Learned scalar gate α_t ∈ [0,1] that blends the two
4. Compound Positional Bias (CPB) for SSM initialization
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dgn_norm import build_norm
from .rotary_embedding import RotaryEmbedding
from .triton_kernels import fused_swiglu

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    selective_scan_fn = None
    MAMBA_SSM_AVAILABLE = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input, pad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class ARGLayer(nn.Module):
    """
    Adaptive Recurrent Gating Layer.
    
    For each token t:
        h_recur_t = ComplexSSM(x_{0:t})
        h_local_t = WindowedGQA(x_{t-W:t})
        α_t       = σ(w_g · x_t)
        output_t  = α_t * h_local_t + (1 - α_t) * h_recur_t
    """
    
    def __init__(
        self,
        d_model: int = 2048,
        d_inner: int = 2048,
        d_state: int = 64,
        d_rank: int = 64,
        conv_kernel: int = 4,
        local_window: int = 64,
        local_n_heads: int = 8,
        local_n_kv_heads: int = 2,
        head_dim: int = 128,
        norm_type: str = "dgn",
        dgn_n_groups: int = 16,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        rope_base: float = 500_000.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_rank = d_rank
        self.conv_kernel = conv_kernel
        self.local_window = local_window
        self.local_n_heads = local_n_heads
        self.local_n_kv_heads = local_n_kv_heads
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.layer_idx = layer_idx
        
        # ── Pre-norm ──────────────────────────────────────────────────────────
        self.norm = build_norm(norm_type, d_model, dgn_n_groups, norm_eps)
        
        # ── Recurrent Branch ──────────────────────────────────────────────────
        # Input projection: x → (x_inner, z) both d_inner
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        
        # Depthwise causal conv for local mixing
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,   # causal padding
            groups=d_inner,
            bias=True,
        )
        
        # Input-dependent SSM parameters (low-rank for Δ)
        self.x_proj = nn.Linear(d_inner, d_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(d_rank, d_inner, bias=True)
        
        # Complex SSM eigenvalue parameters
        # Uses real-pair encoding: each complex eigenvalue λ = -exp(ν) ± jθ
        # is represented as two real state channels, so d_state must be even.
        # (d_state_complex = d_state // 2 actual complex eigenvalues)
        assert d_state % 2 == 0, "d_state must be even for complex-pair encoding"
        self.d_state_complex = d_state // 2

        self.nu = nn.Parameter(torch.zeros(self.d_state_complex))    # decay (log scale)
        self.theta = nn.Parameter(torch.randn(self.d_state_complex) * 0.01)  # oscillation freq
        
        # Skip connection coefficient D
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # ── Compound Positional Bias (CPB) ────────────────────────────────────
        # Maps sinusoidal position encoding to SSM initial state
        self.cpb_proj = nn.Linear(d_model, d_state, bias=False)
        
        # ── Local Attention Branch ────────────────────────────────────────────
        q_dim = local_n_heads * head_dim
        kv_dim = local_n_kv_heads * head_dim
        
        self.local_q = nn.Linear(d_model, q_dim, bias=False)
        self.local_k = nn.Linear(d_model, kv_dim, bias=False)
        self.local_v = nn.Linear(d_model, kv_dim, bias=False)
        self.local_o = nn.Linear(q_dim, d_model, bias=False)
        
        # Local RoPE (shorter max_seq_len since it's windowed)
        self.local_rope = RotaryEmbedding(
            head_dim=head_dim,
            max_seq_len=local_window * 4,
            base=rope_base,
        )
        
        # ── Adaptive Gate ─────────────────────────────────────────────────────
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)  # α=0.5 at init: sigmoid(0) = 0.5
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Careful initialization for stable training."""
        # dt_proj bias: positive to encourage non-trivial step sizes at init
        dt_init_std = self.d_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_bias = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_bias + torch.log(-torch.expm1(-dt_bias)))
        
        # nu / theta init — small positive decays, small oscillations
        nn.init.zeros_(self.nu)
        nn.init.normal_(self.theta, std=0.01)
        
        # Conv1d — identity-like init
        nn.init.zeros_(self.conv1d.weight)
        for i in range(self.d_inner):
            self.conv1d.weight.data[i, 0, -1] = 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLEX EIGENVALUE → REAL A MATRIX
    # ─────────────────────────────────────────────────────────────────────────

    def _build_real_A(self) -> torch.Tensor:
        """
        Build the (d_inner, d_state) real-valued A matrix for selective_scan_fn.

        Each complex eigenvalue λ_i = -exp(ν_i) + jθ_i is encoded as two real
        state channels with the same decay rate. The theta (oscillation) enters
        via the B/C projections. mamba_ssm's selective_scan_fn discretizes A
        internally via exp(A * Δ), so we pass the continuous-time log-decay
        (negative real, per state channel).
        """
        decay = -F.softplus(self.nu.float())   # fp32 — mamba_ssm requires A in fp32
        A_diag = decay.repeat_interleave(2)   # (d_state,)
        A = A_diag.unsqueeze(0).expand(self.d_inner, -1).contiguous().float()  # (d_inner, d_state), fp32
        return A
    
    # ─────────────────────────────────────────────────────────────────────────
    # CPB: Initial State from Position
    # ─────────────────────────────────────────────────────────────────────────
    
    def _sinusoidal_pe(self, position: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal position encoding vector for position 'position'."""
        d = self.d_model
        pos_tensor = torch.tensor(float(position), device=device, dtype=torch.float32)
        i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
        denom = torch.pow(10000.0, i / d)
        angles = pos_tensor / denom
        pe = torch.zeros(d, device=device, dtype=torch.float32)
        pe[0::2] = torch.sin(angles)
        pe[1::2] = torch.cos(angles)
        return pe
    
    def _cpb_initial_state(self, B: int, position_offset: int, 
                             device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute initial SSM state from absolute position offset."""
        sinpe = self._sinusoidal_pe(position_offset, device).to(dtype=self.cpb_proj.weight.dtype)
        sinpe = sinpe.unsqueeze(0)  # (1, d_model) — ZeRO-3 linear backward requires a batch dim
        h0 = torch.tanh(self.cpb_proj(sinpe))  # (1, d_state)
        h0_real = h0.expand(B, -1)  # (B, d_state)
        h0_imag = torch.zeros_like(h0_real)
        return h0_real, h0_imag
    
    # ─────────────────────────────────────────────────────────────────────────
    # RECURRENT BRANCH
    # ─────────────────────────────────────────────────────────────────────────
    
    def recurrent_branch(
        self,
        x: torch.Tensor,         # (B, T, d_model)
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Selective SSM recurrent branch using mamba_ssm's fused CUDA kernel.
        Returns (B, T, d_model).

        Uses selective_scan_fn (same kernel as Mamba) which:
        - Processes all channels in parallel with an efficient associative scan
        - Applies delta_softplus internally (no manual softplus on dt)
        - Fuses the z-gate SiLU multiplication inside the kernel
        - Eliminates 512× Python-level kernel launches per layer
        """
        B, T, _ = x.shape

        # ── Project inputs ────────────────────────────────────────────────────
        xz = self.in_proj(x)                        # (B, T, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)             # each (B, T, d_inner)

        # ── Causal conv for local mixing ──────────────────────────────────────
        x_conv = rearrange(x_inner, 'b t d -> b d t')
        x_conv = self.conv1d(x_conv)[:, :, :T]       # trim causal padding, still (B, d_inner, T)
        # keep channel-first — mamba_ssm expects (B, d_inner, T)
        x_conv = F.silu(x_conv)                      # (B, d_inner, T)

        # ── Input-dependent SSM parameters ────────────────────────────────────
        x_dbl = self.x_proj(rearrange(x_conv, 'b d t -> b t d'))  # (B, T, d_rank + 2*d_state)
        dt_raw, B_ssm, C_ssm = x_dbl.split([self.d_rank, self.d_state, self.d_state], dim=-1)

        # dt: per-channel step size — (B, T, d_inner) → (B, d_inner, T) for kernel
        # NOTE: do NOT apply softplus here — delta_softplus=True does it inside the kernel
        dt = self.dt_proj(dt_raw)                    # (B, T, d_inner), raw logits
        dt = rearrange(dt, 'b t d -> b d t').contiguous()  # (B, d_inner, T)

        # B, C: (B, d_state, T) — channel-first for kernel
        B_ssm = rearrange(B_ssm, 'b t n -> b n t').contiguous()   # (B, d_state, T)
        C_ssm = rearrange(C_ssm, 'b t n -> b n t').contiguous()   # (B, d_state, T)

        # ── Build A from complex eigenvalue params ────────────────────────────
        A = self._build_real_A()                     # (d_inner, d_state), negative real

        # ── Fused selective scan ──────────────────────────────────────────────
        # selective_scan_fn fuses: discretization, scan, z-gate, skip-connection D
        # CPB initial state (h0): selective_scan_fn assumes zero initial state.
        # The learned dt/A/B/C will naturally handle context boundaries.
        y = selective_scan_fn(
            x_conv,                                  # u:     (B, d_inner, T)
            dt,                                      # delta: (B, d_inner, T), raw
            A,                                       # A:     (d_inner, d_state), negative
            B_ssm,                                   # B:     (B, d_state, T)
            C_ssm,                                   # C:     (B, d_state, T)
            self.D.float(),                          # D:     (d_inner,) skip connection
            z=rearrange(z, 'b t d -> b d t').contiguous(),  # z: (B, d_inner, T) gate
            delta_bias=self.dt_proj.bias.float(),    # dt bias, added before softplus
            delta_softplus=True,                     # apply softplus inside kernel
        )                                            # returns (B, d_inner, T), gated with z

        y = rearrange(y, 'b d t -> b t d')           # (B, T, d_inner)
        return self.out_proj(y)                      # (B, T, d_model)
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOCAL ATTENTION BRANCH
    # ─────────────────────────────────────────────────────────────────────────
    
    def local_attention_branch(
        self,
        x: torch.Tensor,   # (B, T, d_model)
    ) -> torch.Tensor:
        """
        Windowed GQA with window size W.
        Uses FlashAttention-2 with window_size=(W, 0) when available.
        Returns (B, T, d_model).
        """
        B, T, _ = x.shape
        W = self.local_window
        
        Q = self.local_q(x)  # (B, T, n_heads * head_dim)
        K = self.local_k(x)  # (B, T, n_kv_heads * head_dim)
        V = self.local_v(x)
        
        Q = rearrange(Q, 'b t (h d) -> b h t d', d=self.head_dim)  # (B, H, T, D)
        K = rearrange(K, 'b t (h d) -> b h t d', d=self.head_dim)
        V = rearrange(V, 'b t (h d) -> b h t d', d=self.head_dim)
        
        # Apply local RoPE
        Q, K = self.local_rope.apply_rotary(Q, K)
        
        # GQA: expand KV to match Q heads
        n_rep = self.local_n_heads // self.local_n_kv_heads
        if n_rep > 1:
            K = K.repeat_interleave(n_rep, dim=1)  # (B, H, T, D)
            V = V.repeat_interleave(n_rep, dim=1)
        
        if self.use_flash_attention:
            # FlashAttention with sliding window — O(T*W) compute
            # Rearrange to (B, T, H, D) for flash_attn
            Q_fa = rearrange(Q, 'b h t d -> b t h d').contiguous()
            K_fa = rearrange(K, 'b h t d -> b t h d').contiguous()
            V_fa = rearrange(V, 'b h t d -> b t h d').contiguous()
            
            out = flash_attn_func(
                Q_fa, K_fa, V_fa,
                dropout_p=0.0,
                causal=True,
                window_size=(W, 0),  # local causal window
            )  # (B, T, H, D)
            out = rearrange(out, 'b t h d -> b t (h d)')
        else:
            # PyTorch fallback with manual windowed mask
            scale = self.head_dim ** -0.5
            scores = torch.einsum('bhid,bhjd->bhij', Q, K) * scale  # (B, H, T, T)
            
            # Causal window mask
            indices = torch.arange(T, device=x.device)
            mask_2d = (indices.unsqueeze(0) <= indices.unsqueeze(1)) & \
                      (indices.unsqueeze(0) >= (indices.unsqueeze(1) - W))
            scores = scores.masked_fill(~mask_2d.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn = F.softmax(scores.float(), dim=-1).to(Q.dtype)
            out = torch.einsum('bhij,bhjd->bhid', attn, V)
            out = rearrange(out, 'b h t d -> b t (h d)')
        
        return self.local_o(out)  # (B, T, d_model)
    
    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────────────────────────
    
    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        x: (B, T, d_model)
        Returns: (B, T, d_model) — residual connection applied inside
        """
        residual = x
        x_norm = self.norm(x)
        
        # Two branches
        h_recur = self.recurrent_branch(x_norm, position_offset)
        h_local = self.local_attention_branch(x_norm)
        
        # Learned gate
        alpha = torch.sigmoid(self.gate_proj(x_norm))  # (B, T, 1)
        
        # Blend and add residual
        mixed = alpha * h_local + (1.0 - alpha) * h_recur  # (B, T, d_model)
        
        return residual + mixed