"""
HSE (Hierarchical Sparse Expert) FFN Layer.

Two-tier MoE:
  Tier 1: Select 1 semantic domain from K1=4 domains
  Tier 2: Select top-2 specialists from K2=8 experts within that domain
  Total: 32 experts, 2 active per token

Load balanced at both tiers independently for stable training.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dgn_norm import build_norm
from .triton_kernels import fused_swiglu


class FeedForwardExpert(nn.Module):
    """Single SwiGLU expert in the HSE layer."""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(fused_swiglu(self.gate(x), self.up(x)))


class DenseSwiGLUFFN(nn.Module):
    """Standard Dense SwiGLU FFN (used in alternating layers)."""
    
    def __init__(
        self,
        d_model: int = 1280,
        d_ff: int = 3200,
        norm_type: str = "dgn",
        dgn_n_groups: int = 16,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = build_norm(norm_type, d_model, dgn_n_groups, norm_eps)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight,   std=0.02)
        nn.init.normal_(self.down.weight, std=0.02 / math.sqrt(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        return residual + self.down(fused_swiglu(self.gate(h), self.up(h)))


class HSELayer(nn.Module):
    """
    Hierarchical Sparse Expert FFN Layer.
    
    Architecture:
      pre-norm → Tier1-router → select domain d*
             → Tier2-router[d*] → select top-2 experts
             → weighted sum of 2 expert outputs
             → residual add
    """
    
    def __init__(
        self,
        d_model: int = 1280,
        n_domains: int = 4,
        n_experts_per_domain: int = 8,
        top_k: int = 2,
        d_ff_expert: int = 288,
        capacity_factor: float = 1.25,
        expert_dropout: float = 0.1,
        aux_loss_alpha: float = 0.01,
        gumbel_tau_init: float = 1.0,
        norm_type: str = "dgn",
        dgn_n_groups: int = 16,
        norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()
        assert top_k <= n_experts_per_domain
        
        self.d_model = d_model
        self.n_domains = n_domains
        self.n_experts_per_domain = n_experts_per_domain
        self.n_experts_total = n_domains * n_experts_per_domain
        self.top_k = top_k
        self.d_ff_expert = d_ff_expert
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout
        self.aux_loss_alpha = aux_loss_alpha
        self.layer_idx = layer_idx
        
        # Pre-norm
        self.norm = build_norm(norm_type, d_model, dgn_n_groups, norm_eps)
        
        # ── Tier-1 Router ────────────────────────────────────────────────────
        # Selects semantic domain: d* = argmax(W_r1 · x / τ)
        self.tier1_router = nn.Linear(d_model, n_domains, bias=False)
        self.gumbel_tau = nn.Parameter(
            torch.tensor(gumbel_tau_init), requires_grad=False  # annealed externally
        )
        
        # ── Tier-2 Routers (one per domain) ──────────────────────────────────
        self.tier2_routers = nn.ModuleList([
            nn.Linear(d_model, n_experts_per_domain, bias=False)
            for _ in range(n_domains)
        ])
        
        # ── Experts ───────────────────────────────────────────────────────────
        self.experts = nn.ModuleList([
            FeedForwardExpert(d_model, d_ff_expert)
            for _ in range(self.n_experts_total)
        ])
        
        # ── Dropout for expert regularization ────────────────────────────────
        self.expert_dropout_layer = nn.Dropout(expert_dropout)
        
        # ── Aux loss accumulator (for logging) ────────────────────────────────
        self._last_aux_loss: Optional[torch.Tensor] = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights to near-uniform distribution."""
        nn.init.normal_(self.tier1_router.weight, std=0.02)
        for router in self.tier2_routers:
            nn.init.normal_(router.weight, std=0.02)
        for expert in self.experts:
            nn.init.normal_(expert.gate.weight, std=0.02)
            nn.init.normal_(expert.up.weight, std=0.02)
            nn.init.normal_(expert.down.weight, std=0.02 / math.sqrt(2))
    
    def update_gumbel_tau(self, progress: float):
        """
        Anneal Gumbel-softmax temperature from init to final.
        progress: float in [0, 1] representing training progress.
        """
        tau_init = 1.0
        tau_final = 0.1
        tau = tau_init * (tau_final / tau_init) ** progress
        self.gumbel_tau.data.fill_(tau)
    
    def _compute_aux_loss(
        self,
        tier1_probs: torch.Tensor,  # (N, n_domains) 
        domain_idx: torch.Tensor,   # (N,)
        x_flat: torch.Tensor,       # (N, d_model)
    ) -> torch.Tensor:
        """
        Hierarchical load-balance auxiliary loss.
        
        Loss = α * [Tier1_balance + Tier2_balance]
        
        Tier1_balance = n_domains * Σ_d [f_d * P_d]
        Tier2_balance = Σ_d {n_exp * Σ_e [f_{d,e} * P_{d,e}]} (over assigned tokens)
        """
        N = x_flat.shape[0]
        
        # Tier-1 balance
        # f_d: fraction of tokens routed to domain d
        one_hot_d = F.one_hot(domain_idx, self.n_domains).float()  # (N, n_domains)
        f1 = one_hot_d.mean(0)                                      # (n_domains,)
        P1 = tier1_probs.mean(0)                                    # (n_domains,)
        loss_t1 = self.n_domains * (f1 * P1).sum()
        
        # Tier-2 balance (over tokens in each domain)
        loss_t2 = torch.zeros(1, device=x_flat.device)
        
        for d in range(self.n_domains):
            mask = (domain_idx == d)
            n_d = mask.sum()
            if n_d < 2:
                continue
            
            x_d = x_flat[mask]                                      # (n_d, d_model)
            t2_logits = self.tier2_routers[d](x_d)                 # (n_d, n_exp)
            t2_probs = F.softmax(t2_logits, dim=-1)                 # (n_d, n_exp)
            
            # Top-k indices for f computation
            topk_idx = t2_logits.topk(self.top_k, dim=-1).indices  # (n_d, top_k)
            one_hot_e = F.one_hot(topk_idx, self.n_experts_per_domain).float()  # (n_d, k, n_exp)
            f2 = one_hot_e.sum(1).mean(0) / self.top_k             # (n_exp,)
            P2 = t2_probs.mean(0)                                   # (n_exp,)
            
            loss_t2 = loss_t2 + self.n_experts_per_domain * (f2 * P2).sum()
        
        return self.aux_loss_alpha * (loss_t1 + loss_t2)
    
    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with hierarchical routing.
        Returns (output, aux_loss) where aux_loss is for load balancing.
        """
        B, T, D = x.shape
        N = B * T
        
        residual = x
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(N, D)  # (N, D)
        
        # ── Tier-1 Routing ────────────────────────────────────────────────────
        tier1_logits = self.tier1_router(x_flat)              # (N, n_domains)
        tier1_probs = F.softmax(tier1_logits / self.gumbel_tau, dim=-1)
        
        if self.training:
            # Gumbel-softmax for differentiable discrete routing
            domain_idx = F.gumbel_softmax(
                tier1_logits, 
                tau=float(self.gumbel_tau), 
                hard=True
            ).argmax(dim=-1)                                   # (N,)
        else:
            domain_idx = tier1_probs.argmax(dim=-1)            # (N,)
        
        # ── Tier-2 Routing and Expert Computation ────────────────────────────
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        
        for d in range(self.n_domains):
            # Tokens assigned to domain d
            d_mask = (domain_idx == d)
            n_d = d_mask.sum()
            if n_d == 0:
                continue
            
            x_d = x_flat[d_mask]                               # (n_d, D)
            
            # Tier-2 routing within domain d
            t2_logits = self.tier2_routers[d](x_d)             # (n_d, n_exp)
            t2_probs = F.softmax(t2_logits, dim=-1)
            
            # Capacity enforcement: each expert handles at most cap tokens
            cap = max(1, int(self.capacity_factor * n_d / self.n_experts_per_domain))
            
            # Top-k expert selection
            topk_weights, topk_indices = t2_probs.topk(self.top_k, dim=-1)  # (n_d, k)
            topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-9)
            
            expert_out = torch.zeros_like(x_d)  # (n_d, D)
            
            for k in range(self.top_k):
                expert_ids = topk_indices[:, k]  # (n_d,)
                weights = topk_weights[:, k:k+1] # (n_d, 1)
                
                # Route to each expert in this domain
                for e in range(self.n_experts_per_domain):
                    global_eid = d * self.n_experts_per_domain + e
                    e_mask = (expert_ids == e)
                    n_e = e_mask.sum()
                    
                    if n_e == 0:
                        continue
                    
                    # Capacity check
                    if n_e > cap:
                        # Drop overflow tokens (set their contribution to 0)
                        # In practice use a priority score; here use first-come
                        e_mask_indices = e_mask.nonzero(as_tuple=True)[0][:cap]
                        e_mask_cap = torch.zeros_like(e_mask)
                        e_mask_cap[e_mask_indices] = True
                        e_mask = e_mask_cap
                        n_e = cap
                    
                    x_e = x_d[e_mask]                           # (n_e, D)
                    
                    # Apply expert dropout during training
                    if self.training and self.expert_dropout > 0:
                        x_e = self.expert_dropout_layer(x_e)
                    
                    # Expert forward pass
                    y_e = self.experts[global_eid](x_e)         # (n_e, D)
                    
                    # Weight and accumulate
                    expert_out[e_mask] += weights[e_mask] * y_e
            
            output[d_mask] = expert_out
        
        output = output.reshape(B, T, D)
        
        # ── Auxiliary loss ────────────────────────────────────────────────────
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(tier1_probs, domain_idx, x_flat)
            self._last_aux_loss = aux_loss.detach()
        
        return residual + output, aux_loss