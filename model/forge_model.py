"""
FORGE-3B: Complete Model Architecture
Integrates ARG, MHA, Dense FFN, HSE FFN into the full autoregressive LM.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .arg_layer import ARGLayer
from .mha_layer import GlobalMHALayer
from .hse_layer import HSELayer, DenseSwiGLUFFN
from .dgn_norm import build_norm

logger = logging.getLogger(__name__)


class ForgeBlock(nn.Module):
    """
    One FORGE block: SeqMixer + FFN (both with pre-norm and residual).
    
    Blocks are composed sequentially. The seq_mixer is either ARG or MHA;
    the ffn is either DenseSwiGLUFFN or HSELayer — determined by layer_idx.
    """
    
    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        is_mha: bool,
        is_hse: bool,
        # ARG config
        arg_d_inner: int = 1280,
        arg_d_state: int = 48,
        arg_d_rank: int = 48,
        arg_conv_kernel: int = 4,
        arg_local_window: int = 64,
        arg_local_n_heads: int = 8,
        arg_local_n_kv_heads: int = 2,
        arg_head_dim: int = 80,
        # MHA config
        mha_n_heads: int = 16,
        mha_n_kv_heads: int = 4,
        mha_head_dim: int = 80,
        rope_base: float = 500_000.0,
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: float = 1.0,
        max_seq_len: int = 4096,
        # Dense FFN config
        dense_d_ff: int = 3200,
        # HSE config
        hse_n_domains: int = 4,
        hse_n_experts_per_domain: int = 8,
        hse_top_k: int = 2,
        hse_d_ff_expert: int = 288,
        hse_capacity_factor: float = 1.25,
        hse_expert_dropout: float = 0.1,
        hse_aux_loss_alpha: float = 0.01,
        # Norm
        norm_type: str = "dgn",
        dgn_n_groups: int = 16,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_mha = is_mha
        self.is_hse = is_hse
        
        # ── Sequence Mixer ────────────────────────────────────────────────────
        if is_mha:
            self.seq_mixer = GlobalMHALayer(
                d_model=d_model,
                n_heads=mha_n_heads,
                n_kv_heads=mha_n_kv_heads,
                head_dim=mha_head_dim,
                rope_base=rope_base,
                rope_scaling_type=rope_scaling_type,
                rope_scaling_factor=rope_scaling_factor,
                max_seq_len=max_seq_len,
                dropout=dropout,
                norm_type=norm_type,
                dgn_n_groups=dgn_n_groups,
                norm_eps=norm_eps,
                use_flash_attention=use_flash_attention,
                layer_idx=layer_idx,
            )
        else:
            self.seq_mixer = ARGLayer(
                d_model=d_model,
                d_inner=arg_d_inner,
                d_state=arg_d_state,
                d_rank=arg_d_rank,
                conv_kernel=arg_conv_kernel,
                local_window=arg_local_window,
                local_n_heads=arg_local_n_heads,
                local_n_kv_heads=arg_local_n_kv_heads,
                head_dim=arg_head_dim,
                norm_type=norm_type,
                dgn_n_groups=dgn_n_groups,
                norm_eps=norm_eps,
                use_flash_attention=use_flash_attention,
                rope_base=rope_base,
                layer_idx=layer_idx,
                max_seq_len=max_seq_len,
            )
        
        # ── FFN ───────────────────────────────────────────────────────────────
        if is_hse:
            self.ffn = HSELayer(
                d_model=d_model,
                n_domains=hse_n_domains,
                n_experts_per_domain=hse_n_experts_per_domain,
                top_k=hse_top_k,
                d_ff_expert=hse_d_ff_expert,
                capacity_factor=hse_capacity_factor,
                expert_dropout=hse_expert_dropout,
                aux_loss_alpha=hse_aux_loss_alpha,
                norm_type=norm_type,
                dgn_n_groups=dgn_n_groups,
                norm_eps=norm_eps,
                layer_idx=layer_idx,
            )
        else:
            self.ffn = DenseSwiGLUFFN(
                d_model=d_model,
                d_ff=dense_d_ff,
                norm_type=norm_type,
                dgn_n_groups=dgn_n_groups,
                norm_eps=norm_eps,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        use_cache: bool = False,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        aux_loss = None
        
        # ── Sequence Mixing ───────────────────────────────────────────────────
        if self.is_mha:
            x, _ = self.seq_mixer(
                x, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=use_cache
            )
        else:
            x = self.seq_mixer(x, position_offset=position_offset)
        
        # ── FFN ───────────────────────────────────────────────────────────────
        if self.is_hse:
            x, aux_loss = self.ffn(x, return_aux_loss=return_aux_loss)
        else:
            x = self.ffn(x)
        
        return x, aux_loss


class ForgeModel(nn.Module):
    """
    FORGE-3B: Full autoregressive language model.
    
    Architecture:
    - Token embedding (tied to LM head)
    - 36 ForgeBlocks (27 ARG + 9 MHA for seq mixing; 18 Dense + 18 HSE for FFN)
    - Final DGN normalization
    - LM head (weight-tied to embedding)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        d = config.d_model
        
        # ── Token Embedding ───────────────────────────────────────────────────
        self.embed_tokens = nn.Embedding(config.vocab_size, d, padding_idx=0)
        
        # ── FORGE Blocks ──────────────────────────────────────────────────────
        mha_set = set(config.mha_layer_indices)
        hse_set = set(config.hse_ffn_layer_indices)
        
        self.layers = nn.ModuleList([
            ForgeBlock(
                d_model=d,
                layer_idx=i,
                is_mha=(i in mha_set),
                is_hse=(i in hse_set),
                arg_d_inner=config.arg_d_inner,
                arg_d_state=config.arg_d_state,
                arg_d_rank=config.arg_d_rank,
                arg_conv_kernel=config.arg_conv_kernel,
                arg_local_window=config.arg_local_window,
                arg_local_n_heads=config.arg_local_n_heads,
                arg_local_n_kv_heads=config.arg_local_n_kv_heads,
                arg_head_dim=config.arg_head_dim,
                mha_n_heads=config.mha_n_heads,
                mha_n_kv_heads=config.mha_n_kv_heads,
                mha_head_dim=config.mha_head_dim,
                rope_base=config.rope_theta,
                rope_scaling_type=config.rope_scaling_type,
                rope_scaling_factor=config.rope_scaling_factor,
                max_seq_len=config.max_seq_len,
                dense_d_ff=config.dense_d_ff,
                hse_n_domains=config.hse_n_domains,
                hse_n_experts_per_domain=config.hse_n_experts_per_domain,
                hse_top_k=config.hse_top_k,
                hse_d_ff_expert=config.hse_d_ff_expert,
                hse_capacity_factor=config.hse_capacity_factor,
                hse_expert_dropout=config.hse_expert_dropout,
                hse_aux_loss_alpha=config.hse_aux_loss_alpha,
                norm_type=config.norm_type,
                dgn_n_groups=config.dgn_n_groups,
                norm_eps=config.norm_eps,
                use_flash_attention=config.use_flash_attention,
            )
            for i in range(config.n_layers)
        ])
        
        # ── Final Norm ────────────────────────────────────────────────────────
        self.norm_final = build_norm(config.norm_type, d, config.dgn_n_groups, config.norm_eps)
        
        # ── LM Head ───────────────────────────────────────────────────────────
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        
        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # ── Gradient Checkpointing Configuration ─────────────────────────────
        self.gradient_checkpointing_enabled = config.use_gradient_checkpointing
        self.gradient_checkpointing_ratio = config.gradient_checkpointing_ratio
        # Checkpoint every N-th layer
        if config.gradient_checkpointing_ratio > 0:
            self._checkpoint_interval = max(1, int(1.0 / config.gradient_checkpointing_ratio))
        else:
            self._checkpoint_interval = 1
        
        # Apply weight initialization
        self._init_weights()
        
        # Log parameter count
        n_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in self.parameters())
        logger.info(f"ForgeModel: {n_params/1e9:.3f}B total parameters")
        emb_p = self.embed_tokens.weight
        emb_numel = emb_p.ds_numel if hasattr(emb_p, "ds_numel") else emb_p.numel()
        logger.info(f"  Embedding: {emb_numel/1e6:.1f}M")
        logger.info(f"  N layers: {config.n_layers} "
                    f"({len(mha_set)} MHA + {config.n_layers - len(mha_set)} ARG, "
                    f"{len(hse_set)} HSE + {config.n_layers - len(hse_set)} Dense)")
    
    def _init_weights(self):
        """Initialize all model weights with scaled normal distribution."""
        n_layers = self.config.n_layers
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        
        # Residual scaling: scale output projections by 1/sqrt(2*n_layers)
        # This prevents the residual stream from growing with depth
        if self.config.init_std_residual_scale:
            scale = 1.0 / math.sqrt(2 * n_layers)
            for layer in self.layers:
                if isinstance(layer.seq_mixer, ARGLayer):
                    nn.init.normal_(layer.seq_mixer.out_proj.weight, std=scale)
                    nn.init.normal_(layer.seq_mixer.local_o.weight, std=scale)
                elif isinstance(layer.seq_mixer, GlobalMHALayer):
                    nn.init.normal_(layer.seq_mixer.o_proj.weight, std=scale)
                
                if isinstance(layer.ffn, DenseSwiGLUFFN):
                    nn.init.normal_(layer.ffn.down.weight, std=scale)
                elif isinstance(layer.ffn, HSELayer):
                    for expert in layer.ffn.experts:
                        nn.init.normal_(expert.down.weight, std=scale)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing_enabled = True
        logger.info("Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing_enabled = False
    
    def forward(
        self,
        input_ids: torch.Tensor,                         # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,           # (B, T) for LM loss
        use_cache: bool = False,
        return_aux_loss: bool = True,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for FORGE-3B.
        
        Returns dict with:
            'loss': cross-entropy + aux_loss (if labels provided)
            'logits': (B, T, vocab_size)
            'aux_loss': sum of MoE load-balance losses
            'hidden_states': list of layer outputs (if output_hidden_states)
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # ── Token Embeddings ──────────────────────────────────────────────────
        h = self.embed_tokens(input_ids)  # (B, T, d_model)
        
        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        # ── Forward Through Layers ────────────────────────────────────────────
        all_hidden_states = [] if output_hidden_states else None
        total_aux_loss = torch.zeros(1, device=device, dtype=h.dtype)
        n_hse_layers = 0
        
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(h)
            
            # Selective gradient checkpointing
            if (self.gradient_checkpointing_enabled and 
                self.training and 
                layer_idx % self._checkpoint_interval == 0):
                
                def _forward_layer(h_, layer_=layer):
                    out, aux = layer_(
                        h_,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_offset=0,
                        use_cache=False,
                        return_aux_loss=True,
                    )
                    return out, aux if aux is not None else torch.zeros(1, device=h_.device)
                
                h, aux = checkpoint(
                    _forward_layer, h,
                    use_reentrant=True,   # standard stable checkpointing for MoE
                    preserve_rng_state=True,
                )
            else:
                h, aux = layer(
                    h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_offset=0,
                    use_cache=use_cache,
                    return_aux_loss=return_aux_loss,
                )
            
            if aux is not None and return_aux_loss:
                total_aux_loss = total_aux_loss + aux
                n_hse_layers += 1
        
        # ── Final Norm ────────────────────────────────────────────────────────
        h = self.norm_final(h)  # (B, T, d_model)
        
        # ── LM Head ───────────────────────────────────────────────────────────
        logits = self.lm_head(h).float()  # (B, T, vocab_size) — FP32 for numerical stability
        
        # ── Loss Computation ──────────────────────────────────────────────────
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Cross-entropy loss (ignore -100 and padding)
            ce_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            
            # Average aux loss over HSE layers
            avg_aux = total_aux_loss / max(1, n_hse_layers)
            loss = ce_loss + avg_aux
        
        output = {
            "loss": loss,
            "logits": logits,
            "aux_loss": total_aux_loss / max(1, n_hse_layers),
        }
        if output_hidden_states:
            output["hidden_states"] = all_hidden_states
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-p/top-k sampling.
        Clears KV caches after generation.
        """
        self.eval()
        B, T_prompt = input_ids.shape
        device = input_ids.device
        
        generated = input_ids.clone()
        
        # Clear KV caches in MHA layers
        for layer in self.layers:
            if isinstance(layer.seq_mixer, GlobalMHALayer):
                layer.seq_mixer.clear_kv_cache()
        
        for step in range(max_new_tokens):
            # Only process new tokens after first step
            if step == 0:
                curr_input = generated
            else:
                curr_input = generated[:, -1:]  # only last token
            
            outputs = self.forward(curr_input, use_cache=True, return_aux_loss=False)
            logits = outputs["logits"][:, -1, :]  # (B, vocab_size)
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tok_id in set(generated[b].tolist()):
                        if logits[b, tok_id] < 0:
                            logits[b, tok_id] *= repetition_penalty
                        else:
                            logits[b, tok_id] /= repetition_penalty
            
            # Temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                topk_vals = torch.topk(logits, min(top_k, logits.shape[-1])).values
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float('-inf'))
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens beyond cumulative probability top_p
                remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float('-inf')
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences hit EOS
            if (generated == eos_token_id).any(dim=1).all():
                break
        
        # Clear caches
        for layer in self.layers:
            if isinstance(layer.seq_mixer, GlobalMHALayer):
                layer.seq_mixer.clear_kv_cache()
        
        return generated
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["embedding"] = self.embed_tokens.weight.numel()
        counts["arg_layers"] = sum(
            p.numel() for i, l in enumerate(self.layers)
            if isinstance(l.seq_mixer, ARGLayer)
            for p in l.seq_mixer.parameters()
        )
        counts["mha_layers"] = sum(
            p.numel() for l in self.layers
            if isinstance(l.seq_mixer, GlobalMHALayer)
            for p in l.seq_mixer.parameters()
        )
        counts["dense_ffn"] = sum(
            p.numel() for l in self.layers
            if isinstance(l.ffn, DenseSwiGLUFFN)
            for p in l.ffn.parameters()
        )
        counts["hse_ffn_total"] = sum(
            p.numel() for l in self.layers
            if isinstance(l.ffn, HSELayer)
            for p in l.ffn.parameters()
        )
        hse_active = 0
        for l in self.layers:
            if isinstance(l.ffn, HSELayer):
                hse_active += l.ffn.top_k * 3 * l.ffn.d_model * l.ffn.d_ff_expert
        counts["hse_ffn_active"] = hse_active
        return counts


def build_forge_1b(config=None) -> ForgeModel:
    """Build FORGE-1B (<= 1.0B total params) from config (or default config)."""
    from config import ForgeModelConfig
    if config is None:
        config = ForgeModelConfig()
    model = ForgeModel(config)
    return model


def build_forge_3b(config=None) -> ForgeModel:
    """Alias for build_forge_1b (FORGE architecture downgraded to 1B max params)."""
    return build_forge_1b(config)