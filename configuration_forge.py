"""
FORGE Configuration Class
Compatible with Hugging Face PretrainedConfig and AutoConfig.
"""

from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List


class ForgeConfig(PretrainedConfig):
    model_type = "forge"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 206464,
        d_model: int = 1024,
        n_layers: int = 18,
        max_seq_len: int = 4096,
        mha_layer_indices: Optional[List[int]] = None,
        dense_ffn_layer_indices: Optional[List[int]] = None,
        hse_ffn_layer_indices: Optional[List[int]] = None,
        arg_d_inner: int = 1024,
        arg_d_state: int = 48,
        arg_d_rank: int = 48,
        arg_conv_kernel: int = 4,
        arg_local_window: int = 64,
        arg_local_n_heads: int = 8,
        arg_local_n_kv_heads: int = 2,
        arg_head_dim: int = 64,
        arg_scalar_gate: bool = True,
        mha_n_heads: int = 16,
        mha_n_kv_heads: int = 4,
        mha_head_dim: int = 64,
        rope_theta: float = 500000.0,
        rope_scaling_type: Optional[str] = None,
        rope_scaling_factor: float = 1.0,
        dense_d_ff: int = 2560,
        hse_n_domains: int = 4,
        hse_n_experts_per_domain: int = 6,
        hse_top_k: int = 2,
        hse_d_ff_expert: int = 210,
        hse_capacity_factor: float = 1.25,
        hse_expert_dropout: float = 0.1,
        hse_aux_loss_alpha: float = 0.01,
        hse_gumbel_tau_init: float = 1.0,
        hse_gumbel_tau_final: float = 0.1,
        hse_gumbel_tau_warmup_tokens: int = 10000000000,
        norm_type: str = "dgn",
        dgn_n_groups: int = 16,
        norm_eps: float = 1e-06,
        init_std: float = 0.02,
        init_std_residual_scale: bool = True,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_flash_attention: bool = True,
        use_triton_kernels: bool = True,
        use_torch_compile: bool = False,
        compile_mode: str = "max-autotune",
        use_gradient_checkpointing: bool = False,
        gradient_checkpointing_ratio: float = 1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.mha_layer_indices = mha_layer_indices if mha_layer_indices is not None else [5, 11, 17]
        self.dense_ffn_layer_indices = dense_ffn_layer_indices if dense_ffn_layer_indices is not None else [1, 3, 5, 7, 9, 11, 13, 15, 17]
        self.hse_ffn_layer_indices = hse_ffn_layer_indices if hse_ffn_layer_indices is not None else [0, 2, 4, 6, 8, 10, 12, 14, 16]
        self.arg_d_inner = arg_d_inner
        self.arg_d_state = arg_d_state
        self.arg_d_rank = arg_d_rank
        self.arg_conv_kernel = arg_conv_kernel
        self.arg_local_window = arg_local_window
        self.arg_local_n_heads = arg_local_n_heads
        self.arg_local_n_kv_heads = arg_local_n_kv_heads
        self.arg_head_dim = arg_head_dim
        self.arg_scalar_gate = arg_scalar_gate
        self.mha_n_heads = mha_n_heads
        self.mha_n_kv_heads = mha_n_kv_heads
        self.mha_head_dim = mha_head_dim
        self.rope_theta = rope_theta
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_factor = rope_scaling_factor
        self.dense_d_ff = dense_d_ff
        self.hse_n_domains = hse_n_domains
        self.hse_n_experts_per_domain = hse_n_experts_per_domain
        self.hse_top_k = hse_top_k
        self.hse_d_ff_expert = hse_d_ff_expert
        self.hse_capacity_factor = hse_capacity_factor
        self.hse_expert_dropout = hse_expert_dropout
        self.hse_aux_loss_alpha = hse_aux_loss_alpha
        self.hse_gumbel_tau_init = hse_gumbel_tau_init
        self.hse_gumbel_tau_final = hse_gumbel_tau_final
        self.hse_gumbel_tau_warmup_tokens = hse_gumbel_tau_warmup_tokens
        self.norm_type = norm_type
        self.dgn_n_groups = dgn_n_groups
        self.norm_eps = norm_eps
        self.init_std = init_std
        self.init_std_residual_scale = init_std_residual_scale
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.use_triton_kernels = use_triton_kernels
        self.use_torch_compile = use_torch_compile
        self.compile_mode = compile_mode
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
