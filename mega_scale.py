def calculate_params(d_model, n_layers, n_experts, d_ff_expert, vocab_size=65536):
    mha_layer_indices = [i for i in range(3, n_layers, 4)]
    n_arg = n_layers - len(mha_layer_indices)
    n_mha = len(mha_layer_indices)
    n_dense = n_layers // 2
    n_hse = n_layers - n_dense

    arg_d_inner = d_model
    arg_d_state = 128
    arg_d_rank = 128
    arg_local_n_heads = d_model // 128
    arg_local_n_kv_heads = arg_local_n_heads // 4
    arg_head_dim = 128

    mha_n_heads = d_model // 128
    mha_n_kv_heads = mha_n_heads // 4
    mha_head_dim = 128

    embed = vocab_size * d_model

    arg_per = (
        d_model * 2 * arg_d_inner +
        arg_d_inner * 4 + # conv
        arg_d_inner * (arg_d_rank + 2 * arg_d_state) +
        arg_d_rank * arg_d_inner +
        arg_d_state * 2 +
        arg_d_inner +
        arg_d_inner * d_model +
        arg_local_n_heads * arg_head_dim * d_model +
        arg_local_n_kv_heads * arg_head_dim * d_model * 2 +
        arg_local_n_heads * arg_head_dim * d_model +
        d_model * arg_d_state
    )

    mha_per = (
        d_model * mha_n_heads * mha_head_dim +
        d_model * mha_n_kv_heads * mha_head_dim * 2 +
        d_model * mha_n_heads * mha_head_dim
    )

    dense_d_ff = int(d_model * 2.7)
    dense_per = 3 * d_model * dense_d_ff

    hse_per = n_experts * 3 * d_model * d_ff_expert

    total = embed + n_arg * arg_per + n_mha * mha_per + n_dense * dense_per + n_hse * hse_per

    hse_active_per = 2 * 3 * d_model * d_ff_expert
    active = embed + n_arg * arg_per + n_mha * mha_per + n_dense * dense_per + n_hse * hse_active_per

    return total, active

# Target ~500B
t500, a500 = calculate_params(8192, 80, 256, 1024)
print(f"500B Target: Total={t500/1e9:.1f}B, Active={a500/1e9:.1f}B, Sparsity={a500/t500*100:.1f}%")

# Target ~1T
t1t, a1t = calculate_params(12288, 96, 256, 2048)
print(f"1T Target: Total={t1t/1e9:.1f}B, Active={a1t/1e9:.1f}B, Sparsity={a1t/t1t*100:.1f}%")
