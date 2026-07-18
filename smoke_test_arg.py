#!/usr/bin/env python3
"""
Smoke test: single forward + backward pass through one ARGLayer.
Verifies selective_scan_fn runs correctly in bf16 model.
"""
import torch
import torch.nn as nn

# Force FORGE_NO_TRITON so we don't wait on Triton JIT
import os; os.environ["FORGE_NO_TRITON"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from model.arg_layer import ARGLayer, MAMBA_SSM_AVAILABLE
print(f"mamba_ssm available: {MAMBA_SSM_AVAILABLE}")
assert MAMBA_SSM_AVAILABLE, "mamba_ssm not installed — run: pip install mamba-ssm"

# Build a single ARGLayer at bf16 (exactly as run_pretrain.py does it)
layer = ARGLayer(
    d_model=256, d_inner=256, d_state=64, d_rank=32,
    conv_kernel=4, local_window=32,
    local_n_heads=4, local_n_kv_heads=2, head_dim=64,
).to(device).to(torch.bfloat16)

# Tiny batch
B, T = 2, 128
x = torch.randn(B, T, 256, device=device, dtype=torch.bfloat16)

print("Running forward pass...")
out = layer(x)
print(f"  Output shape : {out.shape}  (expected: {B, T, 256})")
print(f"  Output dtype : {out.dtype}")
assert out.shape == (B, T, 256), f"Bad shape: {out.shape}"
assert not torch.isnan(out).any(), "NaNs in output!"
assert not torch.isinf(out).any(), "Infs in output!"

print("Running backward pass...")
loss = out.float().mean()
loss.backward()
print(f"  Loss         : {loss.item():.6f}")
print(f"  nu.grad      : {layer.seq_mixer.nu.grad if hasattr(layer, 'seq_mixer') else layer.nu.grad}")

# Verify all params got gradients
no_grad = [name for name, p in layer.named_parameters() if p.requires_grad and p.grad is None]
nan_grad = [name for name, p in layer.named_parameters() if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any()]

if no_grad:
    print(f"  ⚠ Params with no grad: {no_grad}")
if nan_grad:
    print(f"  ✗ Params with NaN grad: {nan_grad}")
    raise RuntimeError("NaN gradients detected")

if no_grad:
    raise AssertionError(f"These params have no gradient (unused in forward): {no_grad}")

print()
print("✓ SMOKE TEST PASSED — mamba_ssm selective_scan_fn works correctly in bf16 model")
