#!/usr/bin/env python
"""
Smoke test: torch.compile x DeepSpeed ZeRO interaction (issue #8).

Verifies that forward + backward + optimizer step complete with finite loss
for each combination of compile strategy and ZeRO stage, using a tiny
ForgeModel so a full matrix runs in minutes on a single GPU.

Compile strategies
------------------
  eager               no torch.compile (baseline — must PASS)
  layerwise           compile each model.layers[i] BEFORE deepspeed.initialize()
                      (the strategy used by training/pretrain_engine.py — must PASS)
  wholemodel-postinit deepspeed.initialize() first, then torch.compile the
                      engine's inner module (reproduces the NoneType-view crash
                      from issue #8 — expected to FAIL; recorded as REPRO)

Usage
-----
  # Full matrix on a single GPU (each cell runs in its own subprocess):
  python scripts/smoke_test_zero_compile.py --matrix

  # One cell directly:
  python scripts/smoke_test_zero_compile.py --mode layerwise --stage 2

  # Multi-GPU cell via the DeepSpeed launcher:
  deepspeed --num_gpus=2 scripts/smoke_test_zero_compile.py --mode layerwise --stage 2

Exit code is 0 when every required cell passes (eager + layerwise for each
requested stage). The wholemodel-postinit cell is informational: a crash there
confirms the issue still reproduces upstream, a pass means the installed
torch/DeepSpeed combination has fixed it natively.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODES = ["eager", "layerwise", "wholemodel-postinit"]
REQUIRED_MODES = {"eager", "layerwise"}

# Signature of the issue #8 crash inside DeepSpeed's gradient hook.
ISSUE8_SIGNATURE = "'NoneType' object has no attribute 'view'"


def tiny_model_config():
    from config import ForgeModelConfig

    return ForgeModelConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        max_seq_len=256,
        mha_layer_indices=[3],
        dense_ffn_layer_indices=[1, 3],
        hse_ffn_layer_indices=[0, 2],
        arg_d_inner=128,
        arg_d_state=16,
        arg_d_rank=8,
        arg_conv_kernel=4,
        arg_local_window=32,
        arg_local_n_heads=4,
        arg_local_n_kv_heads=2,
        arg_head_dim=32,
        mha_n_heads=4,
        mha_n_kv_heads=2,
        mha_head_dim=32,
        dense_d_ff=256,
        hse_n_domains=2,
        hse_n_experts_per_domain=2,
        hse_top_k=1,
        hse_d_ff_expert=64,
        dgn_n_groups=4,
        # Keep the test focused on the compile/ZeRO interaction: no external
        # kernels, no checkpointing noise.
        use_flash_attention=False,
        use_triton_kernels=False,
        use_gradient_checkpointing=False,
        use_torch_compile=False,
    )


def ds_config(stage: int, precision: str, micro_batch: int):
    cfg = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 1000,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_bucket_size": 1 << 20,
            "allgather_bucket_size": 1 << 20,
        },
    }
    if precision == "bf16":
        cfg["bf16"] = {"enabled": True}
    elif precision == "fp16":
        cfg["fp16"] = {"enabled": True, "initial_scale_power": 8}
    return cfg


def ensure_single_process_env():
    """Allow running under plain `python` (no deepspeed launcher)."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29777")


def run_cell(mode: str, stage: int, precision: str, steps: int,
             micro_batch: int, seq_len: int, compile_mode: str) -> None:
    """Run one (mode, stage) combination in-process. Raises on failure."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (DeepSpeed ZeRO + inductor test)")

    import deepspeed
    from model.forge_model import ForgeModel
    from training.gpu_optimizer import compile_model

    ensure_single_process_env()
    torch.manual_seed(1234)

    cfg = tiny_model_config()
    model = ForgeModel(cfg)

    if mode == "layerwise":
        # Mirrors training/pretrain_engine.py: compile inner blocks BEFORE
        # DeepSpeed attaches its post-accumulate-grad hooks.
        for i in range(len(model.layers)):
            model.layers[i] = compile_model(
                model.layers[i], mode=compile_mode, fullgraph=False, dynamic=True,
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config(stage, precision, micro_batch),
    )

    if mode == "wholemodel-postinit":
        # The original (broken) integration from issue #8: compile the inner
        # module after DeepSpeed has already registered its ZeRO hooks.
        object.__setattr__(
            engine,
            "module",
            torch.compile(engine.module, mode=compile_mode,
                          fullgraph=False, dynamic=True, backend="inductor"),
        )

    device = engine.device
    vocab = cfg.vocab_size

    for step in range(steps):
        input_ids = torch.randint(1, vocab, (micro_batch, seq_len), device=device)
        out = engine(input_ids=input_ids, labels=input_ids)
        loss = out["loss"]
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        engine.backward(loss)
        engine.step()
        print(f"[{mode}/zero{stage}] step {step}: loss={loss.item():.4f}", flush=True)

    print(f"[{mode}/zero{stage}] OK — {steps} steps completed", flush=True)


def run_matrix(args) -> int:
    """Run each cell in a fresh subprocess (isolates dynamo/dist state and
    lets an expected crash in one cell not kill the rest)."""
    results = {}
    for stage in args.stages:
        for mode in MODES:
            label = f"{mode}/zero{stage}"
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--mode", mode,
                "--stage", str(stage),
                "--precision", args.precision,
                "--steps", str(args.steps),
                "--micro-batch", str(args.micro_batch),
                "--seq-len", str(args.seq_len),
                "--compile-mode", args.compile_mode,
            ]
            print(f"\n===== {label} =====", flush=True)
            proc = subprocess.run(
                cmd, cwd=str(REPO_ROOT),
                capture_output=True, text=True, timeout=args.cell_timeout,
            )
            output = proc.stdout + proc.stderr
            sys.stdout.write(output[-4000:])
            if proc.returncode == 0:
                results[label] = "PASS"
            elif mode == "wholemodel-postinit" and ISSUE8_SIGNATURE in output:
                results[label] = "REPRO (expected: issue #8 crash)"
            else:
                results[label] = "FAIL"

    print("\n" + "=" * 62)
    print(f"{'cell':36s} result")
    print("-" * 62)
    required_ok = True
    for label, result in results.items():
        print(f"{label:36s} {result}")
        mode = label.split("/")[0]
        if mode in REQUIRED_MODES and result != "PASS":
            required_ok = False
    print("=" * 62)
    print("required cells (eager, layerwise):", "ALL PASS" if required_ok else "FAILURES")
    return 0 if required_ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", action="store_true",
                   help="run the full mode x stage matrix in subprocesses")
    p.add_argument("--mode", choices=MODES, default="layerwise")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--stages", type=int, nargs="+", choices=[1, 2, 3],
                   default=[2, 3], help="stages to cover in --matrix mode")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--compile-mode", default="default",
                   help='torch.compile mode; use "max-autotune" to match production')
    p.add_argument("--cell-timeout", type=int, default=1800,
                   help="per-cell timeout in seconds (matrix mode)")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="(injected by the deepspeed launcher)")
    args = p.parse_args()

    if args.matrix:
        sys.exit(run_matrix(args))

    run_cell(args.mode, args.stage, args.precision, args.steps,
             args.micro_batch, args.seq_len, args.compile_mode)


if __name__ == "__main__":
    main()
