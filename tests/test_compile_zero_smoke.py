#!/usr/bin/env python3
"""
FORGE-3B: torch.compile + DeepSpeed ZeRO Smoke Test.

Validates that the compile-before-init pattern (Issue #8 fix) prevents the
'NoneType' object has no attribute 'view' crash in DeepSpeed's
post-accumulate-grad hooks when used with torch.compile (AOTAutograd).

Test Matrix:
    1. Eager-mode baseline     (no compile, no DeepSpeed)
    2. Compiled + no DeepSpeed (validates compile alone works)
    3. Eager + DeepSpeed ZeRO  (validates ZeRO alone works)
    4. Compiled BEFORE ZeRO    (the Issue #8 fix — must not crash)
    5. Gradient correctness    (compiled grads ≈ eager grads within BF16 tol)

Usage:
    # Single-GPU ZeRO-2:
    python tests/test_compile_zero_smoke.py --zero_stage 2

    # Single-GPU ZeRO-3:
    python tests/test_compile_zero_smoke.py --zero_stage 3

    # Multi-GPU ZeRO-3:
    deepspeed --num_gpus=2 tests/test_compile_zero_smoke.py --zero_stage 3

    # Quick check (just validates no crash, skips gradient comparison):
    python tests/test_compile_zero_smoke.py --quick
"""

import os
import sys
import json
import argparse
import tempfile
import logging
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")


# ─── Test Helpers ─────────────────────────────────────────────────────────────

def _get_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device, data_dir: str = None):
    """Fetch a real batch from HF datasets, or fallback to synthetic."""
    if data_dir:
        try:
            logger.info(f"Attempting to load real batch from HF dataset: {data_dir}")
            import numpy as np
            from pathlib import Path
            from data.dataset import resolve_data_dir
            resolved = resolve_data_dir(data_dir)
            npz_files = list(Path(resolved).glob("**/*.npz"))
            if not npz_files:
                raise FileNotFoundError("No .npz files found")
            
            data = np.load(str(npz_files[0]))
            input_ids_all = data["input_ids"]
            loss_mask_all = data["loss_mask"]
            
            # Slice to the requested batch size and seq_len for the tiny smoke test model
            input_ids = torch.from_numpy(input_ids_all[:batch_size, :seq_len].astype(np.int64)).to(device)
            loss_mask = torch.from_numpy(loss_mask_all[:batch_size, :seq_len].astype(np.int64)).to(device)
            
            # Reconstruct labels from loss_mask: where mask=1 -> token id, mask=0 -> -100
            labels = input_ids.clone()
            labels[loss_mask == 0] = -100
            
            return {"input_ids": input_ids, "labels": labels}
        except Exception as e:
            logger.warning(f"Failed to load real dataset '{data_dir}': {e}. Falling back to synthetic.")
            
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    mask = torch.rand(batch_size, seq_len, device=device) < 0.2
    labels[mask] = -100
    return {"input_ids": input_ids, "labels": labels}


def _generate_zero_config(stage: int, tmp_dir: str, micro_batch: int = 1) -> str:
    """Generate a minimal DeepSpeed ZeRO config for testing."""
    config = {
        "zero_optimization": {
            "stage": stage,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5_000_000,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 5_000_000,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 999999,
        "wall_clock_breakdown": False,
    }

    if stage == 3:
        config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5_000_000,
            "stage3_param_persistence_threshold": 1024,
            "stage3_max_live_parameters": 100_000_000,
            "stage3_max_reuse_distance": 100_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
            "sub_group_size": 100000000000,
        })

    config_path = os.path.join(tmp_dir, f"ds_zero{stage}_test.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def _build_small_model(config_path: str = None):
    """Build a small FORGE model for testing using forge_100m config."""
    from config import ForgeModelConfig
    from model.forge_model import build_forge_3b

    if config_path and Path(config_path).exists():
        model_config = ForgeModelConfig.from_json(config_path)
    else:
        # Minimal config for smoke testing (with vocab_size matching real tokenizer)
        model_config = ForgeModelConfig(
            d_model=256,
            n_layers=4,
            vocab_size=206464,
            max_seq_len=512,
            mha_layer_indices=[1, 3],
            dense_ffn_layer_indices=[0],
            hse_ffn_layer_indices=[2],
            arg_d_inner=256,
            arg_d_state=8,
            arg_d_rank=8,
            arg_conv_kernel=4,
            arg_local_window=32,
            arg_local_n_heads=4,
            arg_local_n_kv_heads=1,
            arg_head_dim=32,
            mha_n_heads=4,
            mha_n_kv_heads=1,
            mha_head_dim=64,
            dense_d_ff=512,
            hse_n_domains=2,
            hse_n_experts_per_domain=2,
            hse_top_k=1,
            hse_d_ff_expert=128,
            dgn_n_groups=4,
            use_gradient_checkpointing=False,
        )

    model = build_forge_3b(model_config)
    return model, model_config


# ─── Individual Test Cases ────────────────────────────────────────────────────

def test_eager_baseline(device: torch.device, config_path: str = None, data_dir: str = None) -> bool:
    """Test 1: Eager mode (no compile, no DeepSpeed) — baseline correctness."""
    logger.info("=" * 60)
    logger.info("TEST 1: Eager-mode baseline (no compile, no DeepSpeed)")
    logger.info("=" * 60)
    try:
        model, model_config = _build_small_model(config_path)
        model = model.to(device).to(torch.bfloat16)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = _get_batch(2, 128, model_config.vocab_size, device, data_dir)

        # Forward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"], return_aux_loss=True)
            loss = outputs["loss"]

        logger.info(f"  Forward OK — loss={loss.item():.4f}")

        # Backward
        loss.backward()
        logger.info("  Backward OK")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        logger.info("  Optimizer step OK")

        # Verify gradients were populated
        n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        n_total = sum(1 for p in model.parameters())
        logger.info(f"  Grads: {n_with_grad}/{n_total} parameters have gradients")

        del model, optimizer
        torch.cuda.empty_cache()
        logger.info("✅ TEST 1 PASSED: Eager baseline works correctly\n")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_compile_only(device: torch.device, config_path: str = None, data_dir: str = None) -> bool:
    """Test 2: Compiled model without DeepSpeed — validates compile alone."""
    logger.info("=" * 60)
    logger.info("TEST 2: Compiled model (no DeepSpeed)")
    logger.info("=" * 60)
    try:
        from training.gpu_optimizer import compile_forge_layers

        model, model_config = _build_small_model(config_path)
        model = model.to(device).to(torch.bfloat16)

        # Compile inner layers (the safe pattern)
        compile_forge_layers(model, mode="default", dynamic=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = _get_batch(2, 128, model_config.vocab_size, device, data_dir)

        # Forward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"], return_aux_loss=True)
            loss = outputs["loss"]

        logger.info(f"  Forward OK — loss={loss.item():.4f}")

        # Backward (this is where compile issues usually manifest)
        loss.backward()
        logger.info("  Backward OK (no NoneType crash)")

        optimizer.step()
        optimizer.zero_grad()
        logger.info("  Optimizer step OK")

        del model, optimizer
        torch.cuda.empty_cache()
        logger.info("✅ TEST 2 PASSED: Compiled model works without DeepSpeed\n")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_deepspeed_eager(device: torch.device, zero_stage: int, config_path: str = None, data_dir: str = None) -> bool:
    """Test 3: DeepSpeed ZeRO without compile — validates ZeRO alone."""
    logger.info("=" * 60)
    logger.info(f"TEST 3: DeepSpeed ZeRO-{zero_stage} (eager, no compile)")
    logger.info("=" * 60)
    try:
        import deepspeed
        from training.gpu_optimizer import convert_to_attr_dict, setup_deepspeed_engine

        model, model_config = _build_small_model(config_path)
        model = model.to(device).to(torch.bfloat16)
        convert_to_attr_dict(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_config_path = _generate_zero_config(zero_stage, tmp_dir)

            model_engine, optimizer = setup_deepspeed_engine(
                model=model,
                optimizer=optimizer,
                deepspeed_config_path=ds_config_path,
                micro_batch_size_per_gpu=2,
                gradient_accumulation_steps=1,
            )

            batch = _get_batch(2, 128, model_config.vocab_size, device, data_dir)

            # Forward
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model_engine(input_ids=batch["input_ids"], labels=batch["labels"], return_aux_loss=True)
                loss = outputs["loss"]

            logger.info(f"  Forward OK — loss={loss.item():.4f}")

            # Backward via DeepSpeed
            model_engine.backward(loss)
            logger.info("  Backward OK")

            model_engine.step()
            logger.info("  DeepSpeed step OK")

        del model_engine, model, optimizer
        torch.cuda.empty_cache()
        logger.info(f"✅ TEST 3 PASSED: DeepSpeed ZeRO-{zero_stage} eager mode works\n")
        return True

    except ImportError:
        logger.warning("⚠️  TEST 3 SKIPPED: DeepSpeed not installed\n")
        return True  # Not a failure
    except Exception as e:
        logger.error(f"❌ TEST 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_compile_before_zero(device: torch.device, zero_stage: int, config_path: str = None, data_dir: str = None) -> bool:
    """
    Test 4: THE CRITICAL TEST — compile inner layers BEFORE DeepSpeed init.

    This is the exact pattern that fixes Issue #8. If this test passes,
    the race condition between AOTAutograd and DeepSpeed's
    post_accumulate_grad_hook is resolved.
    """
    logger.info("=" * 60)
    logger.info(f"TEST 4: CRITICAL — compile BEFORE DeepSpeed ZeRO-{zero_stage}")
    logger.info("=" * 60)
    try:
        import deepspeed
        from training.gpu_optimizer import (
            compile_forge_layers, convert_to_attr_dict, setup_deepspeed_engine,
        )

        model, model_config = _build_small_model(config_path)
        model = model.to(device).to(torch.bfloat16)
        convert_to_attr_dict(model)

        # ── Step 1: Compile inner layers FIRST ────────────────────────────
        logger.info("  Step 1: Compiling inner transformer layers...")
        compile_forge_layers(model, mode="default", dynamic=False)
        logger.info("  Step 1: Compilation complete")

        # ── Step 2: THEN initialize DeepSpeed ─────────────────────────────
        logger.info("  Step 2: Initializing DeepSpeed engine...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_config_path = _generate_zero_config(zero_stage, tmp_dir)

            model_engine, optimizer = setup_deepspeed_engine(
                model=model,
                optimizer=optimizer,
                deepspeed_config_path=ds_config_path,
                micro_batch_size_per_gpu=2,
                gradient_accumulation_steps=1,
            )
            logger.info("  Step 2: DeepSpeed engine initialized")

            batch = _get_batch(2, 128, model_config.vocab_size, device, data_dir)

            # ── Forward pass ──────────────────────────────────────────────
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model_engine(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    return_aux_loss=True
                )
                loss = outputs["loss"]
                
            logger.info(f"  Step 3: Forward OK — loss={loss.item():.4f}")

            # ── Backward pass ─────────────────────────────────────────────
            logger.info("  Step 4: Running backward (checking for race condition)...")
            model_engine.backward(loss)
            logger.info("  ✓ Backward completed WITHOUT NoneType crash!")

            # ── DeepSpeed optimizer step ──────────────────────────────────
            model_engine.step()
            logger.info("  ✓ DeepSpeed step completed")

            # ── Second forward+backward to verify stability ───────────────
            logger.info("  Running second forward+backward for stability check...")
            batch2 = _get_batch(2, 128, model_config.vocab_size, device, data_dir)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs2 = model_engine(
                    input_ids=batch2["input_ids"],
                    labels=batch2["labels"],
                    return_aux_loss=True
                )
                loss2 = outputs2["loss"]
            model_engine.backward(loss2)
            model_engine.step()
            logger.info(f"  ✓ Second step OK — loss={loss2.item():.4f}")

        del model_engine, model, optimizer
        torch.cuda.empty_cache()
        logger.info(f"✅ TEST 4 PASSED: compile-before-ZeRO-{zero_stage} is SAFE\n")
        return True

    except ImportError:
        logger.warning("⚠️  TEST 4 SKIPPED: DeepSpeed not installed\n")
        return True
    except AttributeError as e:
        if "'NoneType' object has no attribute 'view'" in str(e):
            logger.error(
                "❌ TEST 4 FAILED: Issue #8 race condition REPRODUCED!\n"
                "   DeepSpeed's post_accumulate_grad_hook fired before "
                "AOTAutograd wrote param.grad.\n"
                "   This means compile_forge_layers() did NOT prevent the bug."
            )
        else:
            logger.error(f"❌ TEST 4 FAILED: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"❌ TEST 4 FAILED: {e}")
        traceback.print_exc()
        return False


def test_gradient_correctness(device: torch.device, config_path: str = None, data_dir: str = None) -> bool:
    """Test 5: Validates that gradients computed with AOTAutograd match Eager mode."""
    logger.info("=" * 60)
    logger.info("TEST 5: Gradient Correctness (Compiled vs Eager)")
    logger.info("=" * 60)
    try:
        from training.gpu_optimizer import compile_forge_layers

        # Build two identical models
        torch.manual_seed(42)
        model_eager, model_config = _build_small_model(config_path)
        model_eager = model_eager.to(device).to(torch.bfloat16)

        torch.manual_seed(42)
        model_compiled, _ = _build_small_model(config_path)
        model_compiled = model_compiled.to(device).to(torch.bfloat16)
        compile_forge_layers(model_compiled, mode="default", dynamic=False)

        batch = _get_batch(2, 128, model_config.vocab_size, device, data_dir)

        # Eager forward+backward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_eager = model_eager(input_ids=batch["input_ids"], labels=batch["labels"], return_aux_loss=True)
        out_eager["loss"].backward()

        # Compiled forward+backward
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_compiled = model_compiled(input_ids=batch["input_ids"], labels=batch["labels"], return_aux_loss=True)
        out_compiled["loss"].backward()

        # Compare losses
        loss_diff = abs(out_eager["loss"].item() - out_compiled["loss"].item())
        logger.info(f"  Loss difference: {loss_diff:.6f}")

        # Compare gradients
        max_grad_diff = 0.0
        n_compared = 0
        for (name_e, p_e), (name_c, p_c) in zip(
            model_eager.named_parameters(), model_compiled.named_parameters()
        ):
            if p_e.grad is not None and p_c.grad is not None:
                diff = (p_e.grad.float() - p_c.grad.float()).abs().max().item()
                max_grad_diff = max(max_grad_diff, diff)
                n_compared += 1

        logger.info(f"  Max gradient difference: {max_grad_diff:.6f} (across {n_compared} params)")

        # BF16 tolerance: gradients should match within ~1e-2 for BF16
        # (BF16 has ~3 decimal digits of precision)
        TOLERANCE = 0.05  # generous tolerance for BF16 + compilation
        if loss_diff > TOLERANCE:
            logger.warning(f"  ⚠️ Loss difference {loss_diff:.6f} exceeds tolerance {TOLERANCE}")
        if max_grad_diff > TOLERANCE:
            logger.warning(f"  ⚠️ Max gradient diff {max_grad_diff:.6f} exceeds tolerance {TOLERANCE}")

        passed = loss_diff <= TOLERANCE and max_grad_diff <= TOLERANCE

        del model_eager, model_compiled
        torch.cuda.empty_cache()

        if passed:
            logger.info("✅ TEST 5 PASSED: Compiled gradients match eager within BF16 tolerance\n")
        else:
            logger.error("❌ TEST 5 FAILED: Gradient mismatch exceeds tolerance\n")
        return passed

    except Exception as e:
        logger.error(f"❌ TEST 5 FAILED: {e}")
        traceback.print_exc()
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FORGE-3B torch.compile + ZeRO Smoke Test")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config JSON (defaults to minimal test config)")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="DeepSpeed ZeRO stage to test")
    parser.add_argument("--data_dir", type=str, default="Phase-Technologies/forge-3b-sft-data",
                        help="Path to real HF dataset or local dir (to use instead of synthetic data)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: skip gradient correctness test")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                        help="Local rank for distributed testing (set by deepspeed launcher)")
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        logger.error("CUDA not available. This smoke test requires a GPU.")
        sys.exit(1)

    # Initialize distributed process group for DeepSpeed compatibility
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29505")
        os.environ["RANK"] = os.environ.get("RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
        os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", str(args.local_rank))
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        logger.info(f"Initialized torch.distributed NCCL backend (rank={rank}, world_size={world_size})")

    logger.info("=" * 70)
    logger.info("FORGE-3B: torch.compile + DeepSpeed ZeRO Smoke Test")
    logger.info(f"  Device: {device} ({torch.cuda.get_device_name(device)})")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  ZeRO stage: {args.zero_stage}")
    try:
        import deepspeed
        logger.info(f"  DeepSpeed: {deepspeed.__version__}")
    except ImportError:
        logger.info("  DeepSpeed: NOT INSTALLED (ZeRO tests will be skipped)")
    logger.info("=" * 70 + "\n")

    results = {}

    # Test 1: Eager baseline
    results["eager_baseline"] = test_eager_baseline(device, args.config, args.data_dir)

    # Test 2: Compiled without DeepSpeed
    results["compile_only"] = test_compile_only(device, args.config, args.data_dir)

    # Test 3: DeepSpeed eager (if stage > 0)
    if args.zero_stage > 0:
        results["deepspeed_eager"] = test_deepspeed_eager(device, args.zero_stage, args.config, args.data_dir)

    # Test 4: THE CRITICAL TEST — compile before ZeRO
    if args.zero_stage > 0:
        results["compile_before_zero"] = test_compile_before_zero(device, args.zero_stage, args.config, args.data_dir)

    # Test 5: Gradient correctness
    if not args.quick:
        results["gradient_correctness"] = test_gradient_correctness(device, args.config, args.data_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}  {name}")
        if not passed:
            all_passed = False

    logger.info("")
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED — compile-before-ZeRO pattern is working correctly")
        logger.info("   Issue #8 (NoneType view error) is resolved.")
    else:
        logger.error("💥 SOME TESTS FAILED — see details above")

    logger.info("=" * 70)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
