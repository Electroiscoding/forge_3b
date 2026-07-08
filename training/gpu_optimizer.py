"""
GPU Optimization Utilities for FORGE-3B Training.

Implements every available GPU optimization for H100/A100:
- torch.compile with max-autotune
- BF16/FP8 mixed precision contexts
- CUDA stream management
- Memory optimization utilities
- NCCL communication optimization
- Profiling hooks
"""

from __future__ import annotations
import os
import gc
import time
import logging
import contextlib
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GPU WARM-UP AND INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def warmup_gpu(device: torch.device, n_warmup_steps: int = 10):
    """
    Warm up GPU by running dummy computation.
    Ensures CUDA context is initialized and caches are populated before timing.
    """
    logger.info(f"Warming up GPU {device}...")
    
    dummy = torch.randn(512, 4096, device=device, dtype=torch.bfloat16)
    for _ in range(n_warmup_steps):
        result = torch.matmul(dummy, dummy.T)
        result.mean().backward() if result.requires_grad else None
    
    torch.cuda.synchronize(device)
    del dummy, result
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info(f"GPU {device} warmed up. "
                f"Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB allocated, "
                f"{torch.cuda.memory_reserved(device)/1e9:.2f}GB reserved")


def setup_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> Tuple[int, int, int]:
    """
    Initialize torch distributed training.
    Returns (rank, world_size, local_rank).
    """
    import datetime
    
    if "RANK" not in os.environ:
        # Single-GPU training
        return 0, 1, 0
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )
    
    torch.cuda.set_device(local_rank)
    
    # NCCL environment tuning for NVLink
    os.environ["NCCL_MIN_NCHANNELS"] = str(min(world_size * 2, 8))
    os.environ["NCCL_CROSS_NIC"] = "0"
    
    logger.info(f"Distributed initialized: rank={rank}/{world_size}, local={local_rank}")
    return rank, world_size, local_rank


# ─────────────────────────────────────────────────────────────────────────────
# torch.compile WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(
    model: nn.Module,
    mode: str = "max-autotune",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> nn.Module:
    """
    Apply torch.compile to the model with maximum optimization settings.
    
    Modes:
    - "default": standard optimization
    - "reduce-overhead": eliminate Python overhead (fastest for small batches)
    - "max-autotune": profile kernels and select the best (slowest first run)
    
    max-autotune typically gives 15-40% throughput improvement on H100.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — skipping torch.compile")
        return model
    
    major_ver = torch.version.cuda and int(torch.version.cuda.split('.')[0])
    if major_ver and major_ver < 11:
        logger.warning("CUDA < 11 — torch.compile may not be optimal")
    
    logger.info(f"Applying torch.compile(mode={mode!r}, fullgraph={fullgraph}, dynamic={dynamic})")
    
    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend="inductor",
        )
        logger.info("torch.compile applied successfully")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}. Using uncompiled model.")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# MIXED PRECISION CONTEXTS
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def bf16_autocast(device_type: str = "cuda", enabled: bool = True):
    """Context manager for BF16 mixed precision training."""
    if not enabled:
        yield
        return
    
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
        yield


@contextlib.contextmanager  
def fp8_autocast_context(enabled: bool = True):
    """
    FP8 training context for H100.
    Requires transformer-engine or torchao.
    Falls back to BF16 if unavailable.
    """
    if not enabled:
        with bf16_autocast():
            yield
        return
    
    try:
        import transformer_engine.pytorch as te
        with te.fp8_autocast(enabled=True):
            yield
        logger.debug("FP8 forward pass completed via TransformerEngine")
    except ImportError:
        logger.warning("TransformerEngine not available — falling back to BF16")
        with bf16_autocast():
            yield


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class GPUMemoryMonitor:
    """Real-time GPU memory monitoring with OOM prevention."""
    
    def __init__(self, device: torch.device, oom_threshold_gb: float = 5.0):
        self.device = device
        self.oom_threshold_gb = oom_threshold_gb
        self._peak_allocated = 0.0
        self._peak_reserved = 0.0
    
    def snapshot(self) -> Dict[str, float]:
        """Get current memory stats in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        free = total - reserved
        
        self._peak_allocated = max(self._peak_allocated, allocated)
        self._peak_reserved = max(self._peak_reserved, reserved)
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "total_gb": total,
            "peak_allocated_gb": self._peak_allocated,
        }
    
    def check_oom_risk(self) -> bool:
        """Return True if close to OOM."""
        stats = self.snapshot()
        return stats["free_gb"] < self.oom_threshold_gb
    
    def log_memory(self, step: int = -1, prefix: str = ""):
        s = self.snapshot()
        tag = f"[{prefix}]" if prefix else ""
        step_str = f" step={step}" if step >= 0 else ""
        logger.info(
            f"GPU Memory{tag}{step_str}: "
            f"{s['allocated_gb']:.2f}GB alloc / "
            f"{s['reserved_gb']:.2f}GB reserved / "
            f"{s['free_gb']:.2f}GB free / "
            f"{s['total_gb']:.2f}GB total"
        )
    
    def clear(self):
        """Force memory cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)


def offload_to_cpu(tensor: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
    """Move tensor to CPU with optional async transfer."""
    return tensor.to("cpu", non_blocking=non_blocking)


def load_to_gpu(tensor: torch.Tensor, device: torch.device, 
                 non_blocking: bool = True) -> torch.Tensor:
    """Move tensor to GPU with pinned memory for max bandwidth."""
    if not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=non_blocking)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def clip_grad_norm_and_log(
    parameters,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """Clip gradients and return the pre-clip global norm for logging."""
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return float(grad_norm)


def log_gradient_stats(model: nn.Module, step: int, log_every: int = 100):
    """Log gradient statistics for debugging training instability."""
    if step % log_every != 0:
        return
    
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.float()
            stats[name] = {
                "mean": float(g.mean()),
                "std": float(g.std()),
                "max": float(g.abs().max()),
                "norm": float(g.norm()),
            }
    
    # Flag problematic gradients
    for name, s in stats.items():
        if s["max"] > 100 or s["std"] == 0:
            logger.warning(f"Gradient issue at {name}: {s}")
    
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# THROUGHPUT MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────

class ThroughputMeter:
    """
    Measure training throughput in tokens/second and MFU (Model FLOPs Utilization).
    """
    
    def __init__(
        self,
        model_flops_per_token: float,   # 6 * active_params
        device: torch.device,
        world_size: int = 1,
    ):
        self.model_flops_per_token = model_flops_per_token
        self.device = device
        self.world_size = world_size
        
        # Get theoretical peak TFLOPS
        self.peak_tflops = self._get_peak_tflops()
        
        self._step_times: List[float] = []
        self._tokens_per_step: List[int] = []
        self._window_size = 100
        
        self._step_start: Optional[float] = None
    
    def _get_peak_tflops(self) -> float:
        """Get BF16 peak TFLOPS for this GPU."""
        if not torch.cuda.is_available():
            return 1.0
        
        props = torch.cuda.get_device_properties(self.device)
        name = props.name.lower()
        
        # Known peak BF16 TFLOPS values
        gpu_tflops = {
            "h100 sxm": 1979.0,
            "h100 pcie": 989.0,
            "mi300": 653.7,
            "a100 sxm": 312.0,
            "a100 pcie": 312.0,
            "a40": 37.4,
            "rtx 4090": 82.6,
            "rtx 3090": 35.6,
            "v100": 14.1,
        }
        
        for key, tflops in gpu_tflops.items():
            if key in name:
                logger.info(f"GPU detected: {props.name} — {tflops} TFLOPS BF16 (theoretical)")
                return tflops
        
        # Estimate from SM count and clock
        sm_count = props.multi_processor_count
        clock_rate = getattr(props, "max_clock_rate", None)
        if clock_rate is not None:
            clock_ghz = clock_rate / 1e6
        else:
            clock_ghz = 1.5  # standard fallback clock speed (1.5 GHz)
        estimated = sm_count * 128 * 2 * clock_ghz / 1e3  # rough estimate
        logger.warning(f"Unknown GPU {props.name} — estimating {estimated:.1f} TFLOPS")
        return estimated
    
    def start_step(self):
        torch.cuda.synchronize(self.device)
        self._step_start = time.perf_counter()
    
    def end_step(self, n_tokens: int):
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self._step_start
        self._step_times.append(elapsed)
        self._tokens_per_step.append(n_tokens)
        
        # Keep rolling window
        if len(self._step_times) > self._window_size:
            self._step_times.pop(0)
            self._tokens_per_step.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        if not self._step_times:
            return {}
        
        total_time = sum(self._step_times)
        total_tokens = sum(self._tokens_per_step)
        
        tok_per_sec = total_tokens / total_time * self.world_size
        
        # MFU = actual_flops / (peak_flops * time)
        actual_tflops = self.model_flops_per_token * tok_per_sec / 1e12
        mfu = actual_tflops / (self.peak_tflops * self.world_size)
        
        return {
            "tokens_per_sec": tok_per_sec,
            "tokens_per_sec_per_gpu": tok_per_sec / self.world_size,
            "actual_tflops": actual_tflops,
            "peak_tflops_total": self.peak_tflops * self.world_size,
            "mfu": mfu,
            "step_time_ms": (total_time / len(self._step_times)) * 1000,
        }
    
    def log_stats(self, step: int):
        stats = self.get_stats()
        if stats:
            logger.info(
                f"[Step {step}] "
                f"tok/s={stats['tokens_per_sec']:,.0f} "
                f"({stats['tokens_per_sec_per_gpu']:,.0f}/GPU) | "
                f"MFU={stats['mfu']*100:.1f}% | "
                f"step={stats['step_time_ms']:.1f}ms"
            )