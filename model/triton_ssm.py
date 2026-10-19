"""
Pure Triton Fused Selective State Space Model (SSM) kernel.
Eliminates mamba_ssm / selective_scan_cuda dependencies entirely.
Compiles natively on NVIDIA B300 (sm_100), H100 (sm_90), A100 (sm_80).

Fuses:
1. Discretization: dA = exp(dt * A)
2. State input update: u = dt * x * B
3. Recurrent state accumulation: h_t = dA * h_{t-1} + u_t
4. State output contraction: y_t = sum(h_t * C_t)
5. Skip connection: y_t + x_t * D
6. Output gating: out_t = (y_t + x_t * D) * silu(z_t)
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def ssm_fwd_kernel(
    # Pointers
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, h_all_ptr,
    # Strides
    stride_xb, stride_xd, stride_xt,
    stride_dtb, stride_dtd, stride_dtt,
    stride_Ad, stride_An,
    stride_Bb, stride_Bn, stride_Bt,
    stride_Cb, stride_Cn, stride_Ct,
    stride_outb, stride_outd, stride_outt,
    stride_hb, stride_hd, stride_ht, stride_hn,
    # Dimensions
    T: tl.constexpr,
    N: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = tl.arange(0, N_BLOCK)
    mask_n = offs_n < N

    # Load continuous-time decay A and skip coefficient D
    A = tl.load(A_ptr + pid_d * stride_Ad + offs_n * stride_An, mask=mask_n, other=0.0)
    D = tl.load(D_ptr + pid_d)

    # State h accumulator in registers (FP32 precision)
    h = tl.zeros([N_BLOCK], dtype=tl.float32)

    for t in range(0, T):
        dt_val = tl.load(dt_ptr + pid_b * stride_dtb + pid_d * stride_dtd + t * stride_dtt).to(tl.float32)
        x_val = tl.load(x_ptr + pid_b * stride_xb + pid_d * stride_xd + t * stride_xt).to(tl.float32)
        z_val = tl.load(z_ptr + pid_b * stride_xb + pid_d * stride_xd + t * stride_xt).to(tl.float32)

        B_vec = tl.load(B_ptr + pid_b * stride_Bb + offs_n * stride_Bn + t * stride_Bt, mask=mask_n, other=0.0).to(tl.float32)
        C_vec = tl.load(C_ptr + pid_b * stride_Cb + offs_n * stride_Cn + t * stride_Ct, mask=mask_n, other=0.0).to(tl.float32)

        # Discretize: dA = exp(dt * A)
        dA = tl.exp(dt_val * A)

        # u = dt * x * B
        u = (dt_val * x_val) * B_vec

        # State transition: h = dA * h + u
        h = dA * h + u

        # Save state snapshot for exact backward gradient flow
        tl.store(h_all_ptr + pid_b * stride_hb + pid_d * stride_hd + t * stride_ht + offs_n * stride_hn, h, mask=mask_n)

        # Output contraction: y = sum(h * C)
        y = tl.sum(tl.where(mask_n, h * C_vec, 0.0), axis=0)

        # Skip connection + SiLU gate: out = (y + x * D) * silu(z)
        sig_z = 1.0 / (1.0 + tl.exp(-z_val))
        silu_z = z_val * sig_z
        out_val = (y + x_val * D) * silu_z

        tl.store(out_ptr + pid_b * stride_outb + pid_d * stride_outd + t * stride_outt, out_val)


@triton.jit
def ssm_bwd_kernel(
    # Pointers
    dout_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, h_all_ptr,
    dx_ptr, ddt_ptr, dA_ptr, dB_ptr, dC_ptr, dD_ptr, dz_ptr,
    # Strides
    stride_xb, stride_xd, stride_xt,
    stride_dtb, stride_dtd, stride_dtt,
    stride_Ad, stride_An,
    stride_Bb, stride_Bn, stride_Bt,
    stride_Cb, stride_Cn, stride_Ct,
    stride_doutb, stride_doutd, stride_doutt,
    stride_hb, stride_hd, stride_ht, stride_hn,
    # Dimensions
    T: tl.constexpr,
    N: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_n = tl.arange(0, N_BLOCK)
    mask_n = offs_n < N

    A = tl.load(A_ptr + pid_d * stride_Ad + offs_n * stride_An, mask=mask_n, other=0.0)
    D = tl.load(D_ptr + pid_d)

    lambda_h = tl.zeros([N_BLOCK], dtype=tl.float32)
    accum_dD = 0.0
    accum_dA = tl.zeros([N_BLOCK], dtype=tl.float32)

    # Reverse adjoint recurrence from T-1 down to 0
    for t in range(T - 1, -1, -1):
        dout_val = tl.load(dout_ptr + pid_b * stride_doutb + pid_d * stride_doutd + t * stride_doutt).to(tl.float32)
        x_val = tl.load(x_ptr + pid_b * stride_xb + pid_d * stride_xd + t * stride_xt).to(tl.float32)
        z_val = tl.load(z_ptr + pid_b * stride_xb + pid_d * stride_xd + t * stride_xt).to(tl.float32)
        dt_val = tl.load(dt_ptr + pid_b * stride_dtb + pid_d * stride_dtd + t * stride_dtt).to(tl.float32)

        B_vec = tl.load(B_ptr + pid_b * stride_Bb + offs_n * stride_Bn + t * stride_Bt, mask=mask_n, other=0.0).to(tl.float32)
        C_vec = tl.load(C_ptr + pid_b * stride_Cb + offs_n * stride_Cn + t * stride_Ct, mask=mask_n, other=0.0).to(tl.float32)
        h_t = tl.load(h_all_ptr + pid_b * stride_hb + pid_d * stride_hd + t * stride_ht + offs_n * stride_hn, mask=mask_n, other=0.0).to(tl.float32)

        if t > 0:
            h_prev = tl.load(h_all_ptr + pid_b * stride_hb + pid_d * stride_hd + (t - 1) * stride_ht + offs_n * stride_hn, mask=mask_n, other=0.0).to(tl.float32)
        else:
            h_prev = tl.zeros([N_BLOCK], dtype=tl.float32)

        # SiLU backward: silu(z) = z * sig(z); d/dz = sig(z) + z*sig(z)*(1 - sig(z))
        sig_z = 1.0 / (1.0 + tl.exp(-z_val))
        silu_z = z_val * sig_z
        dsilu_z = sig_z * (1.0 + z_val * (1.0 - sig_z))

        y = tl.sum(tl.where(mask_n, h_t * C_vec, 0.0), axis=0)
        core = y + x_val * D

        dz_val = dout_val * core * dsilu_z
        dy = dout_val * silu_z

        tl.store(dz_ptr + pid_b * stride_xb + pid_d * stride_xd + t * stride_xt, dz_val)

        accum_dD += dy * x_val

        # Adjoint state update
        dh_t = dy * C_vec + lambda_h

        # Gradient with respect to C
        dC_vec = tl.where(mask_n, dy * h_t, 0.0)
        tl.atomic_add(dC_ptr + pid_b * stride_Cb + offs_n * stride_Cn + t * stride_Ct, dC_vec, mask=mask_n)

        # Gradient with respect to x
        du_x_dt = tl.sum(tl.where(mask_n, dh_t * B_vec, 0.0), axis=0)
        dx_val = dy * D + du_x_dt * dt_val
        tl.store(dx_ptr + pid_b * stride_xb + pid_d * stride_xd + t * stride_xt, dx_val)

        # Gradient with respect to B
        dB_vec = tl.where(mask_n, dh_t * (dt_val * x_val), 0.0)
        tl.atomic_add(dB_ptr + pid_b * stride_Bb + offs_n * stride_Bn + t * stride_Bt, dB_vec, mask=mask_n)

        # Gradient with respect to dt and A
        dA_step = tl.exp(dt_val * A)
        term_A = tl.sum(tl.where(mask_n, dh_t * h_prev * A * dA_step, 0.0), axis=0)
        ddt_val = du_x_dt * x_val + term_A
        tl.store(ddt_ptr + pid_b * stride_dtb + pid_d * stride_dtd + t * stride_dtt, ddt_val)

        accum_dA += tl.where(mask_n, dh_t * h_prev * dt_val * dA_step, 0.0)

        # Next adjoint state
        lambda_h = dh_t * dA_step

    tl.atomic_add(dA_ptr + pid_d * stride_Ad + offs_n * stride_An, accum_dA, mask=mask_n)
    tl.atomic_add(dD_ptr + pid_d, accum_dD)


class TritonSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B_ssm, C_ssm, D_skip, z):
        B_sz, d_in, T_sz = x.shape
        d_st = A.shape[1]
        out = torch.empty_like(x)
        # Snapshot of states for backward
        h_all = torch.empty(B_sz, d_in, T_sz, d_st, device=x.device, dtype=torch.bfloat16)

        grid = (B_sz, d_in)
        ssm_fwd_kernel[grid](
            x, dt, A, B_ssm, C_ssm, D_skip, z, out, h_all,
            x.stride(0), x.stride(1), x.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0), A.stride(1),
            B_ssm.stride(0), B_ssm.stride(1), B_ssm.stride(2),
            C_ssm.stride(0), C_ssm.stride(1), C_ssm.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            h_all.stride(0), h_all.stride(1), h_all.stride(2), h_all.stride(3),
            T=T_sz, N=d_st, N_BLOCK=64,
        )
        ctx.save_for_backward(x, dt, A, B_ssm, C_ssm, D_skip, z, h_all)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, dt, A, B_ssm, C_ssm, D_skip, z, h_all = ctx.saved_tensors
        B_sz, d_in, T_sz = x.shape
        d_st = A.shape[1]

        dx = torch.empty_like(x)
        ddt = torch.empty_like(dt)
        dz = torch.empty_like(z)
        dA = torch.zeros_like(A, dtype=torch.float32)
        dD = torch.zeros_like(D_skip, dtype=torch.float32)
        dB = torch.zeros_like(B_ssm, dtype=torch.float32)
        dC = torch.zeros_like(C_ssm, dtype=torch.float32)

        grid = (B_sz, d_in)
        ssm_bwd_kernel[grid](
            dout, x, dt, A, B_ssm, C_ssm, D_skip, z, h_all,
            dx, ddt, dA, dB, dC, dD, dz,
            x.stride(0), x.stride(1), x.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0), A.stride(1),
            B_ssm.stride(0), B_ssm.stride(1), B_ssm.stride(2),
            C_ssm.stride(0), C_ssm.stride(1), C_ssm.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2),
            h_all.stride(0), h_all.stride(1), h_all.stride(2), h_all.stride(3),
            T=T_sz, N=d_st, N_BLOCK=64,
        )
        return (
            dx,
            ddt,
            dA.to(A.dtype),
            dB.to(B_ssm.dtype),
            dC.to(C_ssm.dtype),
            dD.to(D_skip.dtype),
            dz,
        )


def fused_triton_ssm(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_ssm: torch.Tensor,
    C_ssm: torch.Tensor,
    D_skip: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Fused selective scan forward and backward in pure Triton.
    Inputs:
      x:      (B, d_inner, T)
      dt:     (B, d_inner, T)
      A:      (d_inner, d_state)
      B_ssm:  (B, d_state, T)
      C_ssm:  (B, d_state, T)
      D_skip: (d_inner,)
      z:      (B, d_inner, T)
    Returns:
      out:    (B, d_inner, T)
    """
    return TritonSSMFunction.apply(
        x.contiguous(),
        dt.contiguous(),
        A.contiguous(),
        B_ssm.contiguous(),
        C_ssm.contiguous(),
        D_skip.contiguous(),
        z.contiguous(),
    )
