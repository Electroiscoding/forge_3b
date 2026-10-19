"""
Parallel Scan (Blelloch algorithm) in pure PyTorch for ARG selective SSM recurrence.

Eliminates all external dependencies (mamba-ssm, selective_scan_cuda).
Runs in 2*log2(T) parallel steps (11 steps for T=2048) on any GPU.
100% mathematically exact equivalence to sequential SSM recurrence.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def npo2(length: int) -> int:
    """Returns the next power of 2 above length."""
    return 2 ** math.ceil(math.log2(length))


def pad_npo2(x: torch.Tensor) -> torch.Tensor:
    """Pads input length dimension (dim 1) to the next power of 2."""
    length = x.size(1)
    len_p2 = npo2(length)
    if len_p2 == length:
        return x
    pad_tuple = (0, 0, 0, 0, 0, len_p2 - length)
    return F.pad(x, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    """
    Computes the parallel associative scan:
        H[t] = A[t] * H[t-1] + X[t]  with H[-1] = 0
    in O(log2(T)) parallel steps via Blelloch tree reduction.
    """

    @staticmethod
    def pscan(A: torch.Tensor, X: torch.Tensor) -> None:
        # A: (B, D, L, N)
        # X: (B, D, L, N)
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # ── Up sweep ──────────────────────────────────────────────────────────
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # ── Down sweep ────────────────────────────────────────────────────────
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            stride = 2 ** k
            Aa = A[:, :, stride - 1 : L : stride]
            Xa = X[:, :, stride - 1 : L : stride]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A: torch.Tensor, X: torch.Tensor) -> None:
        # Backward scan in reverse time order
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # Up sweep
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # Down sweep
        stride_top = 2 ** (num_steps - 2)
        Aa = A[:, :, 0 : L : stride_top]
        Xa = X[:, :, 0 : L : stride_top]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            stride = 2 ** k
            Aa = A[:, :, 0 : L : stride]
            Xa = X[:, :, 0 : L : stride]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
        # A_in: (B, L, D, N)
        # X_in: (B, L, D, N)
        L = X_in.size(1)

        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            A = pad_npo2(A_in).clone()
            X = pad_npo2(X_in).clone()

        A = A.transpose(2, 1).contiguous()  # (B, D, L_p2, N)
        X = X.transpose(2, 1).contiguous()  # (B, D, L_p2, N)

        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)[:, :L].contiguous()

    @staticmethod
    def backward(ctx, grad_output_in: torch.Tensor):
        A_in, X = ctx.saved_tensors
        L = grad_output_in.size(1)

        if L == npo2(L):
            grad_output = grad_output_in.clone()
        else:
            grad_output = pad_npo2(grad_output_in).clone()
            A_in = pad_npo2(A_in)

        grad_output = grad_output.transpose(2, 1).contiguous()  # (B, D, L_p2, N)
        A_in_t = A_in.transpose(2, 1).contiguous()             # (B, D, L_p2, N)

        # Shift A 1 step to the left for adjoint recurrence
        A_shifted = F.pad(A_in_t[:, :, 1:], (0, 0, 0, 1))

        PScan.pscan_rev(A_shifted, grad_output)

        grad_A = torch.zeros_like(X)
        grad_A[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return grad_A.transpose(2, 1)[:, :L].contiguous(), grad_output.transpose(2, 1)[:, :L].contiguous()


pscan = PScan.apply
