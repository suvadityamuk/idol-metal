"""Fused Metal (MPS) kernels for the Idol image-to-3D denoiser blocks.

Each op keeps the rounding points of the PyTorch op chain it replaces; see CARD.md for the exact chains.
"""
from typing import Optional, Tuple

import torch

from ._ops import ops

__all__ = ["resid_ln_mod", "repo_rope_qk", "silu_mul", "gelu_tanh_", "gated_residual", "linear_gelu",
           "rmsnorm_heads", "resid_layernorm"]


def resid_ln_mod(x: torch.Tensor, scale1p: torch.Tensor, shift: torch.Tensor, h: Optional[torch.Tensor] = None,
                 gate: Optional[torch.Tensor] = None, repo_weight: Optional[torch.Tensor] = None,
                 repo_bias: Optional[torch.Tensor] = None, eps: float = 1e-6, eps_repo: float = 1e-5
                 ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """x' = x + h * gate (half; only when h is given), then
    y  = LayerNorm(x'.float(), eps).half() * scale1p + shift      (half arithmetic, as adaLN modulation)
    ry = LayerNorm(x'.float(), eps_repo) * repo_weight + repo_bias (float; only when repo_weight is given)
    Returns (x', y, ry). x: [..., C] half, C % 4 == 0, C <= 4096; scale1p/shift/gate: [C] half;
    repo_weight/repo_bias: [C] float."""
    y = torch.empty_like(x)
    xout = torch.empty_like(x) if h is not None else None
    ry = torch.empty(x.shape, device=x.device, dtype=torch.float32) if repo_weight is not None else None
    ops.resid_ln_mod(y, xout, ry, x, h, gate, scale1p, shift, repo_weight, repo_bias, eps, eps_repo)
    return (xout if xout is not None else x), y, ry


def repo_rope_qk(qkv: torch.Tensor, delta: torch.Tensor, ft: torch.Tensor, fd: torch.Tensor, axis: torch.Tensor,
                 q_gamma: torch.Tensor, k_gamma: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """RePo rotary angles ang_j = (delta[axis_j] * ft_j + delta[axis_j] * fd_j) * pi (ft = tanh(f), fd = f - tanh(f)),
    RoPE on q and k (pairs as complex numbers, result rounded to half), then
    F.normalize(.float(), dim=-1) * gamma * scale -> half.
    qkv: [R, 3, H, 64] half; delta: [R, H * 3] float; ft, fd: [32] float; axis: [32] int32 (0/1/2);
    q_gamma, k_gamma: [H, 64] float. Returns q, k: [R, H, 64] half."""
    R, _, H, D = qkv.shape
    q = torch.empty(R, H, D, device=qkv.device, dtype=torch.float16)
    k = torch.empty_like(q)
    ops.repo_rope_qk(q, k, qkv, delta, ft, fd, axis, q_gamma, k_gamma, scale)
    return q, k


def silu_mul(gate: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """silu(gate) * content, float."""
    out = torch.empty_like(gate)
    ops.silu_mul(out, gate, content)
    return out


def gelu_tanh_(x: torch.Tensor) -> torch.Tensor:
    """In-place gelu(x, approximate='tanh') on a half tensor."""
    ops.gelu_tanh_(x)
    return x


def gated_residual(x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """x + h * gate (half; gate [C] broadcast over the last dim)."""
    out = torch.empty_like(x)
    ops.gated_residual(out, x, h, gate)
    return out


def linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """gelu(F.linear(x, weight, bias), approximate='tanh') for half x [M, K], weight [N, K], bias [N], with the
    GELU applied in the GEMM epilogue (Metal 4 MetalPerformancePrimitives matmul2d)."""
    out = torch.empty(x.shape[0], weight.shape[0], device=x.device, dtype=torch.float16)
    ops.linear_gelu(out, x, weight, bias)
    return out


def rmsnorm_heads(src: torch.Tensor, gamma: torch.Tensor, scale: float, col_offset: int = 0) -> torch.Tensor:
    """F.normalize(x.float(), dim=-1) * gamma * scale -> half for x = src[:, col_offset : col_offset + H * 64]
    viewed as [R, H, 64]; src: [R, W] half contiguous; gamma: [H, 64] float."""
    H = gamma.shape[0]
    out = torch.empty(src.shape[0], H, 64, device=src.device, dtype=torch.float16)
    ops.rmsnorm_heads(out, src, col_offset, gamma, scale)
    return out


def resid_layernorm(x: torch.Tensor, h: Optional[torch.Tensor] = None, weight: Optional[torch.Tensor] = None,
                    bias: Optional[torch.Tensor] = None, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """x' = x + h (half; only when h is given), y = LayerNorm(x'.float(), eps, weight, bias).half().
    Returns (x', y). x: [..., C] half; weight/bias: [C] float or None."""
    y = torch.empty_like(x)
    xout = torch.empty_like(x) if h is not None else None
    ops.resid_layernorm(y, xout, x, h, weight, bias, eps)
    return (xout if xout is not None else x), y
