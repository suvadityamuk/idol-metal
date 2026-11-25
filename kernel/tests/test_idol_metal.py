"""Each op against the PyTorch (MPS) op chain it replaces. The kernels keep the chain's rounding points, so most
comparisons are exact (byte-equal); see EXACT below for the ones where the summation order is not PyTorch's."""
import math

import pytest
import torch
import torch.nn.functional as F

H, D = 16, 64


def offset_view(t: torch.Tensor, pad: int = 3) -> torch.Tensor:
    """The same values in a contiguous tensor with a non-zero storage offset."""
    big = torch.empty(t.numel() + pad, device=t.device, dtype=t.dtype)
    v = big[pad:].view(t.shape)
    v.copy_(t)
    assert v.storage_offset() == pad and v.is_contiguous()
    return v


def ref_ln_mod(x, s1p, sh, h=None, gate=None, rw=None, rb=None, eps=1e-6, eps_r=1e-5):
    if h is not None:
        x = x + h * gate
    y = F.layer_norm(x.float(), x.shape[-1:], eps=eps).half()
    y = y * s1p
    y = y + sh
    ry = F.layer_norm(x.float(), x.shape[-1:], rw, rb, eps=eps_r) if rw is not None else None
    return x, y, ry


@pytest.mark.parametrize("R,C", [(37, 1024), (1, 64), (513, 256)])
@pytest.mark.parametrize("resid", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_resid_ln_mod(kernel, device, R, C, resid, offset):
    g = torch.Generator(device="cpu").manual_seed(R + C)
    mk = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(device)
    x = (mk(R, C, sc=3.0) + 0.5).half()
    h = mk(R, C).half() if resid else None
    gate = mk(C, sc=0.5).half() if resid else None
    s1p, sh = (1 + mk(C, sc=0.2)).half(), mk(C, sc=0.2).half()
    rw, rb = 1 + mk(C, sc=0.1), mk(C, sc=0.1)
    if offset:
        x = offset_view(x)
        h = offset_view(h) if h is not None else None
    xo, y, ry = kernel.resid_ln_mod(x, s1p, sh, h=h, gate=gate, repo_weight=rw, repo_bias=rb)
    xr, yr, ryr = ref_ln_mod(x, s1p, sh, h, gate, rw, rb)
    assert torch.equal(xo, xr)
    assert torch.equal(y, yr)
    assert torch.equal(ry, ryr)


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("resid", [False, True])
def test_resid_layernorm(kernel, device, affine, resid):
    R, C = 77, 1024
    x = (torch.randn(R, C, device=device) * 2).half()
    h = torch.randn(R, C, device=device).half() if resid else None
    w, b = (1 + 0.1 * torch.randn(C, device=device), 0.1 * torch.randn(C, device=device)) if affine else (None, None)
    xo, y = kernel.resid_layernorm(x, h=h, weight=w, bias=b, eps=1e-6)
    xr = x + h if resid else x
    yr = F.layer_norm(xr.float(), (C,), w, b, eps=1e-6).half()
    assert torch.equal(xo, xr)
    assert torch.equal(y, yr)  # torch.equal treats -0 == +0 (the non-affine path may turn -0 into +0)


def ref_repo_rope_qk(qkv, delta, freqs, qg, kg, scale):
    R = qkv.shape[0]
    d = delta.view(R, H, 3)
    angs = []
    for a, f in enumerate(freqs):
        x = d[..., a].unsqueeze(-1)
        ft = f.tanh()
        angs.append((x * ft + x * (f - ft)) * math.pi)
    ang = torch.cat(angs, -1)
    fr = torch.polar(torch.ones_like(ang), ang)
    out = []
    for which, gm in ((0, qg), (1, kg)):
        x = qkv[:, which]
        xr = torch.view_as_complex(x.float().reshape(R, H, D // 2, 2)) * fr
        xr = torch.view_as_real(xr).reshape(R, H, D).type_as(x)
        out.append((F.normalize(xr.float(), dim=-1) * gm * scale).half())
    return out


def assert_half_close(out, ref, max_frac):
    """At most one half ulp of difference, on at most max_frac of the elements."""
    diff = out != ref
    assert diff.float().mean().item() <= max_frac
    if diff.any():
        a, b = out[diff].float(), ref[diff].float()
        ulp = torch.maximum(a.abs(), b.abs()) * 2.0 ** -10
        assert bool(((a - b).abs() <= ulp * 1.0001).all())


@pytest.mark.parametrize("R", [1, 37, 64, 4101, 12294])
def test_repo_rope_qk(kernel, device, R):
    torch.manual_seed(R)
    qkv = torch.randn(R, 3, H, D, device=device).half()
    delta = torch.randn(R, H * 3, device=device) * 0.5
    freqs = [torch.linspace(1.0, 16.0, n, device=device) for n in (10, 10, 12)]
    ft = torch.cat([f.tanh() for f in freqs])
    fd = torch.cat([f - f.tanh() for f in freqs])
    axis = torch.tensor([0] * 10 + [1] * 10 + [2] * 12, dtype=torch.int32, device=device)
    qg, kg = 1 + 0.1 * torch.randn(H, D, device=device), 1 + 0.1 * torch.randn(H, D, device=device)
    q, k = kernel.repo_rope_qk(qkv, delta, ft, fd, axis, qg, kg, 8.0)
    qr, kr = ref_repo_rope_qk(qkv, delta, freqs, qg, kg, 8.0)
    # The kernel mirrors the q/k norm reduction PyTorch uses at the model's sizes (exact at these row counts,
    # and on the model's own tensors); at some small row counts (e.g. 36-38) PyTorch's normalize takes another
    # reduction path and a few values land one half ulp apart.
    if R in (1, 64, 4101, 12294):
        assert torch.equal(q, qr)
        assert torch.equal(k, kr)
    else:
        assert_half_close(q, qr, 1e-3)
        assert_half_close(k, kr, 1e-3)


def test_silu_mul(kernel, device):
    g, c = torch.randn(1000, 128, device=device) * 3, torch.randn(1000, 128, device=device)
    assert torch.equal(kernel.silu_mul(g, c), F.silu(g) * c)


@pytest.mark.parametrize("n", [1, 7, 4096 * 3 + 5])
def test_gelu_tanh_(kernel, device, n):
    x = (torch.randn(n, device=device) * 4).half()
    ref = F.gelu(x, approximate="tanh")
    assert torch.equal(kernel.gelu_tanh_(x.clone()), ref)


def test_gated_residual(kernel, device):
    x, h = torch.randn(300, 1024, device=device).half(), torch.randn(300, 1024, device=device).half()
    gate = torch.randn(1024, device=device).half()
    assert torch.equal(kernel.gated_residual(offset_view(x), h, gate), x + h * gate)


@pytest.mark.parametrize("M,K,N", [(37, 1024, 4096), (12294, 1024, 4096), (130, 64, 128), (5, 48, 200)])
def test_linear_gelu(kernel, device, M, K, N):
    torch.manual_seed(M + N)
    x = torch.randn(M, K, device=device).half()
    w = (torch.randn(N, K, device=device) / math.sqrt(K)).half()
    b = torch.randn(N, device=device).half()
    ref = F.gelu(F.linear(x, w, b), approximate="tanh")
    assert torch.equal(kernel.linear_gelu(x, w, b), ref)


@pytest.mark.parametrize("col_offset,W", [(0, 1024), (1024, 3072), (0, 2048)])
def test_rmsnorm_heads(kernel, device, col_offset, W):
    R = 333
    src = torch.randn(R, W, device=device).half()
    gamma = 1 + 0.1 * torch.randn(H, D, device=device)
    out = kernel.rmsnorm_heads(src, gamma, 8.0, col_offset=col_offset)
    x = src[:, col_offset:col_offset + H * D].reshape(R, H, D)
    ref = (F.normalize(x.float(), dim=-1) * gamma * 8.0).half()
    assert torch.equal(out, ref)


def test_bad_arguments(kernel, device):
    x = torch.randn(4, 64, device=device).half()
    one, zero = torch.ones(64, device=device).half(), torch.zeros(64, device=device).half()
    with pytest.raises(RuntimeError):
        kernel.resid_ln_mod(x.float(), one, zero)                 # wrong dtype
    with pytest.raises(RuntimeError):
        kernel.resid_ln_mod(x.cpu(), one.cpu(), zero.cpu())       # not MPS
    with pytest.raises(RuntimeError):
        kernel.resid_ln_mod(x.t(), one[:4], zero[:4])             # not contiguous
    with pytest.raises(RuntimeError):
        kernel.resid_ln_mod(x, one[:3], zero)                     # wrong vector length
    with pytest.raises(RuntimeError):
        kernel.ops.resid_ln_mod(torch.empty_like(x), None, None, x, x, None, one, zero, None, None, 1e-6, 1e-5)
    with pytest.raises(RuntimeError):
        kernel.linear_gelu(x, torch.randn(8, 32, device=device).half(), torch.zeros(8, device=device).half())
    with pytest.raises(RuntimeError):
        kernel.rmsnorm_heads(torch.randn(4, 100, device=device).half(), torch.ones(2, 64, device=device), 8.0)
    with pytest.raises(RuntimeError):
        qkv = torch.randn(4, 3, H, 32, device=device).half()     # head dim must be 64
        kernel.repo_rope_qk(qkv, torch.zeros(4, H * 3, device=device), torch.zeros(32, device=device),
                            torch.zeros(32, device=device), torch.zeros(32, dtype=torch.int32, device=device),
                            torch.ones(H, 32, device=device), torch.ones(H, 32, device=device), 8.0)
