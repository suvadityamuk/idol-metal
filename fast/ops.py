"""Fused Metal kernels for the denoiser and decoder blocks, run around PyTorch's MPS GEMMs and SDPA.

Kernels come from kernel/idol_metal/idol_kernels.metal (the same source the native runner and the kernel-builder
package use). Backend (use_package() switches at run time):
  shader (default)  torch.mps.compile_shader on the source (development; no build step)
  package           the kernel-builder package `idol_metal` (a local build/ or the Hub), see scripts/run_kernels.py
"""
import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'kernel', 'idol_metal', 'idol_kernels.metal')

_lib = None
FUSE_GELU = os.environ.get('IDOL_FUSE_GELU', '1') == '1'


PKG = None   # the loaded kernel-builder package when the package backend is active


def use_package(pkg):
    """Route every wrapper below through the packaged ops (get_kernel / get_local_kernel result)."""
    global PKG
    PKG = pkg


def lib():
    global _lib
    if _lib is None:
        with open(SRC) as f:
            _lib = torch.mps.compile_shader(f.read())
    return _lib


def _dummy(dev, dtype):
    return torch.empty(4, device=dev, dtype=dtype)


# -- primitive wrappers -------------------------------------------------------------------------------------------

def resid_ln_mod_repo(x, scale1p, shift, h=None, gate=None, repo=None, eps_ln=1e-6):
    """-> (x_new or x, ln_mod half, repo_ln float or None). x [L, C] half contiguous."""
    if PKG is not None:
        rw, rb, eps_r = repo if repo is not None else (None, None, 1e-5)
        return PKG.resid_ln_mod(x, scale1p, shift, h=h, gate=gate, repo_weight=rw, repo_bias=rb, eps=eps_ln,
                                eps_repo=eps_r)
    L, C = x.shape
    dev = x.device
    mode = 0
    xout = x
    if h is not None:
        mode |= 1
        xout = torch.empty_like(x)
    y = torch.empty_like(x)
    if repo is not None:
        mode |= 2
        rw, rb, eps_r = repo
        ry = torch.empty(L, C, device=dev, dtype=torch.float32)
    else:
        rw = rb = ry = _dummy(dev, torch.float32)
        eps_r = 1e-5
    lib().fused_resid_ln_mod_repo(x, h if h is not None else x, gate if gate is not None else scale1p, xout,
                                  scale1p, shift, y, rw, rb, ry, C, eps_ln, eps_r, mode,
                                  threads=L * (C // 4), group_size=C // 4)
    return xout, y, (ry if repo is not None else None)


def silu_mul(g, c):
    if PKG is not None:
        return PKG.silu_mul(g, c)
    out = torch.empty_like(g)
    lib().exact_silu_mul_f32(g, c, out, threads=g.numel(), group_size=256)
    return out


def repo_rope_qk(qkv, delta, ft, fd, axis, qg, kg, scale):
    if PKG is not None:
        return PKG.repo_rope_qk(qkv.view(qkv.shape[0], 3, 16, 64), delta, ft, fd, axis, qg, kg, scale)
    L = qkv.shape[0]
    q = torch.empty(L, 16, 64, device=qkv.device, dtype=torch.float16)
    k = torch.empty_like(q)
    lib().fused_repo_rope_qk(qkv, delta, ft, fd, axis, qg, kg, q, k, L, scale, 16, threads=L * 16 * 4, group_size=256)
    return q, k


def gelu_(x):
    if PKG is not None:
        return PKG.gelu_tanh_(x)
    n = x.numel()
    lib().fused_gelu_tanh_half_inplace(x, n, threads=(n + 3) // 4, group_size=256)
    return x


def linear_gelu(x, w, b):
    """gelu_tanh(F.linear(x, w, b)) for half x [M, K], w [N, K], b [N] (fused MPP GEMM epilogue)."""
    if PKG is not None:
        return PKG.linear_gelu(x, w, b)
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=torch.float16)
    lib().linear_bias_gelu_f16(x, w, y, b, M, N, K, threads=[((N + 127) // 128) * 128, (M + 63) // 64],
                               group_size=[128, 1])
    return y


def resid(x, h, gate):
    if PKG is not None:
        return PKG.gated_residual(x, h, gate)
    out = torch.empty_like(x)
    C = x.shape[-1]
    lib().exact_resid(x, h, gate, out, C, threads=x.numel(), group_size=256)
    return out


def resid_layernorm(x, h=None, weight=None, bias=None, eps=1e-6):
    """(x + h or x, LayerNorm32 half, plain or affine)."""
    if PKG is not None:
        return PKG.resid_layernorm(x, h=h, weight=weight, bias=bias, eps=eps)
    L, C = x.shape
    xout = torch.empty_like(x) if h is not None else x
    y = torch.empty_like(x)
    mode = (8 if h is not None else 0) | (4 if weight is not None else 0)
    one, zero = _consts(C, x.device)
    rw, rb = (weight, bias) if weight is not None else (_dummy(x.device, torch.float32),) * 2
    lib().fused_resid_ln_mod_repo(x, h if h is not None else x, one, xout, one, zero, y, rw, rb,
                                  _dummy(x.device, torch.float32), C, eps, 1e-5, mode,
                                  threads=L * (C // 4), group_size=C // 4)
    return xout, y


_CONSTS = {}


def _consts(C, dev):
    if (C, dev) not in _CONSTS:
        _CONSTS[(C, dev)] = (torch.ones(C, device=dev, dtype=torch.float16),
                             torch.zeros(C, device=dev, dtype=torch.float16))
    return _CONSTS[(C, dev)]


def rmsnorm_heads(src, col_offset, gamma, scale):
    """src [R, W] half contiguous -> F.normalize(src[:, col_offset : col_offset + H*64] as [R, H, 64]) * gamma *
    scale, half [R, H, 64]."""
    if PKG is not None:
        return PKG.rmsnorm_heads(src, gamma, scale, col_offset=col_offset)
    R, W = src.shape
    H = gamma.shape[0]
    o = torch.empty(R, H, 64, device=src.device, dtype=torch.float16)
    lib().rmsnorm_heads_f16(src.view(-1)[col_offset:], gamma, o, R, H, W, scale,
                            threads=R * H * 4, group_size=256)
    return o


def add(x, h):
    """x + h (half)."""
    one, _ = _consts(x.shape[-1], x.device)
    return resid(x, h, one)


# -- denoiser blocks ----------------------------------------------------------------------------------------------

class KBlock:
    """Parameters of one modulated UnifiedTransformerBlock and its RePo layer, in kernel-ready form."""

    def __init__(self, blk, repo):
        assert blk.modulation
        a = blk.attn
        self.table = blk.shift_table.detach()                     # fp32 [1, 6C]
        self.rn = (repo.norm.weight.detach(), repo.norm.bias.detach(), repo.norm.eps)
        self.wg = repo.gate_map.weight.detach()
        self.wc = repo.content_map.weight.detach()
        self.wf = repo.final_map.weight.detach()
        freqs = (repo.freqs_0, repo.freqs_1, repo.freqs_2)
        with torch.no_grad():
            self.ft = torch.cat([f.tanh() for f in freqs]).contiguous()          # same ops as clamp_mul
            self.fd = torch.cat([f - f.tanh() for f in freqs]).contiguous()
        self.axis = torch.tensor(sum(([i] * f.numel() for i, f in enumerate(freqs)), []), dtype=torch.int32,
                                 device=self.ft.device)
        self.wqkv, self.bqkv = a.qkv.weight.detach(), a.qkv.bias.detach()
        self.qg, self.kg = a.q_norm.gamma.detach().contiguous(), a.k_norm.gamma.detach().contiguous()
        self.scale = float(a.q_norm.scale)
        assert a.q_norm.scale == a.k_norm.scale and a.num_heads == 16 and a.head_dim == 64
        self.wo, self.bo = a.out.weight.detach(), a.out.bias.detach()
        self.w1, self.b1 = blk.mlp.mlp[0].weight.detach(), blk.mlp.mlp[0].bias.detach()
        self.w2, self.b2 = blk.mlp.mlp[2].weight.detach(), blk.mlp.mlp[2].bias.detach()
        self.eps1, self.eps2 = blk.norm1.eps, blk.norm2.eps
        assert self.eps1 == self.eps2


def prepare_denoiser(m):
    return {'noise': [KBlock(b, r) for b, r in zip(m.noise_refiner, m.noise_repo_layers)],
            'joint': [KBlock(b, r) for b, r in zip(m.blocks, m.repo_layers)]}


def _mods(blocks, t_mod):
    """Per block: mod = t_mod + shift_table.type(half); chunks (shift_msa, 1+scale_msa, gate_msa, shift_mlp,
    1+scale_mlp, gate_mlp), with the reference's half arithmetic."""
    out = []
    for b in blocks:
        mod = t_mod + b.table.type(t_mod.dtype)
        c = mod.chunk(6, dim=1)
        out.append((c[0][0], (1 + c[1])[0], c[2][0], c[3][0], (1 + c[4])[0], c[5][0]))
    return out


def run_blocks(blocks, h, t_mod, attn_bias=None):
    """h [1, L, C] half -> [1, L, C] half through the given modulated blocks."""
    B, L, C = h.shape
    assert B == 1
    x = h[0].contiguous()
    mods = _mods(blocks, t_mod)
    pend = None
    for b, (sh1, sc1, g1, sh2, sc2, g2) in zip(blocks, mods):
        if pend is None:
            x, y, r = resid_ln_mod_repo(x, sc1, sh1, repo=b.rn, eps_ln=b.eps1)
        else:
            x, y, r = resid_ln_mod_repo(pend[0], sc1, sh1, h=pend[1], gate=pend[2], repo=b.rn, eps_ln=b.eps1)
        a = silu_mul(F.linear(r, b.wg), F.linear(r, b.wc))
        delta = F.linear(a, b.wf)
        qkv = F.linear(y, b.wqkv, b.bqkv)
        q, k = repo_rope_qk(qkv, delta, b.ft, b.fd, b.axis, b.qg, b.kg, b.scale)
        v = qkv.view(1, L, 3, 16, 64)[:, :, 2]
        o = F.scaled_dot_product_attention(q.view(1, L, 16, 64).permute(0, 2, 1, 3), k.view(1, L, 16, 64).permute(0, 2, 1, 3),
                                           v.permute(0, 2, 1, 3), attn_mask=attn_bias)
        ha = F.linear(o.permute(0, 2, 1, 3).reshape(L, C), b.wo, b.bo)
        x1, y2, _ = resid_ln_mod_repo(x, sc2, sh2, h=ha, gate=g1, eps_ln=b.eps2)
        hm = F.linear(linear_gelu(y2, b.w1, b.b1) if FUSE_GELU else gelu_(F.linear(y2, b.w1, b.b1)), b.w2, b.b2)
        pend = (x1, hm, g2)
    return resid(pend[0], pend[1], pend[2])[None]
