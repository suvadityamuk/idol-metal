"""Gaussian decoder (GaussianFixedlenDecoder) forward with the fused kernels: same arithmetic as the reference
(LayerNorm32 / RMSNorm rounding points kept, MPS GEMMs and SDPA), fewer passes over memory.

TransformerCrossBlock: x += self_attn(norm1(x)); x += cross_attn(norm2(x), ctx); x += mlp(norm3(x)).
"""
import torch
import torch.nn.functional as F

from . import ops


class CrossBlockParams:
    def __init__(self, blk):
        sa, ca = blk.self_attn, blk.cross_attn
        assert sa.num_heads == 16 and ca.num_heads == 16
        self.eps = blk.norm1.eps
        assert blk.norm2.eps == self.eps and blk.norm3.eps == self.eps
        self.n2w, self.n2b = blk.norm2.weight.detach(), blk.norm2.bias.detach()
        self.wqkv, self.bqkv = sa.to_qkv.weight.detach(), sa.to_qkv.bias.detach()
        self.sqg, self.skg = sa.q_rms_norm.gamma.detach().contiguous(), sa.k_rms_norm.gamma.detach().contiguous()
        self.so_w, self.so_b = sa.to_out.weight.detach(), sa.to_out.bias.detach()
        self.wq, self.bq = ca.to_q.weight.detach(), ca.to_q.bias.detach()
        self.wkv, self.bkv = ca.to_kv.weight.detach(), ca.to_kv.bias.detach()
        self.cqg, self.ckg = ca.q_rms_norm.gamma.detach().contiguous(), ca.k_rms_norm.gamma.detach().contiguous()
        self.co_w, self.co_b = ca.to_out.weight.detach(), ca.to_out.bias.detach()
        self.w1, self.b1 = blk.mlp.mlp[0].weight.detach(), blk.mlp.mlp[0].bias.detach()
        self.w2, self.b2 = blk.mlp.mlp[2].weight.detach(), blk.mlp.mlp[2].bias.detach()
        self.scale = float(sa.q_rms_norm.scale)


def _ln(x, p, h=None, affine=False):
    """(x', half LN32(x')) with x' = x + h when h is given; affine uses norm2's weight/bias."""
    return ops.resid_layernorm(x, h=h, weight=p.n2w if affine else None, bias=p.n2b if affine else None, eps=p.eps)


def _rms(src, col_offset, gamma, scale):
    return ops.rmsnorm_heads(src, col_offset, gamma, scale)


def _sdpa(q, k, v):
    """q [Lq, H, D], k/v [Lk, H, D] (views allowed) -> [Lq, H * D]."""
    o = F.scaled_dot_product_attention(q[None].permute(0, 2, 1, 3), k[None].permute(0, 2, 1, 3),
                                       v[None].permute(0, 2, 1, 3))
    return o[0].permute(1, 0, 2).reshape(q.shape[0], -1)


def run_cross_blocks(params, x, ctx):
    """x [L, C] half, ctx [Lk, Cc] half -> [L, C] half."""
    L, C = x.shape
    Lk = ctx.shape[0]
    pend = None
    for p in params:
        if pend is None:
            _, y = _ln(x, p)
        else:
            x, y = _ln(x, p, h=pend)
        qkv = F.linear(y, p.wqkv, p.bqkv)                                  # [L, 3C]
        q = _rms(qkv, 0, p.sqg, p.scale)
        k = _rms(qkv, C, p.skg, p.scale)
        v = qkv.view(L, 3, 16, 64)[:, 2]
        a = F.linear(_sdpa(q, k, v), p.so_w, p.so_b)
        x, y = _ln(x, p, h=a, affine=True)
        q2 = _rms(F.linear(y, p.wq, p.bq), 0, p.cqg, p.scale)
        kv = F.linear(ctx, p.wkv, p.bkv)                                   # [Lk, 2C]
        k2 = _rms(kv, 0, p.ckg, p.scale)
        v2 = kv.view(Lk, 2, 16, 64)[:, 1]
        a2 = F.linear(_sdpa(q2, k2, v2), p.co_w, p.co_b)
        x, y = _ln(x, p, h=a2)
        pend = F.linear(ops.linear_gelu(y, p.w1, p.b1), p.w2, p.b2)
    return ops.add(x, pend)


class FastGaussianDecoder:
    def __init__(self, dec):
        self.d = dec
        self.params = [CrossBlockParams(b) for b in dec.blocks]

    @torch.no_grad()
    def forward(self, points, cond):
        d = self.d
        B, L, Cin = points.shape
        assert B == 1
        h = d.in_proj(points) + d.pos_embedder(points.view(-1, Cin)).view(B, L, -1)
        h = d.input_layer(h)
        out = run_cross_blocks(self.params, h.type(d.dtype)[0].contiguous(), cond.type(d.dtype)[0].contiguous())
        out = out[None].type(h.dtype).type(cond.dtype)
        return d.out_proj(F.layer_norm(out, out.shape[-1:]))
