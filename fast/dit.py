"""Restructured Idol denoiser forward for sampling (same weights, same per-op arithmetic as the reference where the
computation is the same). Exact algebraic reuse, no approximation:

  * the context stream (cond_embedder + context refiner) does not depend on the step: computed once per request,
    for the condition and for the zeroed (unconditional) condition;
  * the noise refiner, camera refiner and time embedding do not depend on the condition: computed once per step and
    shared by the conditional and unconditional passes;
  * with the condition zeroed, all L_c context tokens are identical (checked: every row of every block equal in the
    reference), so the unconditional pass carries ONE context token and attention gives its key the weight L_c
    (an additive fp32 log(L_c) bias on that key's logit). Same mathematical result, ~1/3 fewer tokens.

The block math is pluggable (`impl`):
  'torch'   reference ops (bit-compatible with the reference where shapes are the same)
  'kernels' fused Metal kernels (fast/ops.py) around MPS GEMMs and SDPA
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from idol.models.layers import apply_rotary_emb, clamp_mul


# -- reference-op block (torch) ---------------------------------------------------------------------------------

def repo_torch(repo, hidden_states):
    """RePo3DRotaryEmbedding.forward (complex-valued, the eager reference form)."""
    return repo(hidden_states)


def block_torch(blk, x, mod, rope, attn_bias=None):
    """UnifiedTransformerBlock.forward (modulated) with an optional additive attention bias on keys."""
    mod = mod + blk.shift_table.type(mod.dtype)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
    h = blk.norm1(x)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    h = attn_torch(blk.attn, h, rope, attn_bias)
    x = x + h * gate_msa.unsqueeze(1)
    h = blk.norm2(x)
    h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    h = blk.mlp(h)
    return x + h * gate_mlp.unsqueeze(1)


def attn_torch(attn, x, rope, attn_bias=None):
    B, L, C = x.shape
    q, k, v = attn.qkv(x).reshape(B, L, 3, attn.num_heads, attn.head_dim).unbind(2)
    q = attn.q_norm(apply_rotary_emb(q, rope))
    k = attn.k_norm(apply_rotary_emb(k, rope))
    q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    return attn.out(o.permute(0, 2, 1, 3).reshape(B, L, C))


def split_log_weight(n, parts=2):
    """fp16 values b_1..b_k with sum(exp(b_i)) ~= n: greedy, each the largest fp16 not above log(remainder)."""
    vals, rem = [], float(n)
    for i in range(parts):
        if i == parts - 1:
            b = float(torch.tensor(math.log(rem)).half())
        else:
            b = torch.tensor(math.log(rem)).half()
            if float(b) > math.log(rem):
                b = torch.nextafter(b.float(), torch.tensor(-1e9)).half()   # step down one fp16 ulp
                while float(b) > math.log(rem):
                    b = torch.tensor(float(b) - 2 ** -8).half()
            b = float(b)
        vals.append(b)
        rem -= math.exp(b)
    return vals


# -- sampler ------------------------------------------------------------------------------------------------------

class FastDenoiser:
    def __init__(self, model, impl='torch', collapse_uncond=True):
        self.m = model
        self.impl = impl
        self.collapse = collapse_uncond
        self.dtype = model.dtype
        if impl == 'kernels':
            from . import ops
            self.ops = ops
            self.kblocks = ops.prepare_denoiser(model)
        self._pos = None

    # per-request constants ------------------------------------------------------------------------------------
    @torch.no_grad()
    def context(self, cond):
        """h_cond after the context refiner (the reference's own modules and ops)."""
        m = self.m
        h_cond = m.cond_embedder(cond['feature1']) + m.cond_embedder2(cond['feature2'])
        h_cond = h_cond.type(self.dtype)
        for repo, block in zip(m.context_repo_layers, m.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=repo(h_cond))
        return h_cond

    @torch.no_grad()
    def prepare(self, cond, neg_cond):
        m = self.m
        self.h_cond = self.context(cond)
        h_neg = self.context(neg_cond)
        if self.collapse:
            assert bool((h_neg == h_neg[:, :1]).all()), 'unconditional context rows are not identical'
            self.n_ctx = h_neg.shape[1]
            self.bias_vals = split_log_weight(self.n_ctx)
            self.h_neg = h_neg[:, :1].expand(-1, len(self.bias_vals), -1).contiguous()
        else:
            self.n_ctx = None
            self.h_neg = h_neg
        self.pos = m.pos_embedder(m.pos_pe).type(self.dtype)   # constant: the latent token positions

    def _bias(self, L, n_x, dev):
        """fp16 additive key bias on the collapsed unconditional context keys (fp32 masks take a 20x slower SDPA
        path on MPS): exp(b_1) + exp(b_2) = L_c to ~1e-5 relative, with b_i exactly representable in fp16."""
        b = torch.zeros(1, 1, 1, L, device=dev, dtype=torch.float16)
        for i, v in enumerate(self.bias_vals):
            b[..., n_x + i] = v
        return b

    # one step: (v_cond, v_uncond) -----------------------------------------------------------------------------
    @torch.no_grad()
    def velocities(self, x, t):
        m = self.m
        z, cam = x['latent'], x['camera']
        t_emb = m.t_embedder(t)
        t_mod = m.adaLN_modulation(t_emb).type(self.dtype)
        h_x = m.input_layer(z).type(self.dtype)
        h_x = h_x + self.pos
        if self.impl == 'kernels':
            h_x = self.ops.run_blocks(self.kblocks['noise'], h_x, t_mod)
        else:
            for repo, block in zip(m.noise_repo_layers, m.noise_refiner):
                h_x = block_torch(block, h_x, t_mod, repo_torch(repo, h_x))
        h_cam = m.cam_refiner(cam).type(self.dtype)
        outs = []
        for h_ctx, collapsed in ((self.h_cond, False), (self.h_neg, self.collapse)):
            h = torch.cat([h_x, h_ctx, h_cam], dim=1)
            bias = self._bias(h.shape[1], h_x.shape[1], h.device) if collapsed else None
            if self.impl == 'kernels':
                h = self.ops.run_blocks(self.kblocks['joint'], h, t_mod, attn_bias=bias)
            else:
                for repo, block in zip(m.repo_layers, m.blocks):
                    h = block_torch(block, h, t_mod, repo_torch(repo, h), bias)
            outs.append(self._head(h, z, cam, t_emb))
        return outs

    def _head(self, h, z, cam, t_emb):
        m = self.m
        h_x = h[:, :z.shape[1]].type(z.dtype)
        h_x = F.layer_norm(h_x, h_x.shape[-1:])
        h_cam = h[:, -cam.shape[1]:].type(cam.dtype)
        h_cam = F.layer_norm(h_cam, h_cam.shape[-1:])
        shift, scale = (m.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        h_x = h_x * (1 + scale) + shift
        h_cam = h_cam * (1 + scale) + shift
        return {'latent': m.out_layer(h_x), 'camera': m.cam_out_layer(h_cam)}


@torch.no_grad()
def sample_flow_fast(fd, noise, cond, neg_cond, steps=50, rescale_t=3.0, cfg_strength=3.0, verbose=False,
                     batch_cfg=False, on_step=None):
    """sampler.sample_flow with the restructured denoiser (same time grid, same CFG/Euler arithmetic)."""
    assert cfg_strength > 0
    fd.prepare(cond, neg_cond)
    sample = {'latent': noise['latent'], 'camera': noise['camera']}
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    dev = sample['latent'].device
    for i in range(steps):
        t, t_prev = t_seq[i], t_seq[i + 1]
        tt = torch.tensor([1000 * t], device=dev, dtype=torch.float32)
        vc, vu = fd.velocities(sample, tt)
        if on_step is not None:
            on_step(i, sample, vc, vu)
        new = {}
        for k in ('latent', 'camera'):
            v = (1 + cfg_strength) * vc[k] - cfg_strength * vu[k]
            new[k] = sample[k] - v * (t - t_prev)
        sample = new
    from tensordict import TensorDict
    return TensorDict(sample, batch_size=1)
