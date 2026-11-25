"""Section profile of one reference denoiser forward (batch 1) on MPS: synchronised timers around each module type
and around the functional ops (rope, sdpa). Sync per call inflates small ops; use for ranking only."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import StageTimer, sync
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from tensordict import TensorDict
import idol.models.layers as layers
from idol.models import LatentSeqMMFlowModel

def load_denoiser(dev='mps'):
    d = snapshot_download('speridlabs/idol', allow_patterns=['config.json', 'denoiser.safetensors'])
    cfg = json.load(open(os.path.join(d, 'config.json')))
    m = LatentSeqMMFlowModel(**cfg['denoiser']['args'])
    m.load_state_dict(load_file(os.path.join(d, 'denoiser.safetensors')), strict=True)
    return m.eval().requires_grad_(False).to(dev)

def inputs(dump, call=0, dev='mps'):
    from common import DumpReader
    r = DumpReader(dump)
    g = lambda k: r.get(k, dev)
    x = TensorDict({'latent': g(f'dit/call{call:03d}/in/arg0.latent'), 'camera': g(f'dit/call{call:03d}/in/arg0.camera')}, batch_size=1)
    t = g(f'dit/call{call:03d}/in/arg1')
    cond = TensorDict({'feature1': g('encode/feature1'), 'feature2': g('encode/feature2')}, batch_size=1)
    return x, t, cond

if __name__ == '__main__':
    m = load_denoiser()
    x, t, cond = inputs('dumps/ref_dev_A.safetensors')
    with torch.no_grad():
        m(x, t, cond); sync()
        ts = []
        for _ in range(3):
            sync(); t0 = time.perf_counter(); m(x, t, cond); sync(); ts.append(time.perf_counter() - t0)
        print('forward (no instrumentation):', [f'{v*1e3:.0f} ms' for v in ts])
        timer = StageTimer()
        from idol.models.layers import (RePo3DRotaryEmbedding, UnifiedTransformerBlock, FeedForwardNet,
                                        LayerNorm32, MultiHeadRMSNorm)
        import torch.nn as nn
        for mod in m.modules():
            for cls, name in ((RePo3DRotaryEmbedding, 'repo'), (FeedForwardNet, 'mlp'), (LayerNorm32, 'layernorm32'),
                              (MultiHeadRMSNorm, 'qk_rmsnorm')):
                if type(mod) is cls:
                    mod.forward = timer.wrap(name, mod.forward)
        for blk in m.modules():
            if isinstance(blk, UnifiedTransformerBlock):
                blk.attn.qkv.forward = timer.wrap('qkv_linear', blk.attn.qkv.forward)
                blk.attn.out.forward = timer.wrap('out_linear', blk.attn.out.forward)
        layers.apply_rotary_emb = timer.wrap('apply_rope', layers.apply_rotary_emb)
        layers.scaled_dot_product_attention = timer.wrap('sdpa', layers.scaled_dot_product_attention)
        m.forward = timer.wrap('total', m.forward)
        m(x, t, cond)
        tot = timer.times['total']
        for k, v in sorted(timer.times.items(), key=lambda kv: -kv[1]):
            print(f'{k:14s} {v*1e3:8.1f} ms  {100*v/tot:5.1f}%')
