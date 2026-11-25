"""Reference denoiser, unconditional pass (cond features zeroed): are the 4101 context rows identical in every block?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from profile_dit import load_denoiser, inputs
import torch, math, torch.nn.functional as F
m = load_denoiser()
x, t, cond = inputs('dumps/ref_dev_A.safetensors', call=1)
neg = cond.apply(torch.zeros_like)
res = []
def hook(mod, a, out):
    c = out[0, 8192:8192 + 4101]
    res.append(bool((c == c[:1]).all()))
for b in m.blocks: b.register_forward_hook(hook)
with torch.no_grad(): m(x, t, neg)
print('blocks with all-identical context rows:', sum(res), 'of', len(res))
q = torch.randn(1, 16, 512, 64, device='mps', dtype=torch.float16); k, v = torch.randn_like(q), torch.randn_like(q)
for dt in (torch.float32, torch.float16):
    try:
        b = torch.zeros(1, 1, 1, 512, device='mps', dtype=dt); b[..., 3] = math.log(4101)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=b); torch.mps.synchronize()
        # reference: replicate key 3 4101 times
        kk = torch.cat([k, k[:, :, 3:4].expand(1, 16, 4100, 64)], 2); vv = torch.cat([v, v[:, :, 3:4].expand(1, 16, 4100, 64)], 2)
        r = F.scaled_dot_product_attention(q.float(), kk.float(), vv.float())
        print(dt, 'mask ok; max err vs replicated fp32', (o.float() - r).abs().max().item())
    except Exception as e:
        print(dt, 'mask failed:', str(e)[:200])
