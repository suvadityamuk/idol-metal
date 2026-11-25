"""Per-op times of one fused-kernel block at L (sync around each op, 20 reps)."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import torch, torch.nn.functional as F
from profile_dit import load_denoiser
from fast import ops
L = int(sys.argv[1]) if len(sys.argv) > 1 else 12294
m = load_denoiser(); b = ops.KBlock(m.blocks[3], m.repo_layers[3])
x = (torch.randn(L, 1024, device='mps') * 3).half()
t_mod = torch.randn(1, 6144, device='mps').half()
(sh1, sc1, g1, sh2, sc2, g2), = ops._mods([b], t_mod)
T = {}
def tm(name, fn, n=20):
    r = fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize(); T[name] = (time.perf_counter() - t) / n; return r
_, y, r = tm('K_A resid+ln+mod+repoLN', lambda: ops.resid_ln_mod_repo(x, sc1, sh1, h=x, gate=g1, repo=b.rn))
g = tm('repo gate GEMM f32', lambda: F.linear(r, b.wg)); c = tm('repo content GEMM f32', lambda: F.linear(r, b.wc))
a = tm('silu*mul', lambda: ops.silu_mul(g, c)); d = tm('repo final GEMM f32', lambda: F.linear(a, b.wf))
qkv = tm('qkv GEMM', lambda: F.linear(y, b.wqkv, b.bqkv))
q, k = tm('K_B repo+rope+rms qk', lambda: ops.repo_rope_qk(qkv, d, b.ft, b.fd, b.axis, b.qg, b.kg, b.scale))
v = qkv.view(1, L, 3, 16, 64)[:, :, 2]
o = tm('sdpa', lambda: F.scaled_dot_product_attention(q.view(1, L, 16, 64).permute(0, 2, 1, 3), k.view(1, L, 16, 64).permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)))
o2 = tm('permute copy', lambda: o.permute(0, 2, 1, 3).reshape(L, 1024).contiguous())
ha = tm('out GEMM', lambda: F.linear(o2, b.wo, b.bo))
x1, y2, _ = tm('K_A resid+ln+mod', lambda: ops.resid_ln_mod_repo(x, sc2, sh2, h=ha, gate=g1))
f1 = tm('fc1 GEMM', lambda: F.linear(y2, b.w1, b.b1)); tm('gelu', lambda: ops.gelu_(f1))
tm('fc2 GEMM', lambda: F.linear(f1, b.w2, b.b2))
tot = sum(T.values())
for k_, v_ in T.items(): print(f'{k_:28s} {v_*1e3:7.3f} ms {100*v_/tot:5.1f}%')
print(f'block total {tot*1e3:.2f} ms')
