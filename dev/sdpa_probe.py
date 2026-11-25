import time, math, torch, torch.nn.functional as F
def bench(name, fn, n=10):
    out = fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize(); dt = (time.perf_counter()-t)/n
    print(f'{name:50s} {dt*1e3:7.2f} ms'); return out
dev = 'mps'
for L in (12294, 8194):
    q = torch.randn(1, 16, L, 64, device=dev, dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    o1 = bench(f'L={L} contiguous BHLD', lambda: F.scaled_dot_product_attention(q, k, v))
    ql, kl, vl = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    o2 = bench(f'L={L} permuted views of BLHD', lambda: F.scaled_dot_product_attention(ql.transpose(1,2), kl.transpose(1,2), vl.transpose(1,2)))
    print('   same result:', torch.equal(o1, o2))
    bias = torch.zeros(1, 1, 1, L, device=dev, dtype=torch.float16); bias[..., -2] = math.log(4101)
    bench(f'L={L} float mask [1,1,1,L]', lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias))
    bias2 = bias.expand(1, 1, L, L)
    bench(f'L={L} float mask expanded [1,1,L,L]', lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias2))
