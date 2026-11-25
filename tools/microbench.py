"""Per-op timing at the denoiser's shapes (batch 1, joint sequence L = 8192 + 4101 + 1) on MPS."""
import time, sys
import torch, torch.nn.functional as F
dev = 'mps'
L = int(sys.argv[1]) if len(sys.argv) > 1 else 12294
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
C = 1024
def bench(name, fn, n=10):
    fn(); torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize()
    dt = (time.perf_counter() - t) / n
    print(f'{name:40s} {dt*1e3:8.2f} ms', flush=True)
    return dt
x = torch.randn(B, L, C, device=dev, dtype=torch.float16)
W = {n: torch.randn(o, i, device=dev, dtype=torch.float16) * 0.02 for n, (o, i) in
     dict(qkv=(3*C, C), out=(C, C), fc1=(4*C, C), fc2=(C, 4*C)).items()}
bb = {n: torch.zeros(w.shape[0], device=dev, dtype=torch.float16) for n, w in W.items()}
h4 = torch.randn(B, L, 4*C, device=dev, dtype=torch.float16)
tot = 0
for n in W:
    inp = h4 if n == 'fc2' else x
    dt = bench(f'linear {n}', lambda: F.linear(inp, W[n], bb[n]))
    fl = 2 * B * L * W[n].numel()
    print(f'   {fl/dt/1e12:.1f} TFLOP/s'); tot += dt
q = torch.randn(B, 16, L, 64, device=dev, dtype=torch.float16)
k, v = torch.randn_like(q), torch.randn_like(q)
dt = bench('sdpa fp16', lambda: F.scaled_dot_product_attention(q, k, v)); tot += dt
print(f'   {4*B*16*L*L*64/dt/1e12:.1f} TFLOP/s')
dt = bench('layernorm32 (x.float()->ln->half)', lambda: F.layer_norm(x.float(), (C,), eps=1e-6).half()); tot += 2*dt
dt = bench('gelu tanh 4C', lambda: F.gelu(h4, approximate='tanh')); tot += dt
print(f'sum of main ops per block ~ {tot*1e3:.1f} ms; x24 blocks = {tot*24*1e3:.0f} ms')
