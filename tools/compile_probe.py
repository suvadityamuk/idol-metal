"""Apply the reference's compile_for_sampling (torch.compile of UnifiedTransformerBlock / RePo forwards, real-valued
RoPE) to the denoiser on MPS and time one forward. Records success or the exact failure."""
import os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import sync
from profile_dit import load_denoiser, inputs
import torch
from idol.pipeline import compile_for_sampling
m = load_denoiser()
x, t, cond = inputs('dumps/ref_dev_A.safetensors')
with torch.no_grad():
    ref = m(x, t, cond)['latent'].float()
    compile_for_sampling(m)
    try:
        t0 = time.perf_counter(); out = m(x, t, cond); sync(); print(f'first compiled call {time.perf_counter()-t0:.1f}s')
    except Exception:
        traceback.print_exc(); sys.exit(1)
    ts = []
    for _ in range(3):
        sync(); t0 = time.perf_counter(); out = m(x, t, cond); sync(); ts.append(time.perf_counter() - t0)
    print('compiled forward:', [f'{v*1e3:.0f} ms' for v in ts])
    d = (out['latent'].float() - ref).abs()
    print(f'vs eager: max {d.max():.3e} mean {d.mean():.3e} ref absmax {ref.abs().max():.3e}')
