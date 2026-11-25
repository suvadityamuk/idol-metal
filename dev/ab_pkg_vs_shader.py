import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from pathlib import Path
import torch, kernels
from tensordict import TensorDict
from profile_dit import load_denoiser
from common import DumpReader
from fast import ops
from fast.dit import FastDenoiser
pkg = kernels.get_local_kernel(Path('kernel'), trust_remote_code=True)
r = DumpReader('dumps/ref_dev_A.safetensors'); dev = 'mps'
m = load_denoiser()
cond = TensorDict({'feature1': r.get('encode/feature1', dev), 'feature2': r.get('encode/feature2', dev)}, batch_size=1)
fd = FastDenoiser(m, impl='kernels'); fd.prepare(cond, cond.apply(torch.zeros_like))
x = {'latent': r.get('dit/call000/in/arg0.latent', dev), 'camera': r.get('dit/call000/in/arg0.camera', dev)}
t = r.get('dit/call000/in/arg1', dev)
T = {'shader': [], 'package': []}; out = {}
for i in range(5):
    for arm in (('shader', 'package') if i % 2 == 0 else ('package', 'shader')):
        ops.use_package(pkg if arm == 'package' else None)
        torch.mps.synchronize(); t0 = time.perf_counter()
        for _ in range(3): vc, vu = fd.velocities(x, t)
        torch.mps.synchronize(); dt = (time.perf_counter() - t0) / 3
        if i: T[arm].append(dt)
        out[arm] = (vc['latent'].cpu(), vu['latent'].cpu())
med = lambda v: sorted(v)[len(v) // 2]
print({k: f'{med(v)*1e3:.0f} ms (min {min(v)*1e3:.0f})' for k, v in T.items()}, 'outputs equal:', all(torch.equal(a, b) for a, b in zip(out['shader'], out['package'])))
