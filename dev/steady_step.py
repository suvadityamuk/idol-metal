import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import torch
from tensordict import TensorDict
from profile_dit import load_denoiser
from common import DumpReader
from fast.dit import FastDenoiser
r = DumpReader('dumps/ref_dev_A.safetensors'); dev = 'mps'
m = load_denoiser()
cond = TensorDict({'feature1': r.get('encode/feature1', dev), 'feature2': r.get('encode/feature2', dev)}, batch_size=1)
fd = FastDenoiser(m, impl=os.environ.get('IMPL', 'kernels')); fd.prepare(cond, cond.apply(torch.zeros_like))
x = {'latent': r.get('dit/call000/in/arg0.latent', dev), 'camera': r.get('dit/call000/in/arg0.camera', dev)}
t = r.get('dit/call000/in/arg1', dev)
fd.velocities(x, t); torch.mps.synchronize()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
t0 = time.perf_counter()
for _ in range(n): vc, vu = fd.velocities(x, t)
torch.mps.synchronize(); print(f'steady step {(time.perf_counter()-t0)/n*1e3:.0f} ms over {n} steps')
