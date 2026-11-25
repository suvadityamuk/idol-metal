"""Reference denoiser: batch-2 (cond, uncond) forward vs two batch-1 forwards, eager, step 0 inputs."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
from profile_dit import load_denoiser, inputs
import torch
m = load_denoiser()
x, t, cond = inputs('dumps/ref_dev_A.safetensors')
neg = cond.apply(torch.zeros_like)
with torch.no_grad():
    vc, vu = m(x, t, cond)['latent'], m(x, t, neg)['latent']
    both = m(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cond, neg]))['latent']
for n, a, b in (('cond', both[:1], vc), ('uncond', both[1:], vu)):
    d = (a.float() - b.float()).abs(); print(n, 'batch2 vs batch1: max', d.max().item(), 'rel', ((a.float()-b.float()).norm()/b.float().norm()).item())
