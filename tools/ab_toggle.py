"""In-process A/B of one fast-build switch: alternate a module attribute between two values, time full steps
(both CFG passes, back to back, sync at the ends) and byte-compare the velocities of the two arms.

python tools/ab_toggle.py fast.ops FUSE_GELU False True [--rounds 4 --steps 3]
"""
import argparse, importlib, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tensordict import TensorDict
from common import DumpReader, write_json
from profile_dit import load_denoiser
from fast.dit import FastDenoiser


def main():
    p = argparse.ArgumentParser()
    p.add_argument('module'); p.add_argument('attr'); p.add_argument('a'); p.add_argument('b')
    p.add_argument('--rounds', type=int, default=4); p.add_argument('--steps', type=int, default=3)
    p.add_argument('--dump', default='dumps/ref_dev_A.safetensors'); p.add_argument('--json')
    a = p.parse_args()
    mod = importlib.import_module(a.module)
    vals = {'A': eval(a.a), 'B': eval(a.b)}
    r = DumpReader(a.dump); dev = 'mps'
    m = load_denoiser(dev)
    cond = TensorDict({'feature1': r.get('encode/feature1', dev), 'feature2': r.get('encode/feature2', dev)}, batch_size=1)
    fd = FastDenoiser(m, impl='kernels'); fd.prepare(cond, cond.apply(torch.zeros_like))
    x = {'latent': r.get('dit/call000/in/arg0.latent', dev), 'camera': r.get('dit/call000/in/arg0.camera', dev)}
    t = r.get('dit/call000/in/arg1', dev)
    times = {'A': [], 'B': []}; outs = {}
    for rnd in range(a.rounds + 1):
        for arm in (('A', 'B') if rnd % 2 == 0 else ('B', 'A')):
            setattr(mod, a.attr, vals[arm])
            torch.mps.synchronize(); t0 = time.perf_counter()
            for _ in range(a.steps):
                vc, vu = fd.velocities(x, t)
            torch.mps.synchronize(); dt = (time.perf_counter() - t0) / a.steps
            if rnd > 0: times[arm].append(dt)
            outs[arm] = (vc['latent'].float().cpu(), vu['latent'].float().cpu())
    med = lambda v: sorted(v)[len(v) // 2]
    res = {'attr': f'{a.module}.{a.attr}', 'A': a.a, 'B': a.b, 'step_ms_A': [v * 1e3 for v in times['A']],
           'step_ms_B': [v * 1e3 for v in times['B']]}
    for i, n in enumerate(('v_cond', 'v_uncond')):
        d = (outs['A'][i] - outs['B'][i]).abs()
        res[n] = {'bit_equal': bool(torch.equal(outs['A'][i], outs['B'][i])), 'max_abs': float(d.max()),
                  'rel_l2': float((outs['A'][i] - outs['B'][i]).norm() / outs['A'][i].norm())}
    print(f"{a.attr}: A={a.a} {med(times['A'])*1e3:.0f} ms/step (min {min(times['A'])*1e3:.0f}); "
          f"B={a.b} {med(times['B'])*1e3:.0f} ms/step (min {min(times['B'])*1e3:.0f}); "
          f"B/A speed-up {med(times['A'])/med(times['B']):.3f}x; outputs: {res['v_cond']} {res['v_uncond']}")
    if a.json: write_json(a.json, res)


if __name__ == '__main__':
    main()
