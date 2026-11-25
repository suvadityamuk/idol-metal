"""Teacher-forced A/B of the restructured denoiser against a reference dump: at each recorded step feed the
reference's own sample and time, compare v_cond, v_uncond and the guided velocity with the reference's, and time
one step (both passes) with device sync. Also times the reference denoiser (two eager passes) in the same process,
interleaved, as the speed baseline.

python tools/ab_dit.py --impl torch|kernels [--steps 2] [--dump dumps/ref_dev_A.safetensors] [--no-ref-timing]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DumpReader, sync, write_json  # noqa: E402
from profile_dit import load_denoiser  # noqa: E402

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from fast.dit import FastDenoiser  # noqa: E402


def err(a, b):
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return {'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
            'rel_l2': float((a - b).norm() / b.norm()), 'bit_equal': bool(torch.equal(a, b))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--impl', default='torch')
    p.add_argument('--dump', default='dumps/ref_dev_A.safetensors')
    p.add_argument('--steps', type=int, default=2, help='recorded steps to check (from step 0)')
    p.add_argument('--no-collapse', action='store_true')
    p.add_argument('--ref-timing', type=int, default=1, help='also time the reference denoiser (0/1)')
    p.add_argument('--reps', type=int, default=2, help='timed repetitions per step, interleaved')
    p.add_argument('--compile', action='store_true', help='torch.compile the fast blocks (Inductor)')
    p.add_argument('--json')
    a = p.parse_args()

    dev = 'mps'
    r = DumpReader(a.dump)
    m = load_denoiser(dev)
    cfg = 3.0
    cond = TensorDict({'feature1': r.get('encode/feature1', dev), 'feature2': r.get('encode/feature2', dev)},
                      batch_size=1)
    neg = cond.apply(torch.zeros_like)
    fd = FastDenoiser(m, impl=a.impl, collapse_uncond=not a.no_collapse)
    if a.compile:
        import fast.dit as fdit
        fdit.block_torch = torch.compile(fdit.block_torch, dynamic=False)
    sync()
    t0 = time.perf_counter()
    fd.prepare(cond, neg)
    sync()
    prep_s = time.perf_counter() - t0
    results = []
    ft, rt = [], []
    for s in range(a.steps):
        ci, ui = 2 * s, 2 * s + 1
        x = {'latent': r.get(f'dit/call{ci:03d}/in/arg0.latent', dev), 'camera': r.get(f'dit/call{ci:03d}/in/arg0.camera', dev)}
        t = r.get(f'dit/call{ci:03d}/in/arg1', dev)
        ref_c = {k: r.get(f'dit/call{ci:03d}/out.{k}') for k in ('latent', 'camera')}
        ref_u = {k: r.get(f'dit/call{ui:03d}/out.{k}') for k in ('latent', 'camera')}
        for rep in range(a.reps + 1):
            order = ('fast', 'ref') if rep % 2 == 0 else ('ref', 'fast')
            for which in order:
                if which == 'ref' and not a.ref_timing:
                    continue
                sync()
                t1 = time.perf_counter()
                if which == 'fast':
                    vc, vu = fd.velocities(x, t)
                else:
                    with torch.no_grad():
                        xt = TensorDict(x, batch_size=1)
                        m(xt, t, cond)
                        m(xt, t, neg)
                sync()
                dt = time.perf_counter() - t1
                if rep > 0:
                    (ft if which == 'fast' else rt).append(dt)
        row = {'step': s}
        for k in ('latent', 'camera'):
            row[f'v_cond.{k}'] = err(vc[k].cpu(), ref_c[k])
            row[f'v_uncond.{k}'] = err(vu[k].cpu(), ref_u[k])
            g = (1 + cfg) * vc[k].cpu() - cfg * vu[k].cpu()
            gr = (1 + cfg) * ref_c[k] - cfg * ref_u[k]
            row[f'v_guided.{k}'] = err(g, gr)
        results.append(row)
        print(f"step {s}: " + '  '.join(f"{n}: max {e['max_abs']:.2e} rel {e['rel_l2']:.2e}{' (bit-equal)' if e['bit_equal'] else ''}"
                                         for n, e in row.items() if n != 'step' and n.endswith('latent')), flush=True)
    med = lambda v: sorted(v)[len(v) // 2] if v else float('nan')
    print(f'prepare (once per request): {prep_s*1e3:.0f} ms')
    print(f'step time (cond+uncond): fast {med(ft)*1e3:.0f} ms (min {min(ft)*1e3:.0f})' +
          (f', reference {med(rt)*1e3:.0f} ms (min {min(rt)*1e3:.0f}); speed-up {med(rt)/med(ft):.2f}x' if rt else ''))
    if a.json:
        write_json(a.json, {'args': vars(a), 'prepare_s': prep_s, 'fast_step_s': ft, 'ref_step_s': rt,
                            'errors': results})


if __name__ == '__main__':
    main()
