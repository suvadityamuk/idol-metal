"""Run benchmarks/benchmark.py against a local kernel-builder build (what `kernels benchmark <repo>` does for a
Hub repo): per workload, time the kernel and the PyTorch op chain it replaces (verify_*), and check the outputs
with the runner's own criterion (torch.allclose, atol 1e-2) plus exact equality.

python benchmarks/run_local.py [build root, default: this project] [--iterations 50]"""
import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch
import kernels

p = argparse.ArgumentParser()
p.add_argument('root', nargs='?', default=str(Path(__file__).resolve().parents[1]))
p.add_argument('--iterations', type=int, default=50)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('bench', Path(__file__).with_name('benchmark.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
k = kernels.get_local_kernel(Path(a.root), trust_remote_code=True)


def timeit(fn, n):
    fn(); torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1e3


b = mod.IdolMetalBenchmark()
b.kernel, b.device = k, 'mps'
names = [n[len('benchmark_'):] for n in dir(b) if n.startswith('benchmark_')]
print(f"{'workload':24s} {'kernel ms':>10s} {'torch ms':>10s} {'speed-up':>9s}  allclose  exact")
for n in names:
    torch.manual_seed(b.seed or 0)
    getattr(b, 'setup_' + n)()
    tk = timeit(getattr(b, 'benchmark_' + n), a.iterations)
    tr = timeit(getattr(b, 'verify_' + n), a.iterations)
    ref = getattr(b, 'verify_' + n)()
    getattr(b, 'benchmark_' + n)()
    out = b.out
    print(f"{n:24s} {tk:10.3f} {tr:10.3f} {tr / tk:8.2f}x  {str(torch.allclose(out.float(), ref.float(), atol=1e-2)):8s}  "
          f"{torch.equal(out, ref)}")
