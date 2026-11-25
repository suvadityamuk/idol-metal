"""One end-to-end Idol request on MPS with a chosen build, timed per stage (ref_run's timers and memory sampler),
saving an outputs-only dump for the closeness comparison.

  --variant ref       the official pipeline as shipped (eager on MPS; optimize=True is CUDA-only upstream)
  --variant compile   the official pipeline with its own optimize path forced on MPS: torch.compile of the
                      transformer blocks and RePo layers (real-valued RoPE) + batched CFG
  --variant fast      this port: restructured denoiser + fused Metal kernels (fast/), other stages as shipped
                      unless listed in --fast-stages

The same seed gives the same random draws (MPS generator, same call order), so no replay is needed on this device.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import IMAGES, Dump, MemorySampler, StageTimer, environment, footprint, sha256_file, sync, \
    write_json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import idol.pipeline as pipeline_mod  # noqa: E402
from idol import IdolPipeline  # noqa: E402

import ref_run  # noqa: E402


def make_pipeline(variant, fast_stages=()):
    pipe = IdolPipeline.from_pretrained('speridlabs/idol', device='mps')
    if variant == 'compile':
        pipe.optimize = True              # batch_cfg + compile path of the reference
        pipe.cudnn_attention = False
        pipeline_mod.compile_for_sampling(pipe.denoiser)
    elif variant in ('fast', 'parity'):
        from fast.pipeline import install
        install(pipe, list(fast_stages) + (['parity'] if variant == 'parity' else []))
    capture(pipe)
    return pipe


def capture(pipe):
    """Keep the final latent/camera and the decoded anchors of the last request in pipe._last (host copies)."""
    from idol.models.decoder import OctreeProbabilityFixedlenDecoder
    pipe._last = {}
    sf = pipeline_mod.sample_flow

    def sample_flow(*args, **kw):
        out = sf(*args, **kw)
        pipe._last['sample/final/latent'] = out['latent'].detach().cpu()
        pipe._last['sample/final/camera'] = out['camera'].detach().cpu()
        return out
    pipeline_mod.sample_flow = sample_flow
    oct_sample = OctreeProbabilityFixedlenDecoder.__dict__['sample'].__func__

    def octree(model, cond, num_points, level):
        out = oct_sample(model, cond, num_points, level)
        pipe._last['decode/anchors'] = out.detach().cpu()
        return out
    OctreeProbabilityFixedlenDecoder.sample = staticmethod(octree)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', default='ref', choices=['ref', 'compile', 'fast', 'parity'])
    p.add_argument('--fast-stages', default='', help='comma list of extra fast stages (see fast/pipeline.py)')
    p.add_argument('--image', default='robot')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=3.0)
    p.add_argument('--rescale_t', type=float, default=3.0)
    p.add_argument('--num_gaussians', type=int, default=262144)
    p.add_argument('--out', required=True)
    p.add_argument('--save', default=None, help='outputs-only dump (safetensors)')
    p.add_argument('--warmup', default='short', choices=['none', 'short'])
    p.add_argument('--kernels', default='package', choices=['package', 'shader'],
                   help='fast/parity: kernels from the local kernel-builder build (kernel/build) or compile_shader')
    a = p.parse_args()

    image, mode = IMAGES.get(a.image, (a.image, 'synthetic'))
    os.makedirs(a.out, exist_ok=True)
    mem = MemorySampler()
    t0 = time.perf_counter()
    if a.variant in ('fast', 'parity') and a.kernels == 'package':
        import kernels
        from pathlib import Path
        from fast import ops
        ops.use_package(kernels.get_local_kernel(Path(__file__).resolve().parents[1] / 'kernel', trust_remote_code=True))
    pipe = make_pipeline(a.variant, [s for s in a.fast_stages.split(',') if s])
    sync()
    load_s = time.perf_counter() - t0
    print(f'{a.variant}: loaded in {load_s:.1f}s', flush=True)
    warm_s = None
    if a.warmup == 'short':
        class W:
            seed, steps, cfg, rescale_t, num_gaussians = a.seed, 2, a.cfg, a.rescale_t, 32768
        t1 = time.perf_counter()
        ref_run.run_once(pipe, image, mode, W, StageTimer(), Dump(False))
        sync()
        warm_s = time.perf_counter() - t1
        print(f'warm-up {warm_s:.1f}s', flush=True)
    timer = StageTimer()
    torch.mps.empty_cache()
    mem.reset()
    output, _ = ref_run.run_once(pipe, image, mode, a, timer, Dump(False))
    peak = mem.peak()
    ply = os.path.join(a.out, 'run0.ply')
    with timer('save_ply'):
        output.gaussian.save_ply(ply)
    run = {'times_s': dict(timer.times), 'peak_memory': peak, 'ply_sha256': sha256_file(ply), 'camera': output.camera,
           'num_gaussians_out': int(output.gaussian.xyz.shape[0])}
    print(f'run: ' + ', '.join(f'{k}={v:.2f}s' for k, v in timer.times.items()) +
          f"; peak footprint {peak['footprint_peak_gb']:.1f} GB, mps driver {peak['mps_driver_peak_gb']:.1f} GB",
          flush=True)
    if a.save:
        d = Dump()
        g = output.gaussian
        for name in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
            d.put(f'output/{name}', getattr(g, name))
        for k, v in getattr(pipe, '_last', {}).items():
            d.put(k, v)
        d.put('preprocess/prepared_image', torch.from_numpy(np.array(output.preprocessed_image)))
        d.save(a.save, metadata={'variant': a.variant, 'image': image, 'steps': a.steps,
                                 'num_gaussians': a.num_gaussians, 'camera': output.camera})
    write_json(os.path.join(a.out, 'result.json'), {
        'args': vars(a), 'image': image, 'mode': mode, 'load_s': load_s, 'warmup_s': warm_s, 'run': run,
        'lifetime_max_footprint_gb': footprint()[1] / 2**30, 'env': environment()})
    mem.stop()


if __name__ == '__main__':
    main()
