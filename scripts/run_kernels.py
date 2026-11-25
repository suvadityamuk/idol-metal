"""Idol image -> 3D Gaussians on Apple silicon with the packaged Metal kernels (HF kernels).

  python scripts/run_kernels.py [--image robot|avocado|PATH] [--mode synthetic|real] [--seed 42] [--steps 50]
                                [--cfg 3.0] [--rescale_t 3.0] [--num_gaussians 262144] [--out DIR]
                                [--local [BUILD_ROOT]] [--save-outputs FILE]

The official pipeline object (reference/idol-sdk, unmodified) loads the weights and runs preprocessing, the
encoders, the octree decoder and the PLY export as shipped. The fast build replaces:
  - the sampler: restructured denoiser (context stream once per request, refiners shared by the two CFG passes,
    collapsed unconditional context) whose block hot ops are the packaged kernels;
  - the Gaussian decoder blocks: the packaged kernels around MPS GEMMs/SDPA (bit-equal to the reference).
Packaged ops used: resid_ln_mod, repo_rope_qk, silu_mul, linear_gelu, gated_residual, resid_layernorm,
rmsnorm_heads (kernel-builder package suvadityamuk/idol-metal). GEMMs and attention stay PyTorch MPS ops.
--local loads the package from a local kernel-builder build (default: kernel/ of this repository) instead of the Hub.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image', default='robot')
    p.add_argument('--mode', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=3.0)
    p.add_argument('--rescale_t', type=float, default=3.0)
    p.add_argument('--num_gaussians', type=int, default=262144)
    p.add_argument('--out', default='runs/kernels')
    p.add_argument('--local', nargs='?', const=str(ROOT / 'kernel'), default=None,
                   help='use a local kernel-builder build (directory holding build/)')
    p.add_argument('--parity', action='store_true',
                   help='parity mode: collapsed unconditional context off; output byte-identical to the reference')
    p.add_argument('--save-outputs', default=None, help='also write an outputs dump for tools/closeness.py')
    a = p.parse_args()

    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit('error: needs PyTorch with MPS (Apple silicon)')
    from common import IMAGES, Dump
    import kernels
    if a.local:
        pkg = kernels.get_local_kernel(Path(a.local), trust_remote_code=True)
        source = f'local build {a.local}'
    else:
        pkg = kernels.get_kernel('suvadityamuk/idol-metal', version=1, trust_remote_code=True)
        source = 'Hub suvadityamuk/idol-metal v1'
    from fast import ops
    ops.use_package(pkg)
    from fast.pipeline import install
    from idol import IdolPipeline

    image, mode = IMAGES.get(a.image, (a.image, 'synthetic'))
    mode = a.mode or mode
    pipe = IdolPipeline.from_pretrained('speridlabs/idol', device='mps')
    install(pipe, ['parity'] if a.parity else [])
    print(f'device: {pipe.device} ({torch.backends.mps.is_available() and "MPS"}); build: {"parity" if a.parity else "fast"}, kernels from {source}')
    last = {}
    if a.save_outputs:
        import idol.pipeline as pm
        sf = pm.sample_flow

        def keep(*args, **kw):
            out = sf(*args, **kw)
            last['latent'], last['camera'] = out['latent'], out['camera']
            return out
        pm.sample_flow = keep
    t0 = time.perf_counter()
    out = pipe(image, mode=mode, seed=a.seed, steps=a.steps, cfg_strength=a.cfg, rescale_t=a.rescale_t,
               num_gaussians=a.num_gaussians)
    torch.mps.synchronize()
    print(f'generated {out.gaussian.xyz.shape[0]} Gaussians in {time.perf_counter() - t0:.1f} s')
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(image))[0]
    out.gaussian.save_ply(os.path.join(a.out, f'{name}.ply'))
    out.preprocessed_image.save(os.path.join(a.out, f'{name}_input.png'))
    with open(os.path.join(a.out, f'{name}_camera.json'), 'w') as f:
        json.dump(out.camera, f, indent=2)
    print(f'{name}: {a.out}/{name}.ply')
    if a.save_outputs:
        d = Dump()
        for n in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
            d.put(f'output/{n}', getattr(out.gaussian, n))
        d.put('sample/final/latent', last['latent'])
        d.put('sample/final/camera', last['camera'])
        d.save(a.save_outputs, metadata={'camera': out.camera})


if __name__ == '__main__':
    main()
