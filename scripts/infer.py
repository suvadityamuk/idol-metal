#!/usr/bin/env python3
"""idol-metal: one image in, a 3D Gaussian splat (.ply) out, on Apple silicon (PyTorch MPS + fused Metal kernels).

  python scripts/infer.py photo.jpg -o photo.ply                          # fast build, kernels from the Hub
  python scripts/infer.py photo.jpg -o photo.ply --local-kernels kernel   # kernels from a local kernel-builder build
  python scripts/infer.py robot.webp -o robot.ply --mode synthetic --build parity --check reference.ply

Builds:
  fast    the official pipeline object with its sampler restructured (context once per request, refiners shared
          by both guidance passes, one-token unconditional context) and its denoiser and Gaussian-decoder hot ops
          replaced by the idol-metal kernels. Close to the official output, not byte-identical.
  parity  the fast build with its only arithmetic-changing change (the one-token unconditional context) off. On the
          tested machine (M5 Pro, macOS 26.6.2, torch 2.15.0) its .ply is byte-identical to the official pipeline's
          on MPS on the four fixed configs.

Kernels: kernels.get_kernel("suvadityamuk/idol-metal", version=1), or --local-kernels DIR (a kernel-builder project
directory holding build/<variant>/, e.g. kernel/). Everything else is the official idol-sdk code (8db7003).

Needs macOS 26+ on Apple silicon (Metal 4: the linear_gelu kernel uses MetalPerformancePrimitives), PyTorch with MPS,
the packages in requirements.txt, and a Hugging Face token with access to the gated encoders
facebook/dinov3-vith16plus-pretrain-lvd1689m and black-forest-labs/FLUX.2-dev (`hf auth login` or HF_TOKEN).
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KERNEL_REPO = 'suvadityamuk/idol-metal'
KERNEL_VERSION = 1
GATED = ('facebook/dinov3-vith16plus-pretrain-lvd1689m', 'black-forest-labs/FLUX.2-dev')


class SetupError(RuntimeError):
    pass


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', help='input image (a photo, render or generated image of one object)')
    p.add_argument('-o', '--out', default=None, help='output .ply (default: <image name>.ply)')
    p.add_argument('--mode', choices=['real', 'synthetic'], default='real',
                   help='real: photos (looser crop); synthetic: renders / generated images (default: real)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=3.0)
    p.add_argument('--rescale-t', type=float, default=3.0)
    p.add_argument('--gaussians', '--num-gaussians', dest='gaussians', type=int, default=262144,
                   help='number of Gaussians, 32,768 to 1,048,576 (default 262,144)')
    p.add_argument('--build', choices=['fast', 'parity'], default='fast')
    p.add_argument('--local-kernels', default=None, metavar='PATH',
                   help='kernel-builder project directory holding build/<variant>/ (e.g. kernel/) instead of the Hub')
    p.add_argument('--check', default=None, metavar='REF.ply', help='byte-compare the .ply; exit 1 if it differs')
    p.add_argument('--dump', default=None, metavar='FILE.safetensors',
                   help='also save the final latent, camera and Gaussians (for tools/closeness.py)')
    return p.parse_args()


def check_platform():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SetupError(f'idol-metal needs macOS on Apple silicon; this is {platform.system()} {platform.machine()}.')
    ver = platform.mac_ver()[0]
    major = int(ver.split('.')[0]) if ver else 0
    if major < 26:
        raise SetupError(f'macOS {ver or "(unknown)"} is not supported: the linear_gelu kernel uses Metal 4 '
                         '(MetalPerformancePrimitives), which needs macOS 26 or later. Only macOS 26.6.2 was tested.')
    import torch
    if not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        raise SetupError('PyTorch reports no MPS device' + ('' if built else ' (this torch build has no MPS support)')
                         + '; idol-metal needs PyTorch with MPS on Apple silicon.')
    return torch


def load_kernels(local):
    try:
        import kernels
    except ImportError:
        raise SetupError('The `kernels` package is not installed: pip install -r requirements.txt') from None
    from pathlib import Path
    if local:
        if not os.path.isdir(os.path.join(local, 'build')):
            raise SetupError(f'{local}/build not found: --local-kernels takes the kernel project directory '
                             '(the one that contains build/<variant>/), e.g. kernel/')
        try:
            pkg = kernels.get_local_kernel(Path(os.path.abspath(local)), trust_remote_code=True)
        except Exception as e:
            raise SetupError(f'Loading the local kernel build in {local} failed ({type(e).__name__}: {e}). It needs '
                             'a variant for this torch version (torch21x-metal-aarch64-darwin).') from None
        return pkg, f'local kernel-builder build {local} (kernels {kernels.__version__})'
    try:
        pkg = kernels.get_kernel(KERNEL_REPO, version=KERNEL_VERSION, trust_remote_code=True)
    except Exception as e:
        raise SetupError(f'Could not load {KERNEL_REPO} (version {KERNEL_VERSION}) from the Hub '
                         f'({type(e).__name__}: {str(e).splitlines()[0] if str(e) else ""}). If the kernel repo is not '
                         'published yet, or you are offline, build the kernels locally and pass '
                         '--local-kernels kernel/ (see README, Development).') from None
    return pkg, f'Hub {KERNEL_REPO} version {KERNEL_VERSION} (kernels {kernels.__version__})'


def check_gated_access():
    try:
        from idol.conditioning import EncoderAccessError, require_encoder_access
    except ImportError:
        raise SetupError('idol-sdk is not installed: pip install -r requirements.txt') from None
    try:
        require_encoder_access(list(GATED))
    except EncoderAccessError as e:
        raise SetupError(str(e)) from None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    a = parse()
    if not os.path.isfile(a.image):
        sys.exit(f'error: {a.image} not found')
    out = a.out or os.path.splitext(os.path.basename(a.image))[0] + '.ply'
    if not out.endswith('.ply'):
        sys.exit('error: -o must end in .ply')
    t_start = time.perf_counter()
    try:
        torch = check_platform()
        pkg, source = load_kernels(a.local_kernels)      # before any large download: fail fast
        check_gated_access()
    except SetupError as e:
        sys.exit(f'error: {e}')

    from fast import ops
    from fast.pipeline import install
    from idol import IdolPipeline
    ops.use_package(pkg)
    desc = {'fast': 'fast (restructured sampler + idol-metal kernels)',
            'parity': 'parity (byte-identical to the official pipeline on MPS on the tested setup)'}
    print(f'device:  Apple silicon, MPS, macOS {platform.mac_ver()[0]}, torch {torch.__version__}')
    print(f'build:   {desc[a.build]}')
    print(f'kernels: {source}')

    t = time.perf_counter()
    pipe = IdolPipeline.from_pretrained('speridlabs/idol', device='mps')
    install(pipe, ['parity'] if a.build == 'parity' else [])
    torch.mps.synchronize()
    print(f'load:    {time.perf_counter() - t:.1f} s (weights, encoders)')

    last = {}
    if a.dump:
        import idol.pipeline as pm
        sf = pm.sample_flow

        def keep(*args, **kw):
            res = sf(*args, **kw)
            last['latent'], last['camera'] = res['latent'], res['camera']
            return res
        pm.sample_flow = keep

    t = time.perf_counter()
    result = pipe(a.image, mode=a.mode, seed=a.seed, steps=a.steps, cfg_strength=a.cfg, rescale_t=a.rescale_t,
                  num_gaussians=a.gaussians)
    torch.mps.synchronize()
    print(f'request: {time.perf_counter() - t:.1f} s (first request in this process: includes one-off shader '
          'compilation)')

    t = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    result.gaussian.save_ply(out)
    stem = out[:-4]
    result.preprocessed_image.save(stem + '_input.png')
    with open(stem + '_camera.json', 'w') as f:
        json.dump(result.camera, f, indent=2)
    print(f'export:  {time.perf_counter() - t:.2f} s -> {out}, {stem}_input.png, {stem}_camera.json')
    print(f'total:   {time.perf_counter() - t_start:.1f} s wall (including start-up and loading)')
    print(f'sha256:  {sha256(out)}')

    if a.dump:
        from safetensors.torch import save_file
        d = {f'output/{n}': getattr(result.gaussian, n).detach().cpu().contiguous()
             for n in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity')}
        d['sample/final/latent'] = last['latent'].detach().cpu().contiguous()
        d['sample/final/camera'] = last['camera'].detach().cpu().contiguous()
        save_file(d, a.dump)
        print(f'dump:    {a.dump}')
    if a.check:
        same = sha256(out) == sha256(a.check)
        print(f'check vs {a.check}: {"PLY BYTE-IDENTICAL" if same else "PLY DIFFERS"}')
        if not same:
            sys.exit(1)


if __name__ == '__main__':
    main()
