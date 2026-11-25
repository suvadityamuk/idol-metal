"""Python helpers around the native sampler (scripts/run_native.sh). NOT part of the native process:

  prepare  run the official front end (resize, BiRefNet, crop, DINOv3, FLUX.2 VAE) and draw the noise with the
           pipeline's seed and call order; save encode/feature1, encode/feature2, sample/noise/{latent,camera}
           and the MPS RNG state after the noise draws (so a later decode uses the same random stream as the
           official pipeline would)
  decode   turn a native output latent into Gaussians/PLY with the official octree + Gaussian decoders,
           restoring that RNG state first; writes <out>.ply and an outputs dump for tools/closeness.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGES, Dump  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from common import DumpReader  # noqa: E402


def prepare(a):
    from idol import IdolPipeline
    pipe = IdolPipeline.from_pretrained('speridlabs/idol', device='mps')
    image, mode = IMAGES.get(a.image, (a.image, a.mode or 'synthetic'))
    prepared = pipe.preprocess(image, mode=a.mode or mode)
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(pipe.device)
    torch.manual_seed(a.seed)
    cond = pipe.image_encoder(pixels)
    lat = torch.randn(1, pipe.denoiser.q_token_length, pipe.denoiser.in_channels, device=pipe.device)
    cam = torch.randn(1, 1, pipe.denoiser.cam_channels, device=pipe.device)
    state = torch.mps.get_rng_state()
    save_file({'encode/feature1': cond['feature1'].float().cpu().contiguous(),
               'encode/feature2': cond['feature2'].float().cpu().contiguous(),
               'sample/noise/latent': lat.cpu().contiguous(), 'sample/noise/camera': cam.cpu().contiguous(),
               'rng/mps_state_after_noise': state.contiguous()}, a.out, metadata={'image': image, 'seed': str(a.seed)})
    prepared.save(os.path.splitext(a.out)[0] + '_input.png')
    print(f'wrote {a.out}')


def decode(a):
    from idol import IdolPipeline
    from idol.camera import decode_camera
    pipe = IdolPipeline.from_pretrained('speridlabs/idol', device='mps')
    n = DumpReader(a.latent)
    inp = DumpReader(a.inputs)
    latent = n.get('sample/final/latent', 'mps')
    torch.mps.set_rng_state(inp.get('rng/mps_state_after_noise'))
    with torch.no_grad():
        g = pipe.decode(latent, a.num_gaussians)
    g.save_ply(a.out + '.ply')
    d = Dump()
    for name in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
        d.put(f'output/{name}', getattr(g, name))
    d.put('sample/final/latent', latent)
    d.put('sample/final/camera', n.get('sample/final/camera'))
    d.save(a.out + '_outputs.safetensors', metadata={'camera': decode_camera(n.get('sample/final/camera')[0, 0])})
    print(f'wrote {a.out}.ply and {a.out}_outputs.safetensors')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--image', default='robot')
    q.add_argument('--mode', default=None)
    q.add_argument('--seed', type=int, default=42)
    q.add_argument('--out', required=True)
    q = sub.add_parser('decode')
    q.add_argument('--latent', required=True)
    q.add_argument('--inputs', required=True)
    q.add_argument('--num_gaussians', type=int, default=262144)
    q.add_argument('--out', required=True, help='output path prefix')
    a = p.parse_args()
    prepare(a) if a.cmd == 'prepare' else decode(a)


if __name__ == '__main__':
    main()
