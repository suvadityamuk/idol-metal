"""Noise floor of the reference's front end: run preprocess (BiRefNet) + both image encoders in this fresh process
and save the stage outputs, so repeated invocations can be byte-compared. The VAE sample noise is recorded too,
so a difference can be attributed to the generator or to the encoder arithmetic."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGES, Dump, RNG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from idol import IdolPipeline  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='robot')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    image, mode = IMAGES[args.image]
    pipe = IdolPipeline.from_pretrained('speridlabs/idol', device='mps')
    dump = Dump()
    br = pipe.background_remover
    h1 = br.register_forward_hook(lambda m, a, out: dump.put('birefnet_alpha', out))
    dino = pipe.image_encoder.dinov3
    h2 = dino.register_forward_hook(lambda m, a, out: dump.put('dinov3_last_hidden', out.last_hidden_state))
    prepared = pipe.preprocess(image, mode=mode)
    dump.put('prepared_image', torch.from_numpy(np.array(prepared)))
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(pipe.device)
    torch.manual_seed(args.seed)
    with RNG('record') as rng:
        cond = pipe.image_encoder(pixels)
    dump.put('feature1', cond['feature1'])
    dump.put('feature2', cond['feature2'])
    for k, v in rng.as_dict().items():
        dump.put(k, v)
    h1.remove()
    h2.remove()
    dump.save(args.out)


if __name__ == '__main__':
    main()
