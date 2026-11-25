"""Install the fast build into a loaded IdolPipeline (the official object; weights and other stages unchanged).

Stages:
  sampler  restructured denoiser with fused Metal kernels (fast/dit.py, fast/ops.py)
  decoder  Gaussian decoder blocks with the fused kernels (fast/decoder.py; bit-equal to the reference)
  'parity' in stages: parity mode = the fast build with its only arithmetic-changing option (the collapsed
           unconditional context) switched off; every other change is bit-equal, so the output .ply is
           byte-identical to the reference's on this device (regression check: tools/parity_check.sh)
"""
import idol.pipeline as pipeline_mod

from .dit import FastDenoiser, sample_flow_fast


def install(pipe, stages=(), impl='kernels'):
    cache = {}

    def sample_flow(model, noise, cond, neg_cond, steps=50, rescale_t=3.0, cfg_strength=3.0, verbose=True,
                    batch_cfg=False):
        fd = cache.get(id(model))
        if fd is None:
            fd = cache[id(model)] = FastDenoiser(model, impl=impl, collapse_uncond='parity' not in stages)
        return sample_flow_fast(fd, noise, cond, neg_cond, steps=steps, rescale_t=rescale_t,
                                cfg_strength=cfg_strength)

    pipeline_mod.sample_flow = sample_flow
    if 'no_decoder' not in stages:
        from .decoder import FastGaussianDecoder
        pipe.decoder_gs.forward = FastGaussianDecoder(pipe.decoder_gs).forward
    pipe._fast_stages = tuple(stages)
    return pipe
