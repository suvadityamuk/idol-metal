"""Run the official Idol pipeline (reference/idol-sdk, unmodified) on MPS, instrumented from outside.

  --dump PATH   save every random tensor, stage-boundary tensors and first-block intermediates to safetensors
  (always)      per-stage synchronised timings, peak memory, the PLY and its sha256 -> <out>/result.json

Timing runs should not use --dump (hooks copy tensors to the host). The reference code is imported, never edited:
stage timers and dump hooks wrap its functions/modules at runtime.
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (IMAGES, Dump, MemorySampler, RNG, StageTimer, environment, footprint, sha256_file,
                    sync, write_json)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import idol.models.decoder as decoder_mod  # noqa: E402
import idol.models.layers as layers  # noqa: E402
import idol.pipeline as pipeline_mod  # noqa: E402
from idol import IdolPipeline  # noqa: E402
from idol.models.decoder import OctreeProbabilityFixedlenDecoder  # noqa: E402

# modules whose every sub-module input/output is dumped (first block of each distinct layer type)
DIT_DEEP = ('t_embedder', 'adaLN_modulation', 'input_layer', 'cond_embedder', 'cond_embedder2', 'pos_embedder',
            'noise_repo_layers.0', 'noise_refiner.0', 'context_repo_layers.0', 'context_refiner.0', 'cam_refiner',
            'repo_layers.0', 'blocks.0', 'out_layer', 'cam_out_layer')
TOP_BLOCK = re.compile(r'^((blocks|noise_refiner|context_refiner|noise_repo_layers|context_repo_layers)\.\d+'
                       r'|repo_layers\.(0|1|23))$')
DEC_DEEP = ('in_proj', 'pos_embedder', 'input_layer', 'l_embedder', 'adaLN_modulation', 'blocks.0', 'out_proj')
DEC_TOP = re.compile(r'^blocks\.\d+$')


def _tensors(value, prefix):
    """Flatten a module output (tensor, tuple, TensorDict, HF ModelOutput) into (suffix, tensor) pairs."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from _tensors(v, f'{prefix}.{i}')
    elif hasattr(value, 'keys') and hasattr(value, '__getitem__'):
        for k in value.keys():
            try:
                yield from _tensors(value[k], f'{prefix}.{k}')
            except Exception:
                pass


class ModuleCapture:
    """Forward (pre-)hooks on a model: on calls listed in deep_calls, dump inputs/outputs of every sub-module under
    the deep prefixes and outputs of top-level blocks; on every call dump the model's own output."""

    def __init__(self, dump, model, tag, deep, top, deep_calls=(0,), every_call_io=True, record_inputs=True,
                 input_args_every_call=None):
        self.dump, self.tag, self.call = dump, tag, -1
        self.deep_calls = set(deep_calls)
        self.handles = []
        self.functions_active = False

        def pre_root(module, args, kwargs):
            self.call += 1
            if every_call_io and record_inputs:
                for i, a in enumerate(args):
                    if self.call > 0 and input_args_every_call is not None and i not in input_args_every_call:
                        continue
                    for suffix, t in _tensors(a, f'arg{i}'):
                        dump.put(f'{tag}/call{self.call:03d}/in/{suffix}', t)
            self.functions_active = self.call in self.deep_calls

        def post_root(module, args, kwargs, out):
            if every_call_io:
                for suffix, t in _tensors(out, 'out'):
                    dump.put(f'{tag}/call{self.call:03d}/{suffix}', t)
            self.functions_active = False

        self.handles.append(model.register_forward_pre_hook(pre_root, with_kwargs=True))
        self.handles.append(model.register_forward_hook(post_root, with_kwargs=True))
        for name, module in model.named_modules():
            if not name:
                continue
            is_deep = any(name == d or name.startswith(d + '.') for d in deep)
            is_top = bool(top.match(name))
            if not (is_deep or is_top):
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre(name, is_deep), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self._post(name), with_kwargs=True))

    def _pre(self, name, is_deep):
        def hook(module, args, kwargs):
            if self.call in self.deep_calls and is_deep and record_inputs_enabled:
                for i, a in enumerate(args):
                    for suffix, t in _tensors(a, f'in{i}'):
                        self.dump.put(f'{self.tag}/call{self.call:03d}/{name}/{suffix}', t)
                for k, a in kwargs.items():
                    for suffix, t in _tensors(a, f'kw_{k}'):
                        self.dump.put(f'{self.tag}/call{self.call:03d}/{name}/{suffix}', t)
        return hook

    def _post(self, name):
        def hook(module, args, kwargs, out):
            if self.call in self.deep_calls:
                for suffix, t in _tensors(out, 'out'):
                    self.dump.put(f'{self.tag}/call{self.call:03d}/{name}/{suffix}', t)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


record_inputs_enabled = True


def instrument_functions(dump, captures):
    """Record inputs/outputs of the functional ops inside deep calls (rotary embedding, attention)."""
    orig_rope, orig_sdpa = layers.apply_rotary_emb, layers.scaled_dot_product_attention
    counters = {}

    def active():
        for c in captures:
            if c.functions_active:
                return c
        return None

    def rope(hidden_states, freqs):
        out = orig_rope(hidden_states, freqs)
        c = active()
        if c is not None:
            k = counters.setdefault((c.tag, c.call, 'rope'), [0])
            pre = f'{c.tag}/call{c.call:03d}/fn/rope{k[0]:02d}'
            k[0] += 1
            if k[0] <= 6:
                dump.put(pre + '/x', hidden_states)
                dump.put(pre + '/out', out)
        return out

    def sdpa(*args):
        out = orig_sdpa(*args)
        c = active()
        if c is not None:
            k = counters.setdefault((c.tag, c.call, 'sdpa'), [0])
            pre = f'{c.tag}/call{c.call:03d}/fn/sdpa{k[0]:02d}'
            k[0] += 1
            if k[0] <= 6:
                for i, a in enumerate(args):
                    dump.put(f'{pre}/in{i}', a)
                dump.put(pre + '/out', out)
        return out

    layers.apply_rotary_emb, layers.scaled_dot_product_attention = rope, sdpa

    def restore():
        layers.apply_rotary_emb, layers.scaled_dot_product_attention = orig_rope, orig_sdpa
    return restore


def run_once(pipe, image, mode, args, timer, dump, rng_mode='record', rng_stored=None):
    """One pipeline call with stage timers (and dumps when dump.enabled). Returns (output, rng)."""
    restore = []
    # stage timers -------------------------------------------------------------------------------------------
    orig_pre = pipe.preprocess
    pipe.preprocess = timer.wrap('preprocess', orig_pre)
    restore.append(lambda: setattr(pipe, 'preprocess', orig_pre))
    br = pipe.background_remover
    orig_rb = br.remove_background
    br.remove_background = timer.wrap('preprocess/birefnet', orig_rb)
    restore.append(lambda: setattr(br, 'remove_background', orig_rb))
    enc = pipe.image_encoder
    orig_d, orig_f = enc._dinov3_tokens, enc._flux2_tokens
    enc._dinov3_tokens = timer.wrap('encode/dinov3', orig_d)
    enc._flux2_tokens = timer.wrap('encode/flux2_vae', orig_f)
    restore.append(lambda: (setattr(enc, '_dinov3_tokens', orig_d), setattr(enc, '_flux2_tokens', orig_f)))
    orig_sf = pipeline_mod.sample_flow
    pipeline_mod.sample_flow = timer.wrap('sample', orig_sf)
    restore.append(lambda: setattr(pipeline_mod, 'sample_flow', orig_sf))
    orig_oct = OctreeProbabilityFixedlenDecoder.__dict__['sample']
    OctreeProbabilityFixedlenDecoder.sample = staticmethod(timer.wrap('decode/octree', orig_oct.__func__))
    restore.append(lambda: setattr(OctreeProbabilityFixedlenDecoder, 'sample', orig_oct))
    gs = pipe.decoder_gs
    orig_gsf = gs.forward
    gs.forward = timer.wrap('decode/gaussian_decoder', orig_gsf)
    restore.append(lambda: setattr(gs, 'forward', orig_gsf))
    orig_rep = gs.to_representation
    gs.to_representation = timer.wrap('decode/to_representation', orig_rep)
    restore.append(lambda: setattr(gs, 'to_representation', orig_rep))

    captures = []
    if dump.enabled:
        # stage-boundary records -------------------------------------------------------------------------------
        def enc_call(images, _orig=enc.__class__.__call__):
            dump.put('encode/pixels', images)
            out = _orig(enc, images)
            dump.put('encode/feature1', out['feature1'])
            dump.put('encode/feature2', out['feature2'])
            return out
        pipe.image_encoder = type('EncoderTap', (), {'__call__': staticmethod(enc_call)})()
        restore.append(lambda: setattr(pipe, 'image_encoder', enc))

        def sf_tap(model, noise, cond, neg_cond, **kw):
            dump.put('sample/noise/latent', noise['latent'])
            dump.put('sample/noise/camera', noise['camera'])
            out = pipeline_mod.sample_flow.__wrapped__(model, noise, cond, neg_cond, **kw)
            dump.put('sample/final/latent', out['latent'])
            dump.put('sample/final/camera', out['camera'])
            return out
        sf_timed = timer.wrap('sample', sf_tap)
        sf_timed.__wrapped__ = orig_sf
        pipeline_mod.sample_flow = sf_timed

        orig_ss = decoder_mod.systematic_sample

        def ss_tap(probs, counts):
            out = orig_ss(probs, counts)
            dump.put('decode/octree/probs', probs)
            dump.put('decode/octree/counts_in', counts)
            dump.put('decode/octree/counts_out', out)
            return out
        decoder_mod.systematic_sample = ss_tap
        restore.append(lambda: setattr(decoder_mod, 'systematic_sample', orig_ss))

        oct_timed = OctreeProbabilityFixedlenDecoder.sample

        def oct_tap(model, cond, num_points, level):
            out = oct_timed(model, cond, num_points, level)
            dump.put('decode/anchors', out)
            return out
        OctreeProbabilityFixedlenDecoder.sample = staticmethod(oct_tap)

        rep_timed = gs.to_representation

        def rep_tap(points, features):
            dump.put('decode/gs_features', features)
            return rep_timed(points, features)
        gs.to_representation = rep_tap

        # module hooks -------------------------------------------------------------------------------------------
        def br_pre(m, a):
            dump.put('preprocess/birefnet/in', a[0])

        def br_post(m, a, out):
            dump.put('preprocess/birefnet/alpha', out)
        hs = [br.register_forward_pre_hook(br_pre), br.register_forward_hook(br_post)]
        for name in ('bb', 'squeeze_module', 'decoder'):
            sub = getattr(br, name)

            def hook(m, a, out, name=name):
                for suffix, t in _tensors(out, 'out'):
                    dump.put(f'preprocess/birefnet/{name}/{suffix}', t)
            hs.append(sub.register_forward_hook(hook))
        dino = enc.dinov3

        def dino_hook(prefix):
            def hook(m, a, out):
                for suffix, t in _tensors(out, 'out'):
                    dump.put(f'encode/dinov3/{prefix}/{suffix}', t)
            return hook
        for name, m in dino.named_modules():
            if name in ('embeddings', 'layer.0', 'norm') or name.endswith('rope_embeddings'):
                hs.append(m.register_forward_hook(dino_hook(name)))
        hs.append(dino.register_forward_hook(dino_hook('model')))
        vae = enc.flux2_vae
        for name in ('encoder', 'quant_conv'):
            sub = getattr(vae, name, None)
            if sub is not None:
                def vae_hook(m, a, out, name=name):
                    for s, t in _tensors(out, 'out'):
                        dump.put(f'encode/flux2_vae/{name}/{s}', t)
                hs.append(sub.register_forward_hook(vae_hook))
        restore.append(lambda: [h.remove() for h in hs])

        captures.append(ModuleCapture(dump, pipe.denoiser, 'dit', DIT_DEEP, TOP_BLOCK, deep_calls=(0,),
                                      input_args_every_call=(0, 1)))
        level = pipe.config['max_voxel_level']
        captures.append(ModuleCapture(dump, pipe.decoder, 'octree', DEC_DEEP, DEC_TOP, deep_calls=(0, level - 1)))
        captures.append(ModuleCapture(dump, pipe.decoder_gs, 'gsdec', DEC_DEEP, DEC_TOP, deep_calls=(0,)))
        restore.append(instrument_functions(dump, captures))
        for c in captures:
            restore.append(c.remove)

    rng = RNG(rng_mode, rng_stored)
    try:
        with rng, timer('total'):
            output = pipe(image, mode=mode, seed=args.seed, steps=args.steps, cfg_strength=args.cfg,
                          rescale_t=args.rescale_t, num_gaussians=args.num_gaussians, verbose=False)
    finally:
        for r in reversed(restore):
            r()
    return output, rng


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='robot', help='robot | avocado | a path')
    p.add_argument('--mode', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=3.0)
    p.add_argument('--rescale_t', type=float, default=3.0)
    p.add_argument('--num_gaussians', type=int, default=262144)
    p.add_argument('--out', required=True)
    p.add_argument('--dump', default=None, help='safetensors path for the reference dump')
    p.add_argument('--repeat', type=int, default=1, help='timed repetitions in this process')
    p.add_argument('--warmup', default='short', choices=['none', 'short'],
                   help="'short': one 2-step, 32,768-Gaussian call first (shader/graph compilation, not timed)")
    p.add_argument('--no_optimize', action='store_true', help='pass optimize=False (identical to default on MPS)')
    p.add_argument('--device', default='mps')
    p.add_argument('--dinov3_dtype', default='bf16', choices=['bf16', 'fp32'])
    p.add_argument('--real_valued_rope', action='store_true',
                   help="RePo layers return (cos, sin) pairs, the RoPE form the reference's compile path uses")
    p.add_argument('--replay', default=None,
                   help='a reference dump: replay its random numbers instead of drawing them (self-check)')
    args = p.parse_args()

    image, default_mode = IMAGES.get(args.image, (args.image, 'synthetic'))
    mode = args.mode or default_mode
    os.makedirs(args.out, exist_ok=True)
    mem = MemorySampler()
    t0 = time.perf_counter()
    pipe = IdolPipeline.from_pretrained('speridlabs/idol', device=args.device, optimize=not args.no_optimize,
                                        dinov3_dtype=args.dinov3_dtype)
    if args.real_valued_rope:
        from idol.models.layers import RePo3DRotaryEmbedding
        for m in pipe.denoiser.modules():
            if isinstance(m, RePo3DRotaryEmbedding):
                m.real_valued = True
    rng_stored = None
    if args.replay:
        from safetensors.torch import load_file
        rng_stored = RNG.from_dict({k: v for k, v in load_file(args.replay).items() if k.startswith('rng/')})
        print(f'replaying {len(rng_stored)} random tensors from {args.replay}', flush=True)
    sync()
    load_s = time.perf_counter() - t0
    load_mem = mem.peak()
    print(f'loaded in {load_s:.1f}s; optimize={pipe.optimize}; attn={layers.ATTN_BACKEND}', flush=True)

    if args.warmup == 'short':
        class W:
            seed, steps, cfg, rescale_t, num_gaussians = args.seed, 2, args.cfg, args.rescale_t, 32768
        run_once(pipe, image, mode, W, StageTimer(), Dump(False))
        sync()
        print(f'warm-up done', flush=True)

    runs = []
    for r in range(args.repeat):
        timer = StageTimer()
        dump = Dump(enabled=bool(args.dump) and r == 0)
        torch.mps.empty_cache()
        mem.reset()
        output, rng = run_once(pipe, image, mode, args, timer, dump, rng_mode='replay' if rng_stored else 'record',
                               rng_stored=rng_stored)
        if rng_stored is not None and rng.index != len(rng_stored):
            raise SystemExit(f'replay consumed {rng.index} of {len(rng_stored)} recorded random tensors')
        peak = mem.peak()
        ply = os.path.join(args.out, f'run{r}.ply')
        with timer('save_ply'):
            output.gaussian.save_ply(ply)
        run = {'times_s': dict(timer.times), 'peak_memory': peak, 'ply_sha256': sha256_file(ply),
               'camera': output.camera, 'num_gaussians_out': int(output.gaussian.xyz.shape[0])}
        runs.append(run)
        print(f'run {r}: ' + ', '.join(f'{k}={v:.2f}s' for k, v in timer.times.items()) +
              f"; peak footprint {peak['footprint_peak_gb']:.1f} GB, mps driver {peak['mps_driver_peak_gb']:.1f} GB"
              f"; ply {run['ply_sha256'][:16]}", flush=True)
        if r == 0:
            output.preprocessed_image.save(os.path.join(args.out, 'input.png'))
        if dump.enabled:
            g = output.gaussian
            for name in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
                dump.put(f'output/{name}', getattr(g, name))
            dump.put('preprocess/prepared_image', torch.from_numpy(np.array(output.preprocessed_image)))
            if rng_stored is None:
                for k, v in rng.as_dict().items():
                    dump.put(k, v)
            else:
                for i, (name, v) in enumerate(rng_stored):
                    dump.put(f'rng/{i:04d}_{name}', v)
            dump.save(args.dump, metadata={'image': image, 'mode': mode, 'seed': args.seed, 'steps': args.steps,
                                           'cfg': args.cfg, 'rescale_t': args.rescale_t,
                                           'num_gaussians': args.num_gaussians, 'ply_sha256': run['ply_sha256'],
                                           'camera': output.camera})
            print(f'dump: {args.dump} ({len(dump.data)} tensors)', flush=True)
        del output
    write_json(os.path.join(args.out, 'result.json'), {
        'args': vars(args), 'image': image, 'mode': mode, 'load_s': load_s, 'load_peak_memory': load_mem,
        'lifetime_max_footprint_gb': footprint()[1] / 2**30, 'runs': runs, 'env': environment(),
        'optimize_effective': pipe.optimize, 'attn_backend': layers.ATTN_BACKEND,
        'PYTORCH_MPS_DISABLE_PREFILL_ATTENTION': os.environ.get('PYTORCH_MPS_DISABLE_PREFILL_ATTENTION')})
    mem.stop()


if __name__ == '__main__':
    main()
