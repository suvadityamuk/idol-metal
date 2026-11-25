"""Teacher-forced A/B of the Gaussian decoder: reference anchors + final latent from a dump through the reference
decoder_gs.forward and the fused-kernel forward, interleaved timing, byte/err comparison of the per-anchor outputs."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DumpReader, write_json
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from idol.models import GaussianFixedlenDecoder
from fast.decoder import FastGaussianDecoder


def load_dec(dev='mps'):
    d = snapshot_download('speridlabs/idol', allow_patterns=['config.json', 'decoder_gs.safetensors'])
    cfg = json.load(open(os.path.join(d, 'config.json')))
    m = GaussianFixedlenDecoder(**cfg['decoder_gs']['args'])
    m.load_state_dict(load_file(os.path.join(d, 'decoder_gs.safetensors')), strict=True)
    return m.eval().requires_grad_(False).to(dev)


def main():
    dump = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    r = DumpReader(dump)
    dec = load_dec()
    fdec = FastGaussianDecoder(dec)
    pts = r.get('decode/anchors', 'mps'); lat = r.get('sample/final/latent', 'mps')
    T = {'ref': [], 'fast': []}
    with torch.no_grad():
        for i in range(reps + 1):
            for arm in (('ref', 'fast') if i % 2 == 0 else ('fast', 'ref')):
                torch.mps.synchronize(); t0 = time.perf_counter()
                out = dec(pts, lat) if arm == 'ref' else fdec.forward(pts, lat)
                torch.mps.synchronize(); dt = time.perf_counter() - t0
                if i: T[arm].append(dt)
                if arm == 'ref': o_ref = out
                else: o_fast = out
    d = (o_fast - o_ref).abs()
    res = {'anchors': int(pts.shape[1]), 'ref_s': T['ref'], 'fast_s': T['fast'], 'bit_equal': bool(torch.equal(o_fast, o_ref)),
           'max_abs': float(d.max()), 'rel_l2': float((o_fast - o_ref).norm() / o_ref.norm())}
    if 'decode/gs_features' in r:
        g = r.get('decode/gs_features', 'mps'); res['ref_vs_dump_bit_equal'] = bool(torch.equal(o_ref, g))
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"anchors {res['anchors']}: ref {med(T['ref'])*1e3:.0f} ms, fast {med(T['fast'])*1e3:.0f} ms, speed-up {med(T['ref'])/med(T['fast']):.2f}x; "
          f"bit_equal {res['bit_equal']} max {res['max_abs']:.2e} rel {res['rel_l2']:.2e}; ref==dump {res.get('ref_vs_dump_bit_equal')}")
    if len(sys.argv) > 3: write_json(sys.argv[3], res)


if __name__ == '__main__':
    main()
