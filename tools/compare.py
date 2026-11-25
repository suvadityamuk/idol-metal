"""Compare two dumps tensor by tensor: bit equality, differing count, first differing element, max abs / ulp error.

python tools/compare.py A.safetensors B.safetensors [--prefix dit/] [--only-common] [--json out.json]
Exit status 1 if any compared tensor differs (or a key is missing, unless --only-common).
"""
import argparse
import json
import sys

import torch
from safetensors import safe_open

_INT_VIEW = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16,
             torch.float64: torch.int64}


def ulp_distance(a, b):
    """Elementwise distance in units in the last place (monotone integer mapping of the float bit patterns)."""
    it = _INT_VIEW[a.dtype]
    ia, ib = a.contiguous().view(it).to(torch.int64), b.contiguous().view(it).to(torch.int64)
    bits = torch.iinfo(it).bits
    sign = 1 << (bits - 1)
    ia = torch.where(ia < 0, -(ia + sign), ia) if bits < 64 else ia
    ib = torch.where(ib < 0, -(ib + sign), ib) if bits < 64 else ib
    return (ia - ib).abs()


def compare_tensors(a, b):
    """-> dict with equal (bitwise), n_diff, first index, max_abs, mean_abs, max_ulp, rel_l2."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'equal': False, 'error': f'shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}'}
    if a.dtype.is_floating_point:
        it = _INT_VIEW[a.dtype]
        neq = a.contiguous().view(it) != b.contiguous().view(it)
    else:
        neq = a != b
    n = int(neq.sum())
    out = {'equal': n == 0, 'n_diff': n, 'numel': a.numel()}
    if n:
        flat = neq.flatten().nonzero()[0, 0].item()
        idx = list(torch.unravel_index(torch.tensor(flat), a.shape)) if a.dim() else []
        out['first'] = [int(i) for i in idx]
        af, bf = a.flatten()[flat].item(), b.flatten()[flat].item()
        out['first_values'] = [af, bf]
        if a.dtype.is_floating_point:
            d = (a.double() - b.double()).abs()
            finite = torch.isfinite(d)
            out['max_abs'] = float(d[finite].max()) if finite.any() else float('nan')
            out['mean_abs'] = float(d[finite].mean()) if finite.any() else float('nan')
            out['max_ulp'] = int(ulp_distance(a, b).max())
            denom = float(b.double().norm())
            out['rel_l2'] = float((a.double() - b.double()).norm()) / denom if denom else float('inf')
            out['ref_absmax'] = float(b.double().abs().max())
        else:
            out['max_abs'] = int((a.long() - b.long()).abs().max())
    return out


def compare_files(path_a, path_b, prefix=None, only_common=False, keys=None):
    fa, fb = safe_open(path_a, 'pt'), safe_open(path_b, 'pt')
    ka, kb = set(fa.keys()), set(fb.keys())
    if prefix:
        ka = {k for k in ka if k.startswith(tuple(prefix))}
        kb = {k for k in kb if k.startswith(tuple(prefix))}
    if keys is not None:
        ka &= set(keys)
        kb &= set(keys)
    results = {}
    for k in sorted(ka & kb, key=_natural):
        results[k] = compare_tensors(fa.get_tensor(k), fb.get_tensor(k))
    missing = {'only_a': sorted(ka - kb), 'only_b': sorted(kb - ka)}
    return results, missing


def _natural(k):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', k)]


def summarize(results, missing, only_common=False, show=40, out=sys.stdout):
    diff = {k: r for k, r in results.items() if not r['equal']}
    print(f'{len(results)} tensors compared, {len(results) - len(diff)} bit-identical, {len(diff)} differ', file=out)
    if not only_common and (missing['only_a'] or missing['only_b']):
        print(f"missing: only in A {len(missing['only_a'])}, only in B {len(missing['only_b'])}", file=out)
        for k in (missing['only_a'] + missing['only_b'])[:10]:
            print('   ', k, file=out)
    for k, r in list(diff.items())[:show]:
        if 'error' in r:
            print(f'  {k}: {r["error"]}', file=out)
            continue
        extra = ''
        if 'max_ulp' in r:
            extra = (f" max_abs={r['max_abs']:.3e} mean_abs={r['mean_abs']:.3e} max_ulp={r['max_ulp']} "
                     f"rel_l2={r['rel_l2']:.3e}")
        elif 'max_abs' in r:
            extra = f" max_abs={r['max_abs']}"
        print(f"  {k}: {r['n_diff']}/{r['numel']} differ, first {r['first']} {r['first_values']}{extra}", file=out)
    if len(diff) > show:
        print(f'  ... {len(diff) - show} more', file=out)
    ok = not diff and (only_common or not (missing['only_a'] or missing['only_b']))
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--prefix', nargs='*')
    p.add_argument('--only-common', action='store_true')
    p.add_argument('--json')
    p.add_argument('--show', type=int, default=40)
    args = p.parse_args()
    results, missing = compare_files(args.a, args.b, args.prefix, args.only_common)
    ok = summarize(results, missing, args.only_common, args.show)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'results': results, 'missing': missing}, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
