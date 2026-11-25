"""End-to-end closeness of one build's outputs to a reference run (same image, seed, config).

python tools/closeness.py REF.safetensors OTHER.safetensors [--json out.json]

Reports
  latent        final sampled latent [1, 8192, 16]: max/mean abs error, relative L2
  camera        final camera latent
  anchors       octree anchor cells (integer cell of each anchor at level 8): multiset overlap fraction
  gaussians     elementwise attribute errors when the anchor cells coincide exactly; always:
                Chamfer distance on positions (mean NN distance both ways, in object units, object in [-0.5, 0.5]^3)
                NN-matched attribute errors (OTHER -> nearest REF Gaussian): |d colour|, |d opacity|,
                |d log scale|, rotation angle between unit quaternions (deg)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DumpReader  # noqa: E402


def err(a, b):
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return {'max_abs': float(d.max()), 'mean_abs': float(d.mean()), 'rel_l2': float((a - b).norm() / b.norm()),
            'bit_equal': bool(torch.equal(a, b))}


def gaussian_metrics(ra, rb):
    from scipy.spatial import cKDTree
    xa, xb = ra['xyz'].numpy().astype(np.float64), rb['xyz'].numpy().astype(np.float64)
    ta, tb = cKDTree(xa), cKDTree(xb)
    dba, iba = ta.query(xb, workers=-1)      # each B point -> nearest A
    dab, _ = tb.query(xa, workers=-1)
    out = {'chamfer_mean': float(0.5 * (dab.mean() + dba.mean())), 'nn_dist_p99': float(np.percentile(dba, 99)),
           'nn_dist_max': float(max(dab.max(), dba.max()))}
    m = lambda k: ra[k].reshape(ra[k].shape[0], -1)[iba]
    col = (rb['features_dc'].reshape(len(xb), -1) - m('features_dc')).abs()
    op = (rb['opacity'].reshape(-1) - m('opacity').reshape(-1)).abs()
    ls = (torch.log(rb['scaling']) - torch.log(m('scaling'))).abs()
    dot = (rb['rotation'] * m('rotation')).sum(-1).abs().clamp(max=1)
    ang = torch.rad2deg(2 * torch.acos(dot))
    for name, v in (('nn_colour_abs', col), ('nn_opacity_abs', op), ('nn_logscale_abs', ls), ('nn_rot_deg', ang)):
        out[name + '_mean'] = float(v.double().mean())
        out[name + '_p99'] = float(torch.quantile(v.double().flatten()[:2_000_000], 0.99))
    return out


def compare(ref_path, other_path):
    A, B = DumpReader(ref_path), DumpReader(other_path)
    res = {}
    for k in ('sample/final/latent', 'sample/final/camera'):
        if k in A and k in B:
            res[k.split('/')[-1]] = err(B.get(k), A.get(k))
    if 'decode/anchors' in A and 'decode/anchors' in B:
        ca = torch.floor(A.get('decode/anchors')[0] * 256).long()
        cb = torch.floor(B.get('decode/anchors')[0] * 256).long()
        key = lambda c: (c[:, 0] * 256 + c[:, 1]) * 256 + c[:, 2]
        ka, kb = torch.sort(key(ca)).values, torch.sort(key(cb)).values
        ua, na = torch.unique(ka, return_counts=True)
        ub, nb = torch.unique(kb, return_counts=True)
        common, ia, ib = np.intersect1d(ua.numpy(), ub.numpy(), return_indices=True)
        overlap = int(torch.minimum(na[ia], nb[ib]).sum())
        res['anchors'] = {'n': int(len(ka)), 'cell_multiset_overlap': overlap / len(ka),
                          'identical_cells': bool(torch.equal(ka, kb))}
    names = ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity')
    if not all(f'output/{n}' in A and f'output/{n}' in B for n in names):
        return res
    ra = {n: A.get(f'output/{n}').float() for n in names}
    rb = {n: B.get(f'output/{n}').float() for n in names}
    if ra['xyz'].shape == rb['xyz'].shape and res.get('anchors', {}).get('identical_cells'):
        res['gaussians_elementwise'] = {n: err(rb[n], ra[n]) for n in names}
    res['gaussians_set'] = gaussian_metrics(ra, rb)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ref')
    p.add_argument('other')
    p.add_argument('--json')
    a = p.parse_args()
    res = compare(a.ref, a.other)
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
