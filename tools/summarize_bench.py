"""Benchmark table + closeness from runs/bench (tools/bench_all.sh) -> docs/benchmark.md and logs/bench_summary.json."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closeness import compare

REF_DUMP = {('robot', 262144): 'dumps/ref_robot_262144_A.safetensors', ('avocado', 262144): 'dumps/ref_avocado_262144_A.safetensors',
            ('robot', 1048576): 'dumps/ref_robot_1048576_A.safetensors', ('avocado', 1048576): 'dumps/ref_avocado_1048576_A.safetensors'}
REF_SHA = {}
for (img, ng), p in REF_DUMP.items():
    r = json.load(open(f'runs/ref_{img}_{ng}_A/result.json'))
    REF_SHA[(img, ng)] = r['runs'][0]['ply_sha256']
rows = []
for f in sorted(glob.glob('runs/bench/*/result.json')):
    tag, var, img, ng = os.path.basename(os.path.dirname(f)).split('_')
    r = json.load(open(f)); run = r['run']; t = run['times_s']
    row = {'round': tag, 'variant': var, 'image': img, 'num_gaussians': int(ng), 'load_s': r['load_s'], 'warmup_s': r['warmup_s'],
           'total_s': t['total'], 'sample_s': t['sample'], 'preprocess_s': t['preprocess'],
           'encode_s': t['encode/dinov3'] + t['encode/flux2_vae'],
           'decode_s': t['decode/octree'] + t['decode/gaussian_decoder'] + t['decode/to_representation'],
           'save_ply_s': t['save_ply'], 'peak_footprint_gb': run['peak_memory']['footprint_peak_gb'],
           'peak_mps_driver_gb': run['peak_memory']['mps_driver_peak_gb'],
           'ply_sha256': run['ply_sha256'], 'ply_identical_to_reference_dump_run': run['ply_sha256'] == REF_SHA[(img, int(ng))]}
    c = compare(REF_DUMP[(img, int(ng))], f'dumps/bench/{tag}_{var}_{img}_{ng}.safetensors')
    row['latent_rel_l2'] = c['latent']['rel_l2']; row['latent_max_abs'] = c['latent']['max_abs']; row['latent_mean_abs'] = c['latent']['mean_abs']
    row['camera_max_abs'] = c['camera']['max_abs']; row['anchor_overlap'] = c['anchors']['cell_multiset_overlap']
    row['chamfer'] = c['gaussians_set']['chamfer_mean']; row['nn_dist_p99'] = c['gaussians_set']['nn_dist_p99']
    rows.append(row)
json.dump(rows, open('logs/bench_summary.json', 'w'), indent=1)
L = ['| round | image | Gaussians | variant | total s | sample s | decode s | other s | load s | warm-up s | peak footprint GB | MPS driver GB | speed-up vs ref (total) | PLY == reference |',
     '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    ref = [x for x in rows if x['round'] == r['round'] and x['image'] == r['image'] and x['num_gaussians'] == r['num_gaussians'] and x['variant'] == 'ref']
    sp = ref[0]['total_s'] / r['total_s'] if ref else float('nan')
    other = r['total_s'] - r['sample_s'] - r['decode_s']
    L.append(f"| {r['round']} | {r['image']} | {r['num_gaussians']:,} | {r['variant']} | {r['total_s']:.1f} | {r['sample_s']:.1f} | {r['decode_s']:.2f} | {other:.2f} | {r['load_s']:.1f} | {r['warmup_s']:.1f} | {r['peak_footprint_gb']:.1f} | {r['peak_mps_driver_gb']:.1f} | {sp:.2f}x | {'byte-identical' if r['ply_identical_to_reference_dump_run'] else 'no'} |")
L += ['', '| round | image | Gaussians | variant | latent rel L2 | latent mean abs | latent max abs | camera max abs | anchor-cell overlap | Chamfer | NN dist p99 |', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    L.append(f"| {r['round']} | {r['image']} | {r['num_gaussians']:,} | {r['variant']} | {r['latent_rel_l2']:.2e} | {r['latent_mean_abs']:.2e} | {r['latent_max_abs']:.2e} | {r['camera_max_abs']:.2e} | {r['anchor_overlap']:.3f} | {r['chamfer']:.2e} | {r['nn_dist_p99']:.2e} |")
open('logs/bench_tables.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
