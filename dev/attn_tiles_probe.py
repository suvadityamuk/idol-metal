"""Re-instantiate torch's MPP flash-attention template (ATen MppAttention.h) with other tile shapes and time it."""
import os, re, struct, time, math, sys
import torch, torch.nn.functional as F
INC = os.path.join(os.path.dirname(torch.__file__), 'include')
seen = set()
def inline(path):
    out = []
    for line in open(path):
        m = re.match(r'\s*#include\s*[<"]((?:c10|ATen)/[^>"]+)[>"]', line)
        if m:
            p = os.path.join(INC, m.group(1))
            if p not in seen:
                seen.add(p); out.append(inline(p))
            continue
        out.append(line)
    return ''.join(out)
src = '#include <metal_stdlib>\nusing namespace metal;\n' + inline(os.path.join(INC, 'c10/metal/utils.h')) + inline(os.path.join(INC, 'ATen/native/mps/kernels/MppAttention.h'))
src = src.replace('#pragma once', '').replace('instantiate_mpp_attn_mask(float16, half);', '').replace('instantiate_mpp_attn_mask(bfloat16, bfloat);', '')
CFG = [(64, 32, 4), (64, 64, 4), (64, 128, 4), (128, 32, 8), (128, 64, 8), (32, 32, 2), (32, 64, 2)]
for bq, bk, wm in CFG:
    src += f'\ninstantiate_mpp_attn_arch("a10", true, float16, half, {bq}, {bk}, 64, {wm}, 1, 0, 0, float16, half);'
    src += f'\ninstantiate_mpp_attn_arch("a10", true, float16, half, {bq}, {bk}, 64, {wm}, 1, 1, 0, float16, half);'
lib = torch.mps.compile_shader(src)
def params(L, Lk, qs, ks, vs, os_):
    b = struct.pack('6i2f', 1, 16, 64, L, Lk, 1, 1 / 8, 1.0) + struct.pack('12q', *qs, *ks, *vs, *os_)
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).to('mps')
def run(name, bq, wm, q, k, v, mask=None):
    L, Lk = q.shape[2], k.shape[2]
    o = torch.empty_like(q)
    st = lambda t: (t.stride(0), t.stride(1), t.stride(2))
    pr = params(L, Lk, st(q), st(k), st(v), st(o))
    mp = torch.zeros(32, dtype=torch.uint8, device='mps') if mask is None else torch.frombuffer(bytearray(struct.pack('4q', *mask.stride())), dtype=torch.uint8).to('mps')
    mk = mask if mask is not None else torch.zeros(4, device='mps', dtype=torch.float16)
    fn = getattr(lib, name)
    call = lambda: fn(q, k, v, o, pr, mp, mk, threads=[((L + bq - 1) // bq) * 32, 16 * wm, 1], group_size=[32, wm, 1])
    return o, call
for L in (12294, 8195):
    q = torch.randn(1, 16, L, 64, device='mps', dtype=torch.float16); k, v = torch.randn_like(q), torch.randn_like(q)
    ref = F.scaled_dot_product_attention(q, k, v)
    for bq, bk, wm in CFG:
        name = f'prefill_attention_mpp_a10_float16_bq{bq}_bk{bk}_bd64_wm{wm}_wn1_hm0_dc0_maskfloat16'
        try:
            o, call = run(name, bq, wm, q, k, v)
            call(); torch.mps.synchronize(); t = time.perf_counter()
            for _ in range(10): call()
            torch.mps.synchronize(); dt = (time.perf_counter() - t) / 10
            print(f'L={L} bq{bq} bk{bk} wm{wm}: {dt*1e3:6.2f} ms  {4*16*L*L*64/dt/1e12:5.1f} TFLOP/s  max|d| vs torch {(o-ref).abs().max().item():.2e}  equal {torch.equal(o, ref)}', flush=True)
        except Exception as e:
            print(name, 'failed', str(e)[:300])
    t = time.perf_counter()
    for _ in range(10): F.scaled_dot_product_attention(q, k, v)
    torch.mps.synchronize(); print(f'L={L} torch sdpa {(time.perf_counter()-t)/10*1e3:.2f} ms')
