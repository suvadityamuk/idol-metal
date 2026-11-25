// Idol Metal kernels.
//
// Parity kernels ("exact_*") reproduce, bit for bit, the arithmetic PyTorch 2.15's MPS backend performs for the
// op chains of the Idol model (LayerNorm32 + adaLN modulation, gated residuals, RePo rotary angles, complex RoPE,
// q/k RMS normalisation, GELU-tanh, SiLU). They are written to be compiled at runtime with
// MTLMathModeSafe + precise math functions, which is what PyTorch's own shaders get from `-fno-fast-math`:
// within one C expression the compiler may contract a*b+c into an FMA (as PyTorch's kernels may), across
// statements it may not. Where PyTorch stores an intermediate to memory (a separate kernel), these kernels round
// to the same dtype at the same point and keep the next operation in a separate statement.
//
// Portions mirror PyTorch's MPS kernels (aten/src/ATen/native/mps/kernels/{LayerNorm,ReduceOps,ActivationKernel,
// UnaryKernel,BinaryKernel}.metal and c10/metal/reduction_utils.h), BSD-3-Clause, Copyright (c) Meta Platforms, Inc.
// and affiliates.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// The rounding points below only hold without fast-math reassociation or FMA contraction. The kernel-builder
// compiles .metal sources with the compiler's default (fast) math, so pin safe math and contraction within one
// expression only (FP_CONTRACT ON, as PyTorch's own shaders get), whatever the build flags; precise:: / fast::
// functions are named explicitly where PyTorch uses them.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(on)

constant constexpr uint SIMD = 32;

// c10::metal::threadgroup_sum for float: simd_sum per simdgroup, then simd_sum of the partials in simdgroup 0.
inline float tg_sum(threadgroup float* data, float val, uint idx, uint size) {
  float rc = simd_sum(val);
  if (idx % SIMD == 0) {
    data[idx / SIMD] = rc;
  }
  if (size > SIMD) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (idx < ((size + SIMD - 1) / SIMD)) {
      float rc1 = simd_sum(data[idx]);
      if (idx == 0) {
        data[0] = rc1;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return data[0];
}

// LayerNorm statistics exactly as layer_norm_single_row<float> (N_READS = 4, one threadgroup per row,
// lsize = ceil(C / 4) threads): returns mean and inv_std; vals holds this thread's 4 inputs as float.
struct LnStats {
  float mean;
  float inv_std;
};

inline LnStats ln_stats(thread float* vals, int count, threadgroup float* sh_sum, threadgroup float* sh_sq,
                        uint tid, uint lsize, uint C, float eps) {
  float partial_sum = 0.0f;
  for (int i = 0; i < 4; i++) {
    partial_sum += vals[i];
  }
  float mean = tg_sum(sh_sum, partial_sum, tid, lsize) / float(C);
  float partial_sum_sq = 0.0f;
  for (int i = 0; i < 4; i++) {
    float d = (i < count) ? (vals[i] - mean) : 0.0f;
    partial_sum_sq += d * d;
  }
  float var = tg_sum(sh_sq, partial_sum_sq, tid, lsize) / float(C);
  float inv_std = precise::rsqrt(var + eps);
  return {mean, inv_std};
}

// ---------------------------------------------------------------------------------------------------------------
// exact_ln: y = LayerNorm(x.float(), eps, weight?, bias?) with the output cast to TO.
//   flags bit0: weight present, bit1: bias present
// Mirrors LayerNorm32 (x.float() -> F.layer_norm -> .type(x.dtype)) and F.layer_norm on fp32 rows.
template <typename TI, typename TO>
kernel void exact_ln(device const TI* x [[buffer(0)]], device TO* y [[buffer(1)]],
                     device const float* weight [[buffer(2)]], device const float* bias [[buffer(3)]],
                     constant uint& C [[buffer(4)]], constant float& eps [[buffer(5)]],
                     constant uint& flags [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]], uint lsize [[threads_per_threadgroup]]) {
  threadgroup float sh_sum[SIMD];
  threadgroup float sh_sq[SIMD];
  const ulong base = ulong(row) * C;
  uint lane0 = tid * 4;
  int count = min(4, max(0, int(C) - int(lane0)));
  float vals[4];
  for (int i = 0; i < 4; i++) {
    vals[i] = (i < count) ? float(x[base + lane0 + i]) : 0.0f;
  }
  LnStats s = ln_stats(vals, count, sh_sum, sh_sq, tid, lsize, C, eps);
  for (int i = 0; i < count; i++) {
    float norm = (vals[i] - s.mean) * s.inv_std;
    if (flags & 1u) norm *= weight[lane0 + i];
    if (flags & 2u) norm += bias[lane0 + i];
    y[base + lane0 + i] = static_cast<TO>(norm);
  }
}

// exact_ln_mod: h = LayerNorm32(x) (half), then h * (1 + scale) + shift in half, as the modulated blocks do:
//   h = norm(x); h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
// scale1p is the half tensor (1 + scale) the reference materialises (computed by exact_one_plus); shift half.
// Optional residual: x_in + h_in * gate (half) first (the previous gated residual), written to x_out.
//   mode 0: x = xin;  mode 1: x = xin + hin * gate (x written to xout)
kernel void exact_resid_ln_mod(device const half* xin [[buffer(0)]], device const half* hin [[buffer(1)]],
                               device const half* gate [[buffer(2)]], device half* xout [[buffer(3)]],
                               device const half* scale1p [[buffer(4)]], device const half* shift [[buffer(5)]],
                               device half* y [[buffer(6)]], constant uint& C [[buffer(7)]],
                               constant float& eps [[buffer(8)]], constant uint& mode [[buffer(9)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint lsize [[threads_per_threadgroup]]) {
  threadgroup float sh_sum[SIMD];
  threadgroup float sh_sq[SIMD];
  const ulong base = ulong(row) * C;
  uint lane0 = tid * 4;
  int count = min(4, max(0, int(C) - int(lane0)));
  float vals[4];
  for (int i = 0; i < 4; i++) {
    if (i < count) {
      half xv = xin[base + lane0 + i];
      if (mode == 1u) {
        half hg = hin[base + lane0 + i] * gate[lane0 + i];
        xv = xv + hg;
        xout[base + lane0 + i] = xv;
      }
      vals[i] = float(xv);
    } else {
      vals[i] = 0.0f;
    }
  }
  LnStats s = ln_stats(vals, count, sh_sum, sh_sq, tid, lsize, C, eps);
  for (int i = 0; i < count; i++) {
    float norm = (vals[i] - s.mean) * s.inv_std;
    half hn = static_cast<half>(norm);
    half hm = hn * scale1p[lane0 + i];
    half ho = hm + shift[lane0 + i];
    y[base + lane0 + i] = ho;
  }
}

// exact_resid: x + h * gate (half), the block's final gated residual when no LayerNorm follows.
kernel void exact_resid(device const half* xin [[buffer(0)]], device const half* hin [[buffer(1)]],
                        device const half* gate [[buffer(2)]], device half* xout [[buffer(3)]],
                        constant uint& C [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  half hg = hin[i] * gate[i % C];
  xout[i] = xin[i] + hg;
}

// ---------------------------------------------------------------------------------------------------------------
// Elementwise ops with PyTorch's functors.

// silu_functor: x / (1 + precise::exp(-x)) in float.
inline float silu_f(float xf) {
  return xf / (1.0f + precise::exp(-xf));
}

// RePo: act(gate_map(h)) * content_map(h), fp32 (two separate PyTorch kernels: silu, then mul).
kernel void exact_silu_mul_f32(device const float* g [[buffer(0)]], device const float* c [[buffer(1)]],
                               device float* y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float s = silu_f(g[i]);
  y[i] = s * c[i];
}

kernel void exact_silu_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
  y[i] = silu_f(x[i]);
}

// gelu_tanh_functor (approximate='tanh'), half: tanh is fast::tanh on a clamped argument; float: precise tanh.
template <typename T>
inline float gelu_tanh_dispatch(float x);
template <>
inline float gelu_tanh_dispatch<float>(float x) {
  return precise::tanh(x);
}
template <>
inline float gelu_tanh_dispatch<half>(float x) {
  return fast::tanh(clamp(x, -10.0f, 10.0f));
}

template <typename T>
kernel void exact_gelu_tanh(device const T* x [[buffer(0)]], device T* y [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
  const float xf = float(x[i]);
  constexpr float kBeta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
  constexpr float kKappa = 0.044715f;
  const float inner = kBeta * (xf + kKappa * xf * xf * xf);
  y[i] = static_cast<T>(0.5f * xf * (1.0f + gelu_tanh_dispatch<T>(inner)));
}

// ---------------------------------------------------------------------------------------------------------------
// RePo rotary angles -> (cos, sin), from final_map's output delta [R, H*3] (fp32):
//   ang_a = clamp_mul(delta[..., a], freqs_a) * pi,  clamp_mul(x, f) = x * tanh(f) + x * (f - tanh(f))
//   freqs = polar(1, cat(ang_0, ang_1, ang_2)) = (1 * cos(ang), 1 * sin(ang))
// ft and fd hold tanh(f) and f - tanh(f) for the concatenated 32 frequencies (computed once with the same ops);
// axis[j] says which delta component frequency j uses. Output [R, H, 32, 2] fp32 (cos, sin) pairs.
kernel void exact_repo_freqs(device const float* delta [[buffer(0)]], device const float* ft [[buffer(1)]],
                             device const float* fd [[buffer(2)]], device const uint* axis [[buffer(3)]],
                             device float2* out [[buffer(4)]], constant uint& H [[buffer(5)]],
                             constant uint& NF [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  uint j = i % NF;
  uint rh = i / NF;  // row * H + head
  float x = delta[rh * 3 + axis[j]];
  float a = x * ft[j];
  float b = x * fd[j];
  float s = a + b;
  float ang = s * M_PI_F;
  float c = precise::cos(ang);
  float sn = precise::sin(ang);
  float re = 1.0f * c;
  float im = 1.0f * sn;
  out[i] = float2(re, im);
}

// ---------------------------------------------------------------------------------------------------------------
// q/k path of RopeMultiHeadAttention:
//   q = q_norm(apply_rotary_emb(q, freqs)):
//     x = q.float() viewed as complex pairs; x * freqs (complex mul, c10::metal::mul); .type_as(half)
//     F.normalize(x.float(), dim=-1): x / clamp_min(||x||_2, 1e-12), the norm by reduction_innermost_chunk
//       (4 lanes per 64-wide row, interleaved quads, (l0 + l2) + (l1 + l3), precise::sqrt)
//     * gamma[h, d] (fp32) * scale (fp32) -> half
// One thread per (row, head, lane in 0..3); each lane owns quads lane, lane+4, lane+8, lane+12 of its row.
// qkv: [R, 3, H, D] half (the qkv linear output); which = 0 (q) or 1 (k); freqs [R, H, D/2] float2 or null.
// Output: o[r, h, d] at o + r * o_rs + h * o_hs + d.
inline float c_mul_re(float2 x, float2 y) {
  return x.x * y.x - x.y * y.y;
}
inline float c_mul_im(float2 x, float2 y) {
  return x.x * y.y + x.y * y.x;
}

kernel void exact_rope_rmsnorm(device const half* qkv [[buffer(0)]], device const float2* freqs [[buffer(1)]],
                               device const float* gamma [[buffer(2)]], device half* o [[buffer(3)]],
                               constant uint4& dims [[buffer(4)]],      // R, H, D, which
                               constant ulong2& ostr [[buffer(5)]],     // row stride, head stride (elements)
                               constant float& scale [[buffer(6)]], constant uint& has_rope [[buffer(7)]],
                               uint gid [[thread_position_in_grid]], uint lane_in_simd [[thread_index_in_simdgroup]]) {
  const uint R = dims.x, H = dims.y, D = dims.z, which = dims.w;
  const uint lane = gid & 3u;
  const uint rh = gid >> 2;
  if (rh >= R * H) {
    return;  // whole quads of lanes exit together
  }
  const uint r = rh / H, h = rh % H;
  device const half* src = qkv + (ulong(r) * 3 + which) * H * D + ulong(h) * D;
  // rotated (and rounded to half) values for this lane's 16 elements
  half rv[16];
  for (uint qi = 0; qi < 4; qi++) {
    uint q = lane + 4 * qi;
    for (uint c = 0; c < 4; c += 2) {
      uint d = 4 * q + c;
      float2 x = float2(float(src[d]), float(src[d + 1]));
      if (has_rope) {
        float2 f = freqs[(ulong(r) * H + h) * (D / 2) + d / 2];
        float re = c_mul_re(x, f);
        float im = c_mul_im(x, f);
        rv[4 * qi + c] = half(re);
        rv[4 * qi + c + 1] = half(im);
      } else {
        rv[4 * qi + c] = half(x.x);
        rv[4 * qi + c + 1] = half(x.y);
      }
    }
  }
  // SumOp<LOAD_SQUARE>: load_val (r * r) and combine (a + b) are separate expressions in PyTorch's source, so
  // they are never contracted into an FMA; keep them as separate statements here.
  float acc = 0.0f;
  for (uint k = 0; k < 16; k++) {
    float v = float(rv[k]);
    float sq = v * v;
    acc = acc + sq;
  }
  float val = (acc + 0.0f) + (0.0f + 0.0f);
  val = val + simd_shuffle_down(val, ushort(2));
  val = val + simd_shuffle_down(val, ushort(1));
  float nrm = precise::sqrt(val);
  nrm = simd_shuffle(nrm, ushort(lane_in_simd & ~3u));
  float den = max(nrm, 1e-12f);
  device half* dst = o + ulong(r) * ostr.x + ulong(h) * ostr.y;
  for (uint qi = 0; qi < 4; qi++) {
    uint q = lane + 4 * qi;
    for (uint c = 0; c < 4; c++) {
      uint d = 4 * q + c;
      float xn = float(rv[4 * qi + c]) / den;
      float xg = xn * gamma[h * D + d];
      float xs = xg * scale;
      dst[d] = half(xs);
    }
  }
}

// one_plus: 1 + x in half (the (1 + scale) tensor of adaLN).
kernel void exact_one_plus_h(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  y[i] = half(1.0h) + x[i];
}


// RePo constants: tanh(f) (tanh_functor: precise::tanh) and f - tanh(f) (sub kernel), for the 32 frequencies.
kernel void exact_repo_consts(device const float* f [[buffer(0)]], device float* ft [[buffer(1)]],
                              device float* fd [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float t = precise::tanh(f[i]);
  ft[i] = t;
  fd[i] = f[i] - t;
}

// adaLN chunk prep for a modulated block: mod = t_mod + shift_table.type(half) (cast, then half add), then
// (1 + scale) for the two scale chunks. Layout of the 6C vector: shift_msa, scale_msa, gate_msa, shift_mlp,
// scale_mlp, gate_mlp. out holds the same layout with the scale chunks replaced by 1 + scale.
kernel void exact_mod_prep(device const half* tmod [[buffer(0)]], device const float* table [[buffer(1)]],
                           device half* out [[buffer(2)]], constant uint& C [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  half st = static_cast<half>(table[i]);
  half m = tmod[i] + st;
  uint chunk = i / C;
  if (chunk == 1u || chunk == 4u) {
    m = half(1.0h) + m;
  }
  out[i] = m;
}

// ---------------------------------------------------------------------------------------------------------------
// Fused kernels of the fast build. Each keeps the rounding points of the op chain it replaces (values PyTorch
// would store to memory as half/float are rounded to that type in registers here), so the result equals the
// unfused chain; only the memory traffic and the number of dispatches change.

// fused_resid_ln_mod_repo: one pass over a row of x (half, C wide), one threadgroup of C/4 threads per row.
//   mode bit0: x = xin + hin * gate (the previous gated residual, written to xout); else x = xin
//   y  = LayerNorm32(x, eps_ln) * scale1p + shift (half)           [exact_resid_ln_mod]
//   mode bit1: ry = LayerNorm(x.float(), eps_repo, rw, rb) (float)  [exact_ln<half,float>, RePo's norm]
//   mode bit2: y = LayerNorm32(x, eps_ln, rw, rb) (affine, half) instead of the modulated y  [exact_ln<half,half>]
//   mode bit3: residual without gate (x = xin + hin)
// Both norms use the same statistics code, so the mean and variance are computed once.
kernel void fused_resid_ln_mod_repo(device const half* xin [[buffer(0)]], device const half* hin [[buffer(1)]],
                                    device const half* gate [[buffer(2)]], device half* xout [[buffer(3)]],
                                    device const half* scale1p [[buffer(4)]], device const half* shift [[buffer(5)]],
                                    device half* y [[buffer(6)]], device const float* rw [[buffer(7)]],
                                    device const float* rb [[buffer(8)]], device float* ry [[buffer(9)]],
                                    constant uint& C [[buffer(10)]], constant float& eps_ln [[buffer(11)]],
                                    constant float& eps_repo [[buffer(12)]], constant uint& mode [[buffer(13)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint lsize [[threads_per_threadgroup]]) {
  threadgroup float sh_sum[SIMD];
  threadgroup float sh_sq[SIMD];
  const ulong base = ulong(row) * C;
  uint lane0 = tid * 4;
  int count = min(4, max(0, int(C) - int(lane0)));
  float vals[4];
  for (int i = 0; i < 4; i++) {
    if (i < count) {
      half xv = xin[base + lane0 + i];
      if (mode & 1u) {
        half hg = hin[base + lane0 + i] * gate[lane0 + i];
        xv = xv + hg;
        xout[base + lane0 + i] = xv;
      } else if (mode & 8u) {
        xv = xv + hin[base + lane0 + i];
        xout[base + lane0 + i] = xv;
      }
      vals[i] = float(xv);
    } else {
      vals[i] = 0.0f;
    }
  }
  // ln_stats with eps folded out, so one reduction serves both epsilons
  float partial_sum = 0.0f;
  for (int i = 0; i < 4; i++) {
    partial_sum += vals[i];
  }
  float mean = tg_sum(sh_sum, partial_sum, tid, lsize) / float(C);
  float partial_sum_sq = 0.0f;
  for (int i = 0; i < 4; i++) {
    float d = (i < count) ? (vals[i] - mean) : 0.0f;
    partial_sum_sq += d * d;
  }
  float var = tg_sum(sh_sq, partial_sum_sq, tid, lsize) / float(C);
  float inv_std = precise::rsqrt(var + eps_ln);
  if (mode & 4u) {
    for (int i = 0; i < count; i++) {
      float norm = (vals[i] - mean) * inv_std;
      norm *= rw[lane0 + i];
      norm += rb[lane0 + i];
      y[base + lane0 + i] = static_cast<half>(norm);
    }
    return;
  }
  for (int i = 0; i < count; i++) {
    float norm = (vals[i] - mean) * inv_std;
    half hn = static_cast<half>(norm);
    half hm = hn * scale1p[lane0 + i];
    half ho = hm + shift[lane0 + i];
    y[base + lane0 + i] = ho;
  }
  if (mode & 2u) {
    float inv_std_r = precise::rsqrt(var + eps_repo);
    for (int i = 0; i < count; i++) {
      float norm = (vals[i] - mean) * inv_std_r;
      norm *= rw[lane0 + i];
      norm += rb[lane0 + i];
      ry[base + lane0 + i] = norm;
    }
  }
}

// fused_repo_rope_qk: RePo angles (exact_repo_freqs) computed in registers, then RoPE + RMSNorm for q AND k
// (exact_rope_rmsnorm twice). One thread per (row, head, lane 0..3); lane owns quads lane, lane+4, lane+8, lane+12.
//   qkv [R, 3, H, D] half; delta [R, H*3] float (final_map output); ft/fd/axis: 32-entry RePo constants;
//   qg/kg [H, D] float gammas; q/k out [R, H, D] half. Head dim must be 64 (4 lanes x 16 values).
kernel void fused_repo_rope_qk(device const half* qkv [[buffer(0)]], device const float* delta [[buffer(1)]],
                               device const float* ft [[buffer(2)]], device const float* fd [[buffer(3)]],
                               device const uint* axis [[buffer(4)]], device const float* qg [[buffer(5)]],
                               device const float* kg [[buffer(6)]], device half* qo [[buffer(7)]],
                               device half* ko [[buffer(8)]], constant uint& R [[buffer(9)]],
                               constant float& scale [[buffer(10)]], constant uint& H [[buffer(11)]],
                               uint gid [[thread_position_in_grid]], uint lane_in_simd [[thread_index_in_simdgroup]]) {
  constexpr uint D = 64;  // RePo splits the head into 10 + 10 + 12 frequency pairs: head dim 64 only
  const uint lane = gid & 3u;
  const uint rh = gid >> 2;
  if (rh >= R * H) {
    return;
  }
  const uint r = rh / H, h = rh % H;
  float2 f[8];
  for (uint qi = 0; qi < 4; qi++) {
    uint q = lane + 4 * qi;
    for (uint c = 0; c < 4; c += 2) {
      uint j = (4 * q + c) / 2;
      float x = delta[rh * 3 + axis[j]];
      float a = x * ft[j];
      float b = x * fd[j];
      float s = a + b;
      float ang = s * M_PI_F;
      float cs = precise::cos(ang);
      float sn = precise::sin(ang);
      float re = 1.0f * cs;
      float im = 1.0f * sn;
      f[2 * qi + c / 2] = float2(re, im);
    }
  }
  for (uint which = 0; which < 2; which++) {
    device const half* src = qkv + (ulong(r) * 3 + which) * H * D + ulong(h) * D;
    device const float* gamma = which ? kg : qg;
    device half* dst = (which ? ko : qo) + (ulong(r) * H + h) * D;
    half rv[16];
    for (uint qi = 0; qi < 4; qi++) {
      uint q = lane + 4 * qi;
      for (uint c = 0; c < 4; c += 2) {
        uint d = 4 * q + c;
        float2 x = float2(float(src[d]), float(src[d + 1]));
        float2 fr = f[2 * qi + c / 2];
        float re = c_mul_re(x, fr);
        float im = c_mul_im(x, fr);
        rv[4 * qi + c] = half(re);
        rv[4 * qi + c + 1] = half(im);
      }
    }
    float acc = 0.0f;
    for (uint k = 0; k < 16; k++) {
      float v = float(rv[k]);
      float sq = v * v;
      acc = acc + sq;
    }
    float val = (acc + 0.0f) + (0.0f + 0.0f);
    val = val + simd_shuffle_down(val, ushort(2));
    val = val + simd_shuffle_down(val, ushort(1));
    float nrm = precise::sqrt(val);
    nrm = simd_shuffle(nrm, ushort(lane_in_simd & ~3u));
    float den = max(nrm, 1e-12f);
    for (uint qi = 0; qi < 4; qi++) {
      uint q = lane + 4 * qi;
      for (uint c = 0; c < 4; c++) {
        uint d = 4 * q + c;
        float xn = float(rv[4 * qi + c]) / den;
        float xg = xn * gamma[h * D + d];
        float xs = xg * scale;
        dst[d] = half(xs);
      }
    }
  }
}

// rmsnorm_heads: F.normalize(x.float(), dim=-1) * gamma[h] * scale -> half for rows of H heads x 64 (no RoPE;
// exact_rope_rmsnorm with has_rope = 0). src row r, head h at src + r * row_stride + h * 64; out [R, H, 64].
kernel void rmsnorm_heads_f16(device const half* src [[buffer(0)]], device const float* gamma [[buffer(1)]],
                              device half* o [[buffer(2)]], constant uint& R [[buffer(3)]], constant uint& H [[buffer(4)]],
                              constant uint& row_stride [[buffer(5)]], constant float& scale [[buffer(6)]],
                              uint gid [[thread_position_in_grid]], uint lane_in_simd [[thread_index_in_simdgroup]]) {
  constexpr uint D = 64;
  const uint lane = gid & 3u;
  const uint rh = gid >> 2;
  if (rh >= R * H) {
    return;
  }
  const uint r = rh / H, h = rh % H;
  device const half* s = src + ulong(r) * row_stride + ulong(h) * D;
  half rv[16];
  for (uint qi = 0; qi < 4; qi++) {
    uint q = lane + 4 * qi;
    for (uint c = 0; c < 4; c++) {
      rv[4 * qi + c] = s[4 * q + c];
    }
  }
  float acc = 0.0f;
  for (uint k = 0; k < 16; k++) {
    float v = float(rv[k]);
    float sq = v * v;
    acc = acc + sq;
  }
  float val = (acc + 0.0f) + (0.0f + 0.0f);
  val = val + simd_shuffle_down(val, ushort(2));
  val = val + simd_shuffle_down(val, ushort(1));
  float nrm = precise::sqrt(val);
  nrm = simd_shuffle(nrm, ushort(lane_in_simd & ~3u));
  float den = max(nrm, 1e-12f);
  device half* dst = o + (ulong(r) * H + h) * D;
  for (uint qi = 0; qi < 4; qi++) {
    uint q = lane + 4 * qi;
    for (uint c = 0; c < 4; c++) {
      uint d = 4 * q + c;
      float xn = float(rv[4 * qi + c]) / den;
      float xg = xn * gamma[h * D + d];
      float xs = xg * scale;
      dst[d] = half(xs);
    }
  }
}

// gelu_tanh in place on half (exact_gelu_tanh<half> arithmetic), 4 elements per thread.
kernel void fused_gelu_tanh_half_inplace(device half* x [[buffer(0)]], constant uint& n [[buffer(1)]],
                                         uint i [[thread_position_in_grid]]) {
  for (uint e = i * 4; e < min(n, i * 4 + 4); e++) {
    const float xf = float(x[e]);
    constexpr float kBeta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
    constexpr float kKappa = 0.044715f;
    const float inner = kBeta * (xf + kKappa * xf * xf * xf);
    x[e] = static_cast<half>(0.5f * xf * (1.0f + gelu_tanh_dispatch<half>(inner)));
  }
}

// linear_bias_gelu_f16: Y = gelu_tanh(half(X W^T + bias)) with X [M, K] half, W [N, K] half, bias [N] half,
// on MetalPerformancePrimitives matmul2d (Metal 4). The fp32 accumulator plus float(bias), rounded once to half, is
// what PyTorch's MPS F.linear returns (byte-compared on the model's shapes); GELU is then applied to that half value
// exactly as exact_gelu_tanh<half> does, while the tile is still in registers. One threadgroup (4 simdgroups) per
// 64 x 128 output tile.
template <int TM, int TN>
[[kernel, max_total_threads_per_threadgroup(128)]] void linear_bias_gelu_f16(
    device half* X [[buffer(0)]], device half* W [[buffer(1)]], device half* Y [[buffer(2)]],
    device const half* bias [[buffer(3)]], constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(K, M));
  auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(W, dextents<int32_t, 2>(K, N));
  auto C = tensor<device half, dextents<int32_t, 2>, tensor_inline>(Y, dextents<int32_t, 2>(N, M));
  const int r0 = tg.y * TM, c0 = tg.x * TN;
  auto mA = A.slice(0, r0);
  auto mB = B.slice(0, c0);
  auto mC = C.slice(c0, r0);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  for (ushort i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) cT[i] = 0;
  }
  op.run(mA, mB, cT);
  auto cH = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
  for (ushort i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto idx = cT.get_multidimensional_index(i);
      const int c = min(c0 + int(idx[0]), int(N) - 1);
      half h = half(cT[i] + float(bias[c]));
      const float xf = float(h);
      constexpr float kBeta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
      constexpr float kKappa = 0.044715f;
      const float inner = kBeta * (xf + kKappa * xf * xf * xf);
      cH[i] = static_cast<half>(0.5f * xf * (1.0f + gelu_tanh_dispatch<half>(inner)));
    }
  }
  cH.store(mC);
}
template [[host_name("linear_bias_gelu_f16")]] [[kernel]] decltype(linear_bias_gelu_f16<64, 128>)
    linear_bias_gelu_f16<64, 128>;

// ---------------------------------------------------------------------------------------------------------------
// Small helpers used by the native runner (scripts/run_native.sh) for the denoiser's non-block glue.
kernel void cast_f32_f16(device const float* x [[buffer(0)]], device half* y [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
  y[i] = half(x[i]);
}
kernel void add_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                    device float* y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  y[i] = a[i] + b[i];
}
kernel void add_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* y [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  y[i] = a[i] + b[i];
}
kernel void copy_u16(device const ushort* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
  y[i] = x[i];
}
// y = act_out(W act_in(x) + b) for one row: x [K], W [N, K], b [N] (or none), float. One simdgroup per output.
//   flags bit0: SiLU on the input, bit1: GELU-tanh (float) on the output, bit2: bias present
kernel void gemv_f32(device const float* x [[buffer(0)]], device const float* W [[buffer(1)]],
                     device const float* b [[buffer(2)]], device float* y [[buffer(3)]], constant uint& K [[buffer(4)]],
                     constant uint& N [[buffer(5)]], constant uint& flags [[buffer(6)]],
                     uint n [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (n >= N) return;
  float acc = 0.0f;
  for (uint k = lane; k < K; k += 32) {
    float xv = x[k];
    if (flags & 1u) xv = silu_f(xv);
    acc += W[ulong(n) * K + k] * xv;
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (flags & 4u) acc += b[n];
    if (flags & 2u) {
      constexpr float kBeta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
      constexpr float kKappa = 0.044715f;
      const float inner = kBeta * (acc + kKappa * acc * acc * acc);
      acc = 0.5f * acc * (1.0f + precise::tanh(inner));
    }
    y[n] = acc;
  }
}
// PcdAbsolutePositionEmbedder: out = cat(sin(o), cos(o), -1) with o = outer(pos.reshape(-1), freqs) * 2 * pi,
// reshaped to [R, 6F] (per axis a: F sines then F cosines), zero-padded to C. pos [R, 3], freqs [F], float.
kernel void pos_embed_f32(device const float* pos [[buffer(0)]], device const float* freqs [[buffer(1)]],
                          device float* out [[buffer(2)]], constant uint& F [[buffer(3)]],
                          constant uint& C [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const uint r = g.y, c = g.x;
  if (c >= C) return;
  float v = 0.0f;
  if (c < 6 * F) {
    const uint a = c / (2 * F), w = c % (2 * F), j = w % F;
    const float o1 = pos[r * 3 + a] * freqs[j];
    const float o2 = o1 * 2.0f;
    const float o3 = o2 * M_PI_F;
    v = (w < F) ? precise::sin(o3) : precise::cos(o3);
  }
  out[ulong(r) * C + c] = v;
}
// y = x * (1 + scale) + shift on float rows of C (scale/shift [C]); (1 + scale), the product and the sum are
// separate tensors in the reference, so they are separate statements here.
kernel void mod_f32(device const float* x [[buffer(0)]], device const float* scale [[buffer(1)]],
                    device const float* shift [[buffer(2)]], device float* y [[buffer(3)]],
                    constant uint& C [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  const uint c = i % C;
  const float s1 = 1.0f + scale[c];
  const float m = x[i] * s1;
  y[i] = m + shift[c];
}
// SinusoidalEmbedder input: [cos(t f_j), sin(t f_j)] for j < half (float, precise functions like PyTorch's MPS
// unary kernels).
kernel void temb_cos_sin(device const float* freqs [[buffer(0)]], device float* out [[buffer(1)]],
                         constant float& t [[buffer(2)]], constant uint& half_dim [[buffer(3)]],
                         uint j [[thread_position_in_grid]]) {
  if (j >= half_dim) return;
  const float a = t * freqs[j];
  out[j] = precise::cos(a);
  out[half_dim + j] = precise::sin(a);
}
// half rows -> float (the head's .type(float32))
kernel void cast_f16_f32(device const half* x [[buffer(0)]], device float* y [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
  y[i] = float(x[i]);
}

// ---------------------------------------------------------------------------------------------------------------
// instantiations
template [[host_name("exact_ln_half_half")]] kernel void exact_ln<half, half>(
    device const half*, device half*, device const float*, device const float*, constant uint&, constant float&,
    constant uint&, uint, uint, uint);
template [[host_name("exact_ln_half_float")]] kernel void exact_ln<half, float>(
    device const half*, device float*, device const float*, device const float*, constant uint&, constant float&,
    constant uint&, uint, uint, uint);
template [[host_name("exact_ln_float_float")]] kernel void exact_ln<float, float>(
    device const float*, device float*, device const float*, device const float*, constant uint&,
    constant float&, constant uint&, uint, uint, uint);
template [[host_name("exact_gelu_tanh_half")]] kernel void exact_gelu_tanh<half>(device const half*, device half*,
                                                                                uint);
template [[host_name("exact_gelu_tanh_float")]] kernel void exact_gelu_tanh<float>(device const float*,
                                                                                  device float*, uint);
