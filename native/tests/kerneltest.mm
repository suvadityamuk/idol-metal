// Byte-compare the native parity kernels against tensors of a reference dump (first DiT call, step 0).
// usage: kerneltest <idol_kernels.metal> <torch_metallib_dir> <dump.safetensors> <denoiser.safetensors>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "../src/ops.h"
#include "../src/safetensors.h"

using namespace idol;

static int failures = 0;

static DType dt_of(const std::string& s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  if (s == "I64") return DType::I64;
  throw std::runtime_error("dtype " + s);
}

static Tensor load(Context& c, const st::File& f, const std::string& k) {
  const auto& t = f.get(k);
  return c.from_host(dt_of(t.dtype), t.shape, t.data);
}

// fp32 weight converted to half like load_state_dict into fp16 parameters
static Tensor load_half(Context& c, const st::File& f, const std::string& k) {
  const auto& t = f.get(k);
  std::vector<uint16_t> h(t.numel());
  f32_to_f16((const float*)t.data, h.data(), h.size());
  return c.from_host(DType::F16, t.shape, h.data());
}

static void check(const char* name, const Tensor& got, const st::File& f, const std::string& k) {
  const auto& r = f.get(k);
  size_t es = dsize(got.dtype);
  size_t n = std::min<size_t>(got.numel(), r.numel());
  if ((size_t)got.numel() != (size_t)r.numel() || r.nbytes() != n * es) {
    printf("  %-34s SHAPE/DTYPE MISMATCH got %s %s ref %s %s\n", name, shape_str(got.shape).c_str(),
           dname(got.dtype), shape_str(r.shape).c_str(), r.dtype.c_str());
    failures++;
    return;
  }
  const uint8_t* g = (const uint8_t*)got.host();
  size_t nd = 0, first = SIZE_MAX;
  for (size_t i = 0; i < n; i++)
    if (memcmp(g + i * es, r.data + i * es, es)) {
      if (first == SIZE_MAX) first = i;
      nd++;
    }
  if (nd) {
    double a = 0, b = 0;
    if (got.dtype == DType::F16) { a = ((const __fp16*)g)[first]; b = ((const __fp16*)r.data)[first]; }
    else if (got.dtype == DType::F32) { a = ((const float*)g)[first]; b = ((const float*)r.data)[first]; }
    printf("  %-34s DIFF %zu/%zu (first %zu: got %.9g ref %.9g)\n", name, nd, n, first, a, b);
    failures++;
  } else {
    printf("  %-34s bit-identical (%zu elements)\n", name, n);
  }
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 5) {
      fprintf(stderr, "usage: %s kernels.metal torch_metallib_dir dump.safetensors denoiser.safetensors\n", argv[0]);
      return 2;
    }
    std::ifstream fs(argv[1]);
    std::stringstream ss;
    ss << fs.rdbuf();
    Context c(argv[2]);
    c.add_source_library(LIB_EXACT, ss.str(), false);
    st::File d(argv[3]), w(argv[4]);
    const std::string P = "dit/call000/";
    const uint32_t C = 1024;
    const int64_t L = 12294;

    // 1. LayerNorm32 (no affine, eps 1e-6), half -> half: blocks.0.norm1
    {
      Tensor x = load(c, d, P + "blocks.0.norm1/in0");
      Tensor y = c.empty(DType::F16, x.shape);
      auto p = c.pso(LIB_EXACT, "exact_ln_half_half");
      auto e = c.encoder();
      [e setComputePipelineState:p];
      set_buf(e, x, 0); set_buf(e, y, 1); set_buf(e, x, 2); set_buf(e, x, 3);
      float eps = 1e-6f; uint32_t flags = 0;
      set_val(e, C, 4); set_val(e, eps, 5); set_val(e, flags, 6);
      c.dispatch_groups(p, MTLSizeMake(x.numel() / C, 1, 1), MTLSizeMake(C / 4, 1, 1));
      c.synchronize();
      check("ln32 blocks.0.norm1", y, d, P + "blocks.0.norm1/out");
    }
    // 2. mod prep + LN + modulation: blocks.0 attn input
    Tensor modv;
    {
      Tensor tmod = load(c, d, P + "blocks.0/kw_mod");
      Tensor table = load(c, w, "blocks.0.shift_table");
      modv = c.empty(DType::F16, {6 * C});
      auto p = c.pso(LIB_EXACT, "exact_mod_prep");
      auto e = c.encoder();
      [e setComputePipelineState:p];
      set_buf(e, tmod, 0); set_buf(e, table, 1); set_buf(e, modv, 2); set_val(e, C, 3);
      c.dispatch_threads(p, 6 * C);
      Tensor x = load(c, d, P + "blocks.0/in0");
      Tensor y = c.empty(DType::F16, x.shape);
      auto p2 = c.pso(LIB_EXACT, "exact_resid_ln_mod");
      e = c.encoder();
      [e setComputePipelineState:p2];
      set_buf(e, x, 0); set_buf(e, x, 1); set_buf(e, x, 2); set_buf(e, x, 3);
      Tensor s1p = modv.view({6, C});
      [e setBuffer:modv.buf() offset:modv.offset + 1 * C * 2 atIndex:4];
      [e setBuffer:modv.buf() offset:modv.offset + 0 * C * 2 atIndex:5];
      set_buf(e, y, 6);
      float eps = 1e-6f; uint32_t mode = 0;
      set_val(e, C, 7); set_val(e, eps, 8); set_val(e, mode, 9);
      c.dispatch_groups(p2, MTLSizeMake(L, 1, 1), MTLSizeMake(C / 4, 1, 1));
      c.synchronize();
      check("ln32+mod blocks.0 (attn in)", y, d, P + "blocks.0.attn/in0");
    }
    // 3. LayerNorm32 affine (context refiner norm1), half -> half
    {
      Tensor x = load(c, d, P + "context_refiner.0.norm1/in0");
      Tensor y = c.empty(DType::F16, x.shape);
      Tensor wt = load(c, w, "context_refiner.0.norm1.weight"), bs = load(c, w, "context_refiner.0.norm1.bias");
      auto p = c.pso(LIB_EXACT, "exact_ln_half_half");
      auto e = c.encoder();
      [e setComputePipelineState:p];
      set_buf(e, x, 0); set_buf(e, y, 1); set_buf(e, wt, 2); set_buf(e, bs, 3);
      float eps = 1e-6f; uint32_t flags = 3;
      set_val(e, C, 4); set_val(e, eps, 5); set_val(e, flags, 6);
      c.dispatch_groups(p, MTLSizeMake(x.numel() / C, 1, 1), MTLSizeMake(C / 4, 1, 1));
      c.synchronize();
      check("ln32 affine context_refiner.0", y, d, P + "context_refiner.0.norm1/out");
    }
    // 4. RePo LayerNorm (affine, eps 1e-5), half -> float
    {
      Tensor x = load(c, d, P + "repo_layers.0/in0");
      Tensor y = c.empty(DType::F32, x.shape);
      Tensor wt = load(c, w, "repo_layers.0.norm.weight"), bs = load(c, w, "repo_layers.0.norm.bias");
      auto p = c.pso(LIB_EXACT, "exact_ln_half_float");
      auto e = c.encoder();
      [e setComputePipelineState:p];
      set_buf(e, x, 0); set_buf(e, y, 1); set_buf(e, wt, 2); set_buf(e, bs, 3);
      float eps = 1e-5f; uint32_t flags = 3;
      set_val(e, C, 4); set_val(e, eps, 5); set_val(e, flags, 6);
      c.dispatch_groups(p, MTLSizeMake(x.numel() / C, 1, 1), MTLSizeMake(C / 4, 1, 1));
      c.synchronize();
      check("repo LN affine f32", y, d, P + "repo_layers.0.norm/out");
      // 5. GEMMs (fp32, no bias) + silu*mul
      Tensor gw = load(c, w, "repo_layers.0.gate_map.weight"), cw = load(c, w, "repo_layers.0.content_map.weight");
      Tensor fw = load(c, w, "repo_layers.0.final_map.weight");
      Tensor g = linear_nd(c, y, gw, nullptr), ct = linear_nd(c, y, cw, nullptr);
      Tensor sm = c.empty(DType::F32, g.shape);
      auto p2 = c.pso(LIB_EXACT, "exact_silu_mul_f32");
      e = c.encoder();
      [e setComputePipelineState:p2];
      set_buf(e, g, 0); set_buf(e, ct, 1); set_buf(e, sm, 2);
      c.dispatch_threads(p2, g.numel());
      Tensor delta = linear_nd(c, sm, fw, nullptr);
      // 6. constants + rotary angles
      std::vector<float> f;
      std::vector<uint32_t> axis;
      for (int a = 0; a < 3; a++) {
        const auto& ft = w.get("repo_layers.0.freqs_" + std::to_string(a));
        for (int64_t j = 0; j < ft.numel(); j++) {
          f.push_back(((const float*)ft.data)[j]);
          axis.push_back(a);
        }
      }
      uint32_t NF = f.size(), H = 16;
      Tensor tf = c.from_host(DType::F32, {(int64_t)NF}, f.data());
      Tensor ta = c.from_host(DType::I32, {(int64_t)NF}, axis.data());
      Tensor tft = c.empty(DType::F32, {(int64_t)NF}), tfd = c.empty(DType::F32, {(int64_t)NF});
      auto p3 = c.pso(LIB_EXACT, "exact_repo_consts");
      e = c.encoder();
      [e setComputePipelineState:p3];
      set_buf(e, tf, 0); set_buf(e, tft, 1); set_buf(e, tfd, 2);
      c.dispatch_threads(p3, NF);
      Tensor fr = c.empty(DType::F32, {L, H, NF, 2});
      auto p4 = c.pso(LIB_EXACT, "exact_repo_freqs");
      e = c.encoder();
      [e setComputePipelineState:p4];
      set_buf(e, delta, 0); set_buf(e, tft, 1); set_buf(e, tfd, 2); set_buf(e, ta, 3); set_buf(e, fr, 4);
      set_val(e, H, 5); set_val(e, NF, 6);
      c.dispatch_threads(p4, (size_t)L * H * NF);
      c.synchronize();
      check("repo gate_map (GEMM f32)", g, d, P + "repo_layers.0.gate_map/out");
      check("repo silu*mul", sm, d, P + "repo_layers.0.final_map/in0");
      check("repo final_map (GEMM f32)", delta, d, P + "repo_layers.0.final_map/out");
      check("repo freqs (polar)", fr, d, P + "repo_layers.0/out");
    }
    // 7. q/k: rope + RMS norm from the qkv output and the reference rotary freqs
    {
      Tensor qkv = load(c, d, P + "blocks.0.attn.qkv/out");
      Tensor fr = load(c, d, P + "blocks.0/kw_rotary_emb");
      auto p = c.pso(LIB_EXACT, "exact_rope_rmsnorm");
      for (uint32_t which = 0; which < 2; which++) {
        Tensor gm = load(c, w, which ? "blocks.0.attn.k_norm.gamma" : "blocks.0.attn.q_norm.gamma");
        Tensor o = c.empty(DType::F16, {1, L, 16, 64});
        auto e = c.encoder();
        [e setComputePipelineState:p];
        set_buf(e, qkv, 0); set_buf(e, fr, 1); set_buf(e, gm, 2); set_buf(e, o, 3);
        simd_uint4 dims = {(uint32_t)L, 16, 64, which};
        simd_ulong2 ostr = {16 * 64, 64};
        float scale = 8.0f; uint32_t hr = 1;
        set_val(e, dims, 4); set_val(e, ostr, 5); set_val(e, scale, 6); set_val(e, hr, 7);
        c.dispatch_threads(p, (size_t)L * 16 * 4, 256);
        c.synchronize();
        check(which ? "rope+k_norm blocks.0" : "rope+q_norm blocks.0", o, d,
              P + (which ? "blocks.0.attn.k_norm/out" : "blocks.0.attn.q_norm/out"));
      }
    }
    // 8. GELU tanh (half)
    {
      Tensor x = load(c, d, P + "blocks.0.mlp.mlp.1/in0");
      Tensor y = c.empty(DType::F16, x.shape);
      auto p = c.pso(LIB_EXACT, "exact_gelu_tanh_half");
      auto e = c.encoder();
      [e setComputePipelineState:p];
      set_buf(e, x, 0); set_buf(e, y, 1);
      c.dispatch_threads(p, x.numel());
      c.synchronize();
      check("gelu_tanh half", y, d, P + "blocks.0.mlp.mlp.1/out");
    }
    // 9. linear fp16 + bias via the shared kernel object, then attention on the reference q/k/v
    {
      Tensor x = load(c, d, P + "blocks.0.attn/in0");
      Tensor wq = load_half(c, w, "blocks.0.attn.qkv.weight"), bq = load_half(c, w, "blocks.0.attn.qkv.bias");
      Tensor y = linear_nd(c, x, wq, &bq);
      Tensor q = load(c, d, P + "fn/sdpa04/in0"), k = load(c, d, P + "fn/sdpa04/in1"), v = load(c, d, P + "fn/sdpa04/in2");
      Tensor o = c.empty(DType::F16, {1, L, 16, 64});
      int64_t sb = L * 1024, sl = 1024, sh = 64;
      attention_prefill(c, {q, sb, sl, sh}, {k, sb, sl, sh}, {v, sb, sl, sh}, {o, sb, sl, sh}, 1, 16, 64, L, L);
      c.synchronize();
      check("linear qkv f16+bias", y, d, P + "blocks.0.attn.qkv/out");
      check("attention (torch MPP kernel)", o, d, P + "fn/sdpa04/out");
    }
    printf("%s (%d failing)\n", failures ? "FAIL" : "ALL BIT-IDENTICAL", failures);
  }
  return failures ? 1 : 0;
}
