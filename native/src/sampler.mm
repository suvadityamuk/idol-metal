#include "sampler.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace idol {

namespace {

constexpr int64_t C = 1024, H = 16, D = 64, NQ = 8192, IN_CH = 16, CAM_CH = 5;
constexpr const char* LIB = "fast";  // kernel/idol_metal/idol_kernels.metal (registered by the caller)

DType dt_of(const std::string& s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  throw std::runtime_error("weights: unexpected dtype " + s);
}

struct W {
  Context* c;
  const st::File* f;
  Tensor f32(const std::string& k) const {
    const auto& t = f->get(k);
    if (t.dtype != "F32") throw std::runtime_error(k + ": expected F32");
    return c->from_host(DType::F32, t.shape, t.data);
  }
  // fp32 parameter stored as half, like convert_module_to_f16 + load_state_dict
  Tensor f16(const std::string& k) const {
    const auto& t = f->get(k);
    if (t.dtype == "F16") return c->from_host(DType::F16, t.shape, t.data);
    std::vector<uint16_t> h(t.numel());
    f32_to_f16((const float*)t.data, h.data(), h.size());
    return c->from_host(DType::F16, t.shape, h.data());
  }
};

struct Repo {
  Tensor nw, nb, wg, wc, wf, ft, fd, axis;
};

struct Block {
  bool modulated = true;
  Tensor table;                       // [6C] f32 (modulated)
  Tensor n1w, n1b, n2w, n2b;          // affine LN (context refiner)
  Tensor wqkv, bqkv, qg, kg, wo, bo, w1, b1, w2, b2;
  Repo repo;
};

}  // namespace

struct NativeDenoiser::Impl {
  Context& c;
  W w;
  Tensor in_w, in_b, ce1_w, ce1_b, ce2_w, ce2_b, pos_pe, pos_freqs;
  Tensor te_w1, te_b1, te_w2, te_b2, ada_w, ada_b, cam_w1, cam_b1, cam_w2, cam_b2;
  Tensor head_shift_table, out_w, out_b, camout_w, camout_b;
  std::vector<Block> noise, context, joint;
  Tensor ones_h;  // [C] half ones (gate for plain residuals: h * 1 == h exactly)

  Impl(Context& ctx, const st::File& f) : c(ctx), w{&ctx, &f} {}

  // -- primitives ---------------------------------------------------------------------------------------------
  void run1d(const char* fn, std::initializer_list<const Tensor*> bufs, size_t n) {
    auto p = c.pso(LIB, fn);
    auto e = c.encoder();
    [e setComputePipelineState:p];
    int i = 0;
    for (auto* t : bufs) set_buf(e, *t, i++);
    c.dispatch_threads(p, n);
  }
  Tensor cast_h(const Tensor& x) {
    Tensor y = c.empty(DType::F16, x.shape);
    run1d("cast_f32_f16", {&x, &y}, x.numel());
    return y;
  }
  Tensor cast_f(const Tensor& x) {
    Tensor y = c.empty(DType::F32, x.shape);
    run1d("cast_f16_f32", {&x, &y}, x.numel());
    return y;
  }
  Tensor add(const Tensor& a, const Tensor& b) {
    Tensor y = c.empty(a.dtype, a.shape);
    run1d(a.dtype == DType::F32 ? "add_f32" : "add_f16", {&a, &b, &y}, a.numel());
    return y;
  }
  void copy_rows(const Tensor& src, Tensor& dst, int64_t row0) {  // half rows of C into dst at row0
    Tensor d = dst.slice_rows(row0, src.numel() / C);
    run1d("copy_u16", {&src, &d}, src.numel());
  }
  // one-row fp32 linear (+ optional SiLU on the input, GELU on the output), as separate ops like the reference
  bool use_nd_gemv = getenv("IDOL_NATIVE_GEMV") == nullptr;
  Tensor gemv(const Tensor& x, const Tensor& Wt, const Tensor* b, uint32_t flags) {
    if (use_nd_gemv) {
      Tensor in = x.view({1, x.numel()});
      if (flags & 1u) {
        Tensor s = c.empty(DType::F32, in.shape);
        run1d("exact_silu_f32", {&in, &s}, in.numel());
        in = s;
      }
      Tensor y = linear_nd(c, in, Wt, b);
      if (flags & 2u) {
        Tensor g = c.empty(DType::F32, y.shape);
        run1d("exact_gelu_tanh_float", {&y, &g}, y.numel());
        y = g;
      }
      return y.view({y.numel()});
    }
    const uint32_t K = (uint32_t)Wt.shape[1], N = (uint32_t)Wt.shape[0];
    Tensor y = c.empty(DType::F32, {N});
    auto p = c.pso(LIB, "gemv_f32");
    auto e = c.encoder();
    [e setComputePipelineState:p];
    set_buf(e, x, 0); set_buf(e, Wt, 1); set_buf(e, b ? *b : Wt, 2); set_buf(e, y, 3);
    uint32_t fl = flags | (b ? 4u : 0u);
    set_val(e, K, 4); set_val(e, N, 5); set_val(e, fl, 6);
    c.dispatch_groups(p, MTLSizeMake(N, 1, 1), MTLSizeMake(32, 1, 1));
    return y;
  }
  // LayerNorm over rows of C: TI -> TO, flags bit0 weight, bit1 bias (exact_ln)
  Tensor ln(const Tensor& x, DType to, const Tensor* wt, const Tensor* bs, float eps) {
    Tensor y = c.empty(to, x.shape);
    std::string fn = std::string("exact_ln_") + (x.dtype == DType::F16 ? "half_" : "float_") +
                     (to == DType::F16 ? "half" : "float");
    auto p = c.pso(LIB, fn);
    auto e = c.encoder();
    [e setComputePipelineState:p];
    set_buf(e, x, 0); set_buf(e, y, 1); set_buf(e, wt ? *wt : x, 2); set_buf(e, bs ? *bs : x, 3);
    uint32_t CC = C, fl = (wt ? 1u : 0u) | (bs ? 2u : 0u);
    set_val(e, CC, 4); set_val(e, eps, 5); set_val(e, fl, 6);
    c.dispatch_groups(p, MTLSizeMake(x.numel() / C, 1, 1), MTLSizeMake(C / 4, 1, 1));
    return y;
  }
  // fused residual + LN32 + modulation (+ RePo LN): returns x', y, ry
  void resid_ln_mod(const Tensor& x, const Tensor* h, const Tensor* gate, const Tensor& s1p, const Tensor& sh,
                    const Repo* repo, Tensor& xo, Tensor& y, Tensor& ry) {
    uint32_t mode = (h ? 1u : 0u) | (repo ? 2u : 0u);
    xo = h ? c.empty(DType::F16, x.shape) : x;
    y = c.empty(DType::F16, x.shape);
    ry = repo ? c.empty(DType::F32, x.shape) : Tensor{};
    auto p = c.pso(LIB, "fused_resid_ln_mod_repo");
    auto e = c.encoder();
    [e setComputePipelineState:p];
    set_buf(e, x, 0); set_buf(e, h ? *h : x, 1); set_buf(e, gate ? *gate : s1p, 2); set_buf(e, xo, 3);
    set_buf(e, s1p, 4); set_buf(e, sh, 5); set_buf(e, y, 6);
    set_buf(e, repo ? repo->nw : s1p, 7); set_buf(e, repo ? repo->nb : s1p, 8); set_buf(e, repo ? ry : y, 9);
    uint32_t CC = C;
    float e1 = 1e-6f, e2 = 1e-5f;
    set_val(e, CC, 10); set_val(e, e1, 11); set_val(e, e2, 12); set_val(e, mode, 13);
    c.dispatch_groups(p, MTLSizeMake(x.numel() / C, 1, 1), MTLSizeMake(C / 4, 1, 1));
  }
  Tensor repo_delta(const Repo& r, const Tensor& rln) {
    Tensor g = linear_nd(c, rln, r.wg, nullptr), ct = linear_nd(c, rln, r.wc, nullptr);
    Tensor a = c.empty(DType::F32, g.shape);
    run1d("exact_silu_mul_f32", {&g, &ct, &a}, g.numel());
    return linear_nd(c, a, r.wf, nullptr);
  }
  void rope_qk(const Block& b, const Tensor& qkv, const Tensor& delta, int64_t L, Tensor& q, Tensor& k) {
    q = c.empty(DType::F16, {L, H, D});
    k = c.empty(DType::F16, {L, H, D});
    auto p = c.pso(LIB, "fused_repo_rope_qk");
    auto e = c.encoder();
    [e setComputePipelineState:p];
    set_buf(e, qkv, 0); set_buf(e, delta, 1); set_buf(e, b.repo.ft, 2); set_buf(e, b.repo.fd, 3);
    set_buf(e, b.repo.axis, 4); set_buf(e, b.qg, 5); set_buf(e, b.kg, 6); set_buf(e, q, 7); set_buf(e, k, 8);
    uint32_t R = (uint32_t)L, HH = H;
    float sc = 8.0f;
    set_val(e, R, 9); set_val(e, sc, 10); set_val(e, HH, 11);
    c.dispatch_threads(p, (size_t)L * H * 4, 256);
  }
  Tensor attention(const Tensor& q, const Tensor& k, const Tensor& qkv, int64_t L, const Tensor* mask) {
    Tensor o = c.empty(DType::F16, {L, C});  // written straight in [L, H*D] layout (no permute copy)
    Tensor v = qkv;
    v.offset += 2 * C * 2;                   // qkv [L, 3, H, D]: v at +2*C elements in each row
    const int64_t s = H * D;
    attention_prefill(c, {q, L * s, s, D}, {k, L * s, s, D}, {v, L * 3 * s, 3 * s, D}, {o, L * s, s, D}, 1, H, D,
                      L, L, mask);
    return o;
  }
  Tensor linear_gelu(const Tensor& x, const Tensor& Wt, const Tensor& b) {
    const uint32_t M = (uint32_t)(x.numel() / x.shape.back()), K = (uint32_t)Wt.shape[1], N = (uint32_t)Wt.shape[0];
    Tensor y = c.empty(DType::F16, {M, N});
    auto p = c.pso(LIB, "linear_bias_gelu_f16");
    auto e = c.encoder();
    [e setComputePipelineState:p];
    set_buf(e, x, 0); set_buf(e, Wt, 1); set_buf(e, y, 2); set_buf(e, b, 3);
    set_val(e, M, 4); set_val(e, N, 5); set_val(e, K, 6);
    c.dispatch_groups(p, MTLSizeMake((N + 127) / 128, (M + 63) / 64, 1), MTLSizeMake(128, 1, 1));
    return y;
  }
  Tensor resid(const Tensor& x, const Tensor& h, const Tensor& gate) {
    Tensor y = c.empty(DType::F16, x.shape);
    auto p = c.pso(LIB, "exact_resid");
    auto e = c.encoder();
    [e setComputePipelineState:p];
    set_buf(e, x, 0); set_buf(e, h, 1); set_buf(e, gate, 2); set_buf(e, y, 3);
    uint32_t CC = C;
    set_val(e, CC, 4);
    c.dispatch_threads(p, x.numel());
    return y;
  }

  // -- blocks -------------------------------------------------------------------------------------------------
  // modulated blocks over x [L, C] half; mods[i] = [6C] half with the scale chunks already 1 + scale
  Tensor run_modulated(std::vector<Block>& blocks, const std::vector<Tensor>& mods, Tensor x, const Tensor* mask) {
    const int64_t L = x.numel() / C;
    Tensor pend_x, pend_h, pend_g;
    for (size_t i = 0; i < blocks.size(); i++) {
      Block& b = blocks[i];
      const Tensor& m = mods[i];
      auto chunk = [&](int j) { Tensor t = m; t.offset += j * C * 2; t.shape = {C}; return t; };
      Tensor sh1 = chunk(0), sc1 = chunk(1), g1 = chunk(2), sh2 = chunk(3), sc2 = chunk(4), g2 = chunk(5);
      Tensor xo, y, ry;
      if (!pend_x.defined()) resid_ln_mod(x, nullptr, nullptr, sc1, sh1, &b.repo, xo, y, ry);
      else resid_ln_mod(pend_x, &pend_h, &pend_g, sc1, sh1, &b.repo, xo, y, ry);
      Tensor delta = repo_delta(b.repo, ry);
      Tensor qkv = linear_nd(c, y, b.wqkv, &b.bqkv);
      Tensor q, k;
      rope_qk(b, qkv, delta, L, q, k);
      Tensor o = attention(q, k, qkv, L, mask);
      Tensor ha = linear_nd(c, o, b.wo, &b.bo);
      Tensor x1, y2, unused;
      resid_ln_mod(xo, &ha, &g1, sc2, sh2, nullptr, x1, y2, unused);
      Tensor hm = linear_nd(c, linear_gelu(y2, b.w1, b.b1), b.w2, &b.b2);
      pend_x = x1; pend_h = hm; pend_g = g2;
      c.commit();
    }
    return resid(pend_x, pend_h, pend_g);
  }
  // context refiner block (no modulation, affine LayerNorm32)
  Tensor run_context_block(Block& b, const Tensor& x) {
    const int64_t L = x.numel() / C;
    Tensor rl = ln(x, DType::F32, &b.repo.nw, &b.repo.nb, 1e-5f);
    Tensor delta = repo_delta(b.repo, rl);
    Tensor h = ln(x, DType::F16, &b.n1w, &b.n1b, 1e-6f);
    Tensor qkv = linear_nd(c, h, b.wqkv, &b.bqkv);
    Tensor q, k;
    rope_qk(b, qkv, delta, L, q, k);
    Tensor o = attention(q, k, qkv, L, nullptr);
    Tensor x1 = resid(x, linear_nd(c, o, b.wo, &b.bo), ones_h);
    Tensor h2 = ln(x1, DType::F16, &b.n2w, &b.n2b, 1e-6f);
    Tensor hm = linear_nd(c, linear_gelu(h2, b.w1, b.b1), b.w2, &b.b2);
    Tensor out = resid(x1, hm, ones_h);
    c.commit();
    return out;
  }

  // -- loading ------------------------------------------------------------------------------------------------
  Repo load_repo(const std::string& pfx) {
    Repo r;
    r.nw = w.f32(pfx + ".norm.weight");
    r.nb = w.f32(pfx + ".norm.bias");
    r.wg = w.f32(pfx + ".gate_map.weight");
    r.wc = w.f32(pfx + ".content_map.weight");
    r.wf = w.f32(pfx + ".final_map.weight");
    std::vector<float> f;
    std::vector<int32_t> axis;
    for (int a = 0; a < 3; a++) {
      const auto& t = w.f->get(pfx + ".freqs_" + std::to_string(a));
      for (int64_t j = 0; j < t.numel(); j++) {
        f.push_back(((const float*)t.data)[j]);
        axis.push_back(a);
      }
    }
    if (f.size() != 32) throw std::runtime_error("RePo: expected 32 frequencies");
    Tensor tf = c.from_host(DType::F32, {32}, f.data());
    r.axis = c.from_host(DType::I32, {32}, axis.data());
    r.ft = c.empty(DType::F32, {32});
    r.fd = c.empty(DType::F32, {32});
    run1d("exact_repo_consts", {&tf, &r.ft, &r.fd}, 32);
    return r;
  }
  Block load_block(const std::string& pfx, const std::string& repo_pfx, bool modulated) {
    Block b;
    b.modulated = modulated;
    if (modulated) {
      b.table = w.f32(pfx + ".shift_table");
    } else {
      b.n1w = w.f32(pfx + ".norm1.weight"); b.n1b = w.f32(pfx + ".norm1.bias");
      b.n2w = w.f32(pfx + ".norm2.weight"); b.n2b = w.f32(pfx + ".norm2.bias");
    }
    b.wqkv = w.f16(pfx + ".attn.qkv.weight"); b.bqkv = w.f16(pfx + ".attn.qkv.bias");
    b.qg = w.f32(pfx + ".attn.q_norm.gamma"); b.kg = w.f32(pfx + ".attn.k_norm.gamma");
    b.wo = w.f16(pfx + ".attn.out.weight"); b.bo = w.f16(pfx + ".attn.out.bias");
    b.w1 = w.f16(pfx + ".mlp.mlp.0.weight"); b.b1 = w.f16(pfx + ".mlp.mlp.0.bias");
    b.w2 = w.f16(pfx + ".mlp.mlp.2.weight"); b.b2 = w.f16(pfx + ".mlp.mlp.2.bias");
    b.repo = load_repo(repo_pfx);
    return b;
  }
  void load() {
    in_w = w.f32("input_layer.weight"); in_b = w.f32("input_layer.bias");
    ce1_w = w.f32("cond_embedder.weight"); ce1_b = w.f32("cond_embedder.bias");
    ce2_w = w.f32("cond_embedder2.weight"); ce2_b = w.f32("cond_embedder2.bias");
    pos_pe = w.f32("pos_pe");
    te_w1 = w.f32("t_embedder.mlp.0.weight"); te_b1 = w.f32("t_embedder.mlp.0.bias");
    te_w2 = w.f32("t_embedder.mlp.2.weight"); te_b2 = w.f32("t_embedder.mlp.2.bias");
    ada_w = w.f32("adaLN_modulation.1.weight"); ada_b = w.f32("adaLN_modulation.1.bias");
    cam_w1 = w.f32("cam_refiner.mlp.0.weight"); cam_b1 = w.f32("cam_refiner.mlp.0.bias");
    cam_w2 = w.f32("cam_refiner.mlp.2.weight"); cam_b2 = w.f32("cam_refiner.mlp.2.bias");
    head_shift_table = w.f32("shift_table");
    out_w = w.f32("out_layer.weight"); out_b = w.f32("out_layer.bias");
    camout_w = w.f32("cam_out_layer.weight"); camout_b = w.f32("cam_out_layer.bias");
    for (int i = 0; i < 2; i++) {
      noise.push_back(load_block("noise_refiner." + std::to_string(i), "noise_repo_layers." + std::to_string(i), true));
      context.push_back(
          load_block("context_refiner." + std::to_string(i), "context_repo_layers." + std::to_string(i), false));
    }
    for (int i = 0; i < 24; i++)
      joint.push_back(load_block("blocks." + std::to_string(i), "repo_layers." + std::to_string(i), true));
    // PcdAbsolutePositionEmbedder frequencies (CPU float32 in the reference, as here)
    const int F = (int)(C / 3 / 2), max_res = 16, res_dim = std::max(0, F - max_res);
    std::vector<float> fr;
    for (int j = 0; j < max_res && (int)fr.size() < F; j++) fr.push_back(std::exp2f((float)j));
    for (int j = 0; j < res_dim && (int)fr.size() < F; j++)
      fr.push_back(std::exp2f((float)j / (float)res_dim * (float)max_res));
    pos_freqs = c.from_host(DType::F32, {F}, fr.data());
    std::vector<uint16_t> one(C, 0x3C00);
    ones_h = c.from_host(DType::F16, {C}, one.data());
    c.synchronize();
  }
};

NativeDenoiser::NativeDenoiser(Context& c, const st::File& weights) : p_(new Impl(c, weights)) { p_->load(); }
NativeDenoiser::~NativeDenoiser() { delete p_; }

SampleResult NativeDenoiser::sample(const float* feature1, const float* feature2, int64_t Lc,
                                    const float* noise_latent, const float* noise_camera, const SampleConfig& cfg) {
  Impl& P = *p_;
  Context& c = P.c;
  SampleResult res;
  auto t0 = std::chrono::steady_clock::now();
  // ---- per request: context streams (cond, zeroed cond), constant positions ----
  auto context = [&](const float* f1, const float* f2) {
    Tensor a = c.from_host(DType::F32, {Lc, 1280}, f1), b = c.from_host(DType::F32, {Lc, 128}, f2);
    Tensor h = P.cast_h(P.add(linear_nd(c, a, P.ce1_w, &P.ce1_b), linear_nd(c, b, P.ce2_w, &P.ce2_b)));
    for (auto& blk : P.context) h = P.run_context_block(blk, h);
    return h;
  };
  Tensor h_cond = context(feature1, feature2);
  std::vector<float> z1(Lc * 1280, 0.f), z2(Lc * 128, 0.f);
  Tensor h_neg_full = context(z1.data(), z2.data());
  Tensor pos = c.empty(DType::F32, {NQ, C});
  {
    auto p = c.pso(LIB, "pos_embed_f32");
    auto e = c.encoder();
    [e setComputePipelineState:p];
    set_buf(e, P.pos_pe, 0); set_buf(e, P.pos_freqs, 1); set_buf(e, pos, 2);
    uint32_t F = (uint32_t)P.pos_freqs.numel(), CC = C;
    set_val(e, F, 3); set_val(e, CC, 4);
    [e dispatchThreads:MTLSizeMake(C, NQ, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  Tensor pos_h = P.cast_h(pos);
  c.synchronize();
  // collapsed unconditional context: all rows identical (checked), two copies with fp16 log-weight biases
  {
    const uint16_t* r = h_neg_full.ptr<uint16_t>();
    for (int64_t i = 1; i < Lc; i++)
      if (memcmp(r, r + i * C, C * 2)) throw std::runtime_error("unconditional context rows differ");
  }
  std::vector<float> bias;
  {
    double rem = (double)Lc;
    for (int i = 0; i < 2; i++) {
      float b = (float)(__fp16)(float)std::log(rem);
      if (i == 0 && (double)b > std::log(rem)) {
        __fp16 h = (__fp16)b;
        uint16_t u;
        memcpy(&u, &h, 2);
        u -= 1;  // one fp16 ulp down (positive value)
        memcpy(&h, &u, 2);
        b = (float)h;
      }
      bias.push_back(b);
      rem -= std::exp((double)b);
    }
  }
  Tensor h_neg = c.empty(DType::F16, {2, C});
  P.copy_rows(h_neg_full.slice_rows(0, 1), h_neg, 0);
  P.copy_rows(h_neg_full.slice_rows(0, 1), h_neg, 1);
  const int64_t Lcond = NQ + Lc + 1, Lu = NQ + 2 + 1;
  std::vector<uint16_t> mh(Lu, 0);
  for (int i = 0; i < 2; i++) {
    __fp16 v = (__fp16)bias[i];
    memcpy(&mh[NQ + i], &v, 2);
  }
  Tensor mask = c.from_host(DType::F16, {Lu}, mh.data());
  c.synchronize();
  auto t1 = std::chrono::steady_clock::now();
  res.seconds_prepare = std::chrono::duration<double>(t1 - t0).count();

  // ---- sampling loop ----
  std::vector<float> lat(noise_latent, noise_latent + NQ * IN_CH), cam(noise_camera, noise_camera + CAM_CH);
  std::vector<double> ts(cfg.steps + 1);
  for (int i = 0; i <= cfg.steps; i++) {
    double t = (double)i * (-1.0 / cfg.steps) + 1.0;  // np.linspace(1, 0, steps + 1): i * step + start, end exact
    if (i == cfg.steps) t = 0.0;
    ts[i] = cfg.rescale_t * t / (1 + (cfg.rescale_t - 1) * t);
  }
  const float half_dim = 128;
  const int s_end = cfg.run_steps < 0 ? cfg.steps : std::min(cfg.steps, cfg.start_step + cfg.run_steps);
  for (int s = cfg.start_step; s < s_end; s++) {
    @autoreleasepool {
      const double t = ts[s], tp = ts[s + 1];
      const float tt = (float)(1000 * t);
      // time embedding: [cos(t f), sin(t f)], f = exp(-ln(1e4) * j / 128) (float32, as the reference's CPU freqs)
      std::vector<float> fr(128);
      for (int j = 0; j < 128; j++) fr[j] = std::exp(-(float)std::log(10000.0) * (float)j / half_dim);
      Tensor ft = c.from_host(DType::F32, {128}, fr.data());
      Tensor e0 = c.empty(DType::F32, {256});
      {
        auto p = c.pso(LIB, "temb_cos_sin");
        auto e = c.encoder();
        [e setComputePipelineState:p];
        set_buf(e, ft, 0); set_buf(e, e0, 1);
        uint32_t hd = 128;
        set_val(e, tt, 2); set_val(e, hd, 3);
        c.dispatch_threads(p, 128);
      }
      Tensor t_emb = P.gemv(P.gemv(e0, P.te_w1, &P.te_b1, 0), P.te_w2, &P.te_b2, 1u /*silu in*/);
      Tensor t_mod = P.cast_h(P.gemv(t_emb, P.ada_w, &P.ada_b, 1u));
      // per-block mods: t_mod + table.half(), 1 + scale chunks
      auto mods_for = [&](std::vector<Block>& bl) {
        std::vector<Tensor> out;
        for (auto& b : bl) {
          Tensor m = c.empty(DType::F16, {6 * C});
          auto p = c.pso(LIB, "exact_mod_prep");
          auto e = c.encoder();
          [e setComputePipelineState:p];
          set_buf(e, t_mod, 0); set_buf(e, b.table, 1); set_buf(e, m, 2);
          uint32_t CC = C;
          set_val(e, CC, 3);
          c.dispatch_threads(p, 6 * C);
          out.push_back(m);
        }
        return out;
      };
      auto mods_noise = mods_for(P.noise), mods_joint = mods_for(P.joint);
      Tensor z = c.from_host(DType::F32, {NQ, IN_CH}, lat.data());
      Tensor h_x = P.add(P.cast_h(linear_nd(c, z, P.in_w, &P.in_b)), pos_h);
      h_x = P.run_modulated(P.noise, mods_noise, h_x, nullptr);
      Tensor camt = c.from_host(DType::F32, {CAM_CH}, cam.data());
      Tensor h_cam = P.cast_h(P.gemv(P.gemv(camt, P.cam_w1, &P.cam_b1, 2u /*gelu out*/), P.cam_w2, &P.cam_b2, 0));
      std::vector<float> vl[2], vcam[2];
      for (int pass = 0; pass < 2; pass++) {
        const Tensor& hc = pass == 0 ? h_cond : h_neg;
        const int64_t L = pass == 0 ? Lcond : Lu;
        Tensor h = c.empty(DType::F16, {L, C});
        P.copy_rows(h_x, h, 0);
        P.copy_rows(hc, h, NQ);
        P.copy_rows(h_cam, h, L - 1);
        h = P.run_modulated(P.joint, mods_joint, h, pass == 0 ? nullptr : &mask);
        // head
        Tensor shift = P.add(P.head_shift_table.view({2, C}).slice_rows(0, 1).view({C}), t_emb);
        Tensor scale = P.add(P.head_shift_table.view({2, C}).slice_rows(1, 1).view({C}), t_emb);
        auto head = [&](const Tensor& rows, int64_t n) {
          Tensor x = P.ln(P.cast_f(rows), DType::F32, nullptr, nullptr, 1e-5f);
          Tensor y = c.empty(DType::F32, x.shape);
          auto p = c.pso(LIB, "mod_f32");
          auto e = c.encoder();
          [e setComputePipelineState:p];
          set_buf(e, x, 0); set_buf(e, scale, 1); set_buf(e, shift, 2); set_buf(e, y, 3);
          uint32_t CC = C;
          set_val(e, CC, 4);
          c.dispatch_threads(p, x.numel());
          return y;
        };
        Tensor hx = head(h.slice_rows(0, NQ), NQ);
        Tensor out = linear_nd(c, hx, P.out_w, &P.out_b);
        Tensor hcm = head(h.slice_rows(L - 1, 1), 1);
        Tensor ocam = P.gemv(hcm, P.camout_w, &P.camout_b, 0);
        c.synchronize();
        vl[pass].assign(out.ptr<float>(), out.ptr<float>() + NQ * IN_CH);
        vcam[pass].assign(ocam.ptr<float>(), ocam.ptr<float>() + CAM_CH);
      }
      if (s == cfg.start_step) { res.v_first[0] = vl[0]; res.v_first[1] = vl[1]; }
      // CFG + Euler, float32 with the reference's operation order
      const float a = (float)(1 + cfg.cfg), b = (float)cfg.cfg, dt = (float)(t - tp);
      for (size_t i = 0; i < lat.size(); i++) {
        float v1 = a * vl[0][i], v2 = b * vl[1][i];
        float v = v1 - v2;
        float d = v * dt;
        lat[i] = lat[i] - d;
      }
      for (int i = 0; i < CAM_CH; i++) {
        float v1 = a * vcam[0][i], v2 = b * vcam[1][i];
        float v = v1 - v2;
        float d = v * dt;
        cam[i] = cam[i] - d;
      }
    }
  }
  res.seconds_sampling = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  res.latent = lat;
  res.camera = cam;
  return res;
}

}  // namespace idol
