// Torch (MPS) bindings for the Idol Metal kernels. Every op encodes on PyTorch's own MPS command buffer, so it is
// ordered with the surrounding framework ops; outputs are caller-allocated.
#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <optional>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

namespace {

inline id<MTLBuffer> mtlBuffer(const torch::Tensor &t) { return __builtin_bit_cast(id<MTLBuffer>, t.storage().data()); }
inline NSUInteger mtlOffset(const torch::Tensor &t) { return (NSUInteger)(t.storage_offset() * t.element_size()); }

id<MTLComputePipelineState> pipeline(const char *name) {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  static id<MTLLibrary> library = nil;
  static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *cache = [NSMutableDictionary new];
  @synchronized(cache) {
    NSString *key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> pso = cache[key];
    if (pso) return pso;
    NSError *error = nil;
    if (!library) {
      library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
      TORCH_CHECK(library, "Failed to create Metal library from embedded data: ",
                  error.localizedDescription.UTF8String);
    }
    id<MTLFunction> fn = [library newFunctionWithName:key];
    TORCH_CHECK(fn, "Metal function not found: ", name);
    pso = [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, error.localizedDescription.UTF8String);
    cache[key] = pso;
    return pso;
  }
}

template <class F> void encodePass(const char *kernel, F encode) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = pipeline(kernel);
    dispatch_sync(torch::mps::get_dispatch_queue(), ^() {
      id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
      TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");
      id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute command encoder");
      [enc setComputePipelineState:pso];
      encode(enc, pso);
      [enc endEncoding];
      torch::mps::commit();
    });
  }
}

inline void bind(id<MTLComputeCommandEncoder> enc, const torch::Tensor &t, NSUInteger index) {
  [enc setBuffer:mtlBuffer(t) offset:mtlOffset(t) atIndex:index];
}
template <class T> inline void bindVal(id<MTLComputeCommandEncoder> enc, T v, NSUInteger index) {
  [enc setBytes:&v length:sizeof(T) atIndex:index];
}

void check(const torch::Tensor &t, const char *name, torch::ScalarType dtype) {
  TORCH_CHECK(t.device().is_mps(), name, " must be an MPS tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ", dtype, ", got ", t.scalar_type());
}
void checkNumel(const torch::Tensor &t, const char *name, int64_t n) {
  TORCH_CHECK(t.numel() == n, name, " must have ", n, " elements, got ", t.numel());
}

}  // namespace

// y = LayerNorm32(x', eps) * scale1p + shift (half), x' = x + h * gate when h is given (written to xout);
// ry = LayerNorm(x'.float(), eps_repo) * repo_weight + repo_bias (float) when repo_weight is given.
void resid_ln_mod(torch::Tensor &y, const std::optional<torch::Tensor> &xout, const std::optional<torch::Tensor> &ry,
                  const torch::Tensor &x, const std::optional<torch::Tensor> &h,
                  const std::optional<torch::Tensor> &gate, const torch::Tensor &scale1p, const torch::Tensor &shift,
                  const std::optional<torch::Tensor> &repo_weight, const std::optional<torch::Tensor> &repo_bias,
                  double eps, double eps_repo) {
  check(x, "x", torch::kHalf);
  TORCH_CHECK(x.dim() >= 1 && x.numel() > 0, "x must be a non-empty [..., C] tensor");
  const int64_t C = x.size(-1), R = x.numel() / C;
  TORCH_CHECK(C % 4 == 0 && C / 4 <= 1024, "C must be a multiple of 4 and at most 4096, got ", C);
  TORCH_CHECK(R < INT32_MAX, "too many rows");
  check(y, "y", torch::kHalf);
  TORCH_CHECK(y.sizes() == x.sizes(), "y must have the shape of x");
  check(scale1p, "scale1p", torch::kHalf);
  checkNumel(scale1p, "scale1p", C);
  check(shift, "shift", torch::kHalf);
  checkNumel(shift, "shift", C);
  uint32_t mode = 0;
  TORCH_CHECK(h.has_value() == gate.has_value() && h.has_value() == xout.has_value(),
              "h, gate and xout go together");
  if (h) {
    check(*h, "h", torch::kHalf);
    TORCH_CHECK(h->sizes() == x.sizes(), "h must have the shape of x");
    check(*gate, "gate", torch::kHalf);
    checkNumel(*gate, "gate", C);
    check(*xout, "xout", torch::kHalf);
    TORCH_CHECK(xout->sizes() == x.sizes(), "xout must have the shape of x");
    mode |= 1u;
  }
  TORCH_CHECK(repo_weight.has_value() == repo_bias.has_value() && repo_weight.has_value() == ry.has_value(),
              "repo_weight, repo_bias and ry go together");
  if (repo_weight) {
    check(*repo_weight, "repo_weight", torch::kFloat);
    checkNumel(*repo_weight, "repo_weight", C);
    check(*repo_bias, "repo_bias", torch::kFloat);
    checkNumel(*repo_bias, "repo_bias", C);
    check(*ry, "ry", torch::kFloat);
    TORCH_CHECK(ry->sizes() == x.sizes(), "ry must have the shape of x");
    mode |= 2u;
  }
  const float eps_f = (float)eps, eps_r = (float)eps_repo;
  encodePass("fused_resid_ln_mod_repo", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    bind(enc, x, 0);
    bind(enc, h ? *h : x, 1);
    bind(enc, gate ? *gate : scale1p, 2);
    bind(enc, xout ? *xout : y, 3);
    bind(enc, scale1p, 4);
    bind(enc, shift, 5);
    bind(enc, y, 6);
    bind(enc, repo_weight ? *repo_weight : scale1p, 7);
    bind(enc, repo_bias ? *repo_bias : scale1p, 8);
    bind(enc, ry ? *ry : y, 9);
    bindVal<uint32_t>(enc, (uint32_t)C, 10);
    bindVal<float>(enc, eps_f, 11);
    bindVal<float>(enc, eps_r, 12);
    bindVal<uint32_t>(enc, mode, 13);
    [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(C / 4, 1, 1)];
  });
}

// RePo rotary angles from delta (final_map output) + RoPE + RMSNorm(gamma * scale) for q and k (head dim 64).
void repo_rope_qk(torch::Tensor &q, torch::Tensor &k, const torch::Tensor &qkv, const torch::Tensor &delta,
                  const torch::Tensor &ft, const torch::Tensor &fd, const torch::Tensor &axis,
                  const torch::Tensor &q_gamma, const torch::Tensor &k_gamma, double scale) {
  check(qkv, "qkv", torch::kHalf);
  TORCH_CHECK(qkv.dim() == 4 && qkv.size(1) == 3 && qkv.size(3) == 64, "qkv must be [R, 3, H, 64], got ",
              qkv.sizes());
  const int64_t R = qkv.size(0), H = qkv.size(2);
  TORCH_CHECK(R * H * 4 < INT32_MAX && R > 0 && H > 0, "bad qkv size");
  check(delta, "delta", torch::kFloat);
  TORCH_CHECK(delta.numel() == R * H * 3, "delta must be [R, H * 3]");
  check(ft, "ft", torch::kFloat);
  checkNumel(ft, "ft", 32);
  check(fd, "fd", torch::kFloat);
  checkNumel(fd, "fd", 32);
  check(axis, "axis", torch::kInt);
  checkNumel(axis, "axis", 32);
  check(q_gamma, "q_gamma", torch::kFloat);
  checkNumel(q_gamma, "q_gamma", H * 64);
  check(k_gamma, "k_gamma", torch::kFloat);
  checkNumel(k_gamma, "k_gamma", H * 64);
  check(q, "q", torch::kHalf);
  checkNumel(q, "q", R * H * 64);
  check(k, "k", torch::kHalf);
  checkNumel(k, "k", R * H * 64);
  const float sc = (float)scale;
  encodePass("fused_repo_rope_qk", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    bind(enc, qkv, 0);
    bind(enc, delta, 1);
    bind(enc, ft, 2);
    bind(enc, fd, 3);
    bind(enc, axis, 4);
    bind(enc, q_gamma, 5);
    bind(enc, k_gamma, 6);
    bind(enc, q, 7);
    bind(enc, k, 8);
    bindVal<uint32_t>(enc, (uint32_t)R, 9);
    bindVal<float>(enc, sc, 10);
    bindVal<uint32_t>(enc, (uint32_t)H, 11);
    const NSUInteger n = (NSUInteger)(R * H * 4);
    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  });
}

// out = silu(gate) * content, float.
void silu_mul(torch::Tensor &out, const torch::Tensor &gate, const torch::Tensor &content) {
  check(gate, "gate", torch::kFloat);
  check(content, "content", torch::kFloat);
  check(out, "out", torch::kFloat);
  TORCH_CHECK(gate.sizes() == content.sizes() && out.sizes() == gate.sizes(), "shapes must match");
  if (gate.numel() == 0) return;
  TORCH_CHECK(gate.numel() < INT32_MAX, "too large");
  encodePass("exact_silu_mul_f32", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    bind(enc, gate, 0);
    bind(enc, content, 1);
    bind(enc, out, 2);
    [enc dispatchThreads:MTLSizeMake(gate.numel(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  });
}

// x = gelu(x, approximate='tanh') in place, half.
void gelu_tanh_(torch::Tensor &x) {
  check(x, "x", torch::kHalf);
  if (x.numel() == 0) return;
  TORCH_CHECK(x.numel() < INT32_MAX, "too large");
  const uint32_t n = (uint32_t)x.numel();
  encodePass("fused_gelu_tanh_half_inplace", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    bind(enc, x, 0);
    bindVal<uint32_t>(enc, n, 1);
    [enc dispatchThreads:MTLSizeMake((n + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  });
}

// out = x + h * gate (half, gate broadcast over the last dim).
void gated_residual(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &h, const torch::Tensor &gate) {
  check(x, "x", torch::kHalf);
  check(h, "h", torch::kHalf);
  check(out, "out", torch::kHalf);
  check(gate, "gate", torch::kHalf);
  TORCH_CHECK(x.dim() >= 1 && h.sizes() == x.sizes() && out.sizes() == x.sizes(), "shapes must match");
  if (x.numel() == 0) return;
  const int64_t C = x.size(-1);
  checkNumel(gate, "gate", C);
  TORCH_CHECK(x.numel() < INT32_MAX, "too large");
  encodePass("exact_resid", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    bind(enc, x, 0);
    bind(enc, h, 1);
    bind(enc, gate, 2);
    bind(enc, out, 3);
    bindVal<uint32_t>(enc, (uint32_t)C, 4);
    [enc dispatchThreads:MTLSizeMake(x.numel(), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  });
}

// out = gelu_tanh(F.linear(x, weight, bias)) for half x [M, K], weight [N, K], bias [N] (MPP matmul2d, fp32
// accumulation, bias added in fp32, one rounding to half, then GELU).
void linear_gelu(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &bias) {
  check(x, "x", torch::kHalf);
  check(weight, "weight", torch::kHalf);
  check(bias, "bias", torch::kHalf);
  check(out, "out", torch::kHalf);
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x must be [M, K] and weight [N, K]");
  const int64_t M = x.size(0), K = x.size(1), N = weight.size(0);
  TORCH_CHECK(weight.size(1) == K, "weight must be [N, ", K, "], got ", weight.sizes());
  checkNumel(bias, "bias", N);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N, "out must be [", M, ", ", N, "]");
  TORCH_CHECK(M > 0 && N > 0 && K > 0 && M < INT32_MAX && N < INT32_MAX && K < INT32_MAX, "bad sizes");
  const uint32_t m = (uint32_t)M, n = (uint32_t)N, k = (uint32_t)K;
  encodePass("linear_bias_gelu_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    bind(enc, x, 0);
    bind(enc, weight, 1);
    bind(enc, out, 2);
    bind(enc, bias, 3);
    bindVal<uint32_t>(enc, m, 4);
    bindVal<uint32_t>(enc, n, 5);
    bindVal<uint32_t>(enc, k, 6);
    [enc dispatchThreadgroups:MTLSizeMake((N + 127) / 128, (M + 63) / 64, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  });
}

// out[r, h, :] = F.normalize(src[r, col_offset + h*64 : ... + 64].float()) * gamma[h] * scale -> half.
void rmsnorm_heads(torch::Tensor &out, const torch::Tensor &src, int64_t col_offset, const torch::Tensor &gamma,
                   double scale) {
  check(src, "src", torch::kHalf);
  check(out, "out", torch::kHalf);
  check(gamma, "gamma", torch::kFloat);
  TORCH_CHECK(src.dim() == 2, "src must be [R, W]");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == 64, "out must be [R, H, 64]");
  const int64_t R = src.size(0), W = src.size(1), H = out.size(1);
  TORCH_CHECK(out.size(0) == R, "out rows must match src");
  TORCH_CHECK(col_offset >= 0 && col_offset + H * 64 <= W, "col_offset + H * 64 must be within src's width");
  checkNumel(gamma, "gamma", H * 64);
  if (R == 0) return;
  TORCH_CHECK(R * H * 4 < INT32_MAX && W < INT32_MAX, "too large");
  const uint32_t r = (uint32_t)R, hh = (uint32_t)H, rs = (uint32_t)W;
  const float sc = (float)scale;
  const NSUInteger off = mtlOffset(src) + (NSUInteger)col_offset * 2;
  encodePass("rmsnorm_heads_f16", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    [enc setBuffer:mtlBuffer(src) offset:off atIndex:0];
    bind(enc, gamma, 1);
    bind(enc, out, 2);
    bindVal<uint32_t>(enc, r, 3);
    bindVal<uint32_t>(enc, hh, 4);
    bindVal<uint32_t>(enc, rs, 5);
    bindVal<float>(enc, sc, 6);
    const NSUInteger nth = (NSUInteger)(R * H * 4);
    [enc dispatchThreadgroups:MTLSizeMake((nth + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  });
}

// y = LayerNorm32(x') (half; affine with float weight/bias when given), x' = x + h when h is given (-> xout).
void resid_layernorm(torch::Tensor &y, const std::optional<torch::Tensor> &xout, const torch::Tensor &x,
                     const std::optional<torch::Tensor> &h, const std::optional<torch::Tensor> &weight,
                     const std::optional<torch::Tensor> &bias, double eps) {
  check(x, "x", torch::kHalf);
  TORCH_CHECK(x.dim() >= 1 && x.numel() > 0, "x must be a non-empty [..., C] tensor");
  const int64_t C = x.size(-1), R = x.numel() / C;
  TORCH_CHECK(C % 4 == 0 && C / 4 <= 1024, "C must be a multiple of 4 and at most 4096, got ", C);
  check(y, "y", torch::kHalf);
  TORCH_CHECK(y.sizes() == x.sizes(), "y must have the shape of x");
  TORCH_CHECK(h.has_value() == xout.has_value(), "h and xout go together");
  TORCH_CHECK(weight.has_value() == bias.has_value(), "weight and bias go together");
  uint32_t mode = 0;
  if (h) {
    check(*h, "h", torch::kHalf);
    TORCH_CHECK(h->sizes() == x.sizes(), "h must have the shape of x");
    check(*xout, "xout", torch::kHalf);
    TORCH_CHECK(xout->sizes() == x.sizes(), "xout must have the shape of x");
    mode |= 8u;
  }
  if (weight) {
    check(*weight, "weight", torch::kFloat);
    checkNumel(*weight, "weight", C);
    check(*bias, "bias", torch::kFloat);
    checkNumel(*bias, "bias", C);
    mode |= 4u;
  }
  // without affine, the modulated path with scale1p = 1, shift = 0 is used: hn * 1 + 0 == hn except that a
  // negative zero becomes +0 (numerically identical)
  auto one = torch::ones({C}, x.options());
  auto zero = torch::zeros({C}, x.options());
  const float eps_f = (float)eps, eps_r = 1e-5f;
  encodePass("fused_resid_ln_mod_repo", [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState>) {
    bind(enc, x, 0);
    bind(enc, h ? *h : x, 1);
    bind(enc, one, 2);
    bind(enc, xout ? *xout : y, 3);
    bind(enc, one, 4);
    bind(enc, zero, 5);
    bind(enc, y, 6);
    bind(enc, weight ? *weight : y, 7);
    bind(enc, bias ? *bias : y, 8);
    bind(enc, y, 9);
    bindVal<uint32_t>(enc, (uint32_t)C, 10);
    bindVal<float>(enc, eps_f, 11);
    bindVal<float>(enc, eps_r, 12);
    bindVal<uint32_t>(enc, mode, 13);
    [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(C / 4, 1, 1)];
  });
}
