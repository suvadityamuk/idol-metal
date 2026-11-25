#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "resid_ln_mod(Tensor(a!) y, Tensor(b!)? xout, Tensor(c!)? ry, Tensor x, Tensor? h, Tensor? gate, "
      "Tensor scale1p, Tensor shift, Tensor? repo_weight, Tensor? repo_bias, float eps, float eps_repo) -> ()");
  ops.def(
      "repo_rope_qk(Tensor(a!) q, Tensor(b!) k, Tensor qkv, Tensor delta, Tensor ft, Tensor fd, Tensor axis, "
      "Tensor q_gamma, Tensor k_gamma, float scale) -> ()");
  ops.def("silu_mul(Tensor(a!) out, Tensor gate, Tensor content) -> ()");
  ops.def("gelu_tanh_(Tensor(a!) x) -> ()");
  ops.def("gated_residual(Tensor(a!) out, Tensor x, Tensor h, Tensor gate) -> ()");
  ops.def(
      "resid_layernorm(Tensor(a!) y, Tensor(b!)? xout, Tensor x, Tensor? h, Tensor? weight, Tensor? bias, "
      "float eps) -> ()");
  ops.def("linear_gelu(Tensor(a!) out, Tensor x, Tensor weight, Tensor bias) -> ()");
  ops.def("rmsnorm_heads(Tensor(a!) out, Tensor src, int col_offset, Tensor gamma, float scale) -> ()");
#if defined(METAL_KERNEL)
  ops.impl("resid_ln_mod", torch::kMPS, resid_ln_mod);
  ops.impl("repo_rope_qk", torch::kMPS, repo_rope_qk);
  ops.impl("silu_mul", torch::kMPS, silu_mul);
  ops.impl("gelu_tanh_", torch::kMPS, gelu_tanh_);
  ops.impl("gated_residual", torch::kMPS, gated_residual);
  ops.impl("resid_layernorm", torch::kMPS, resid_layernorm);
  ops.impl("linear_gelu", torch::kMPS, linear_gelu);
  ops.impl("rmsnorm_heads", torch::kMPS, rmsnorm_heads);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
