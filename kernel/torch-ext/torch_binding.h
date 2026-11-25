#pragma once

#include <torch/torch.h>

#include <optional>

void resid_ln_mod(torch::Tensor &y, const std::optional<torch::Tensor> &xout, const std::optional<torch::Tensor> &ry,
                  const torch::Tensor &x, const std::optional<torch::Tensor> &h,
                  const std::optional<torch::Tensor> &gate, const torch::Tensor &scale1p, const torch::Tensor &shift,
                  const std::optional<torch::Tensor> &repo_weight, const std::optional<torch::Tensor> &repo_bias,
                  double eps, double eps_repo);
void repo_rope_qk(torch::Tensor &q, torch::Tensor &k, const torch::Tensor &qkv, const torch::Tensor &delta,
                  const torch::Tensor &ft, const torch::Tensor &fd, const torch::Tensor &axis,
                  const torch::Tensor &q_gamma, const torch::Tensor &k_gamma, double scale);
void silu_mul(torch::Tensor &out, const torch::Tensor &gate, const torch::Tensor &content);
void gelu_tanh_(torch::Tensor &x);
void gated_residual(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &h, const torch::Tensor &gate);
void linear_gelu(torch::Tensor &out, const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &bias);
void rmsnorm_heads(torch::Tensor &out, const torch::Tensor &src, int64_t col_offset, const torch::Tensor &gamma,
                   double scale);
void resid_layernorm(torch::Tensor &y, const std::optional<torch::Tensor> &xout, const torch::Tensor &x,
                     const std::optional<torch::Tensor> &h, const std::optional<torch::Tensor> &weight,
                     const std::optional<torch::Tensor> &bias, double eps);
