// Primitive ops of the native Idol port. Each says which PyTorch 2.15 MPS path it reproduces.
#pragma once
#include "runtime.h"

namespace idol {

// Library names
constexpr const char* LIB_EXACT = "exact";  // my kernels, Safe math (parity)
constexpr const char* LIB_FAST = "fast";    // my kernels, fast math (fast build)
constexpr const char* LIB_TORCH = "torch40";

// F.linear for an input with rows > 1 (contiguous): torch's _mps_linear_nograph, flattened to 2D,
// MPSNDArrayMatrixMultiplication with (input, weight^T view, bias) sources. weight [N, K], bias [N] or undefined.
// out: [rows, N] of x's dtype (allocated when undefined).
Tensor linear_nd(Context& c, const Tensor& x, const Tensor& w, const Tensor* bias, Tensor out = {});

// PyTorch's prefill (MPP, Apple10) attention kernel from its own metallib.
// q [B, Lq, H, D], k/v [B, Lk, H, D] given by base tensors + element strides (B, L, H); output written with
// strides o_str (B, L, H) into o (any layout). half only, D in {64, 96, 128, 256}.
struct AttnView {
  Tensor t;            // storage view (offset honoured)
  int64_t sb, sl, sh;  // element strides for batch, seq, head
};
// mask: optional additive half key bias [Lk], broadcast over batch, heads and queries (the hm1 kernel variant).
void attention_prefill(Context& c, const AttnView& q, const AttnView& k, const AttnView& v, const AttnView& o,
                       int B, int H, int D, int Lq, int Lk, const Tensor* mask = nullptr);

}  // namespace idol
