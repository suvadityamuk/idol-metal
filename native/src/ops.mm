#include "ops.h"

#include <cmath>
#include <stdexcept>

namespace idol {

static MPSDataType mps_dtype(DType t) {
  switch (t) {
    case DType::F32: return MPSDataTypeFloat32;
    case DType::F16: return MPSDataTypeFloat16;
    case DType::BF16: return MPSDataTypeBFloat16;
    default: throw std::runtime_error("mps_dtype: unsupported");
  }
}

// getMPSNDArray(t, sizes, strides) for a contiguous tensor (OperationUtils.mm): packed-rows descriptor,
// arrayViewWithShape:strides: with the (already sorted) strides, identity permutation view.
static MPSNDArray* nd_contiguous(const Tensor& t, NSArray<NSNumber*>* shape, NSArray<NSNumber*>* strides) {
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:mps_dtype(t.dtype) shape:shape];
  d.preferPackedRows = YES;
  MPSNDArray* a = [[[MPSNDArray alloc] initWithBuffer:t.buf() offset:t.offset descriptor:d] autorelease];
  a = [a arrayViewWithShape:shape strides:strides];
  return [a arrayViewWithCommandBuffer:nil descriptor:[a descriptor] aliasing:MPSAliasingStrategyShallAlias];
}

Tensor linear_nd(Context& c, const Tensor& x, const Tensor& w, const Tensor* bias, Tensor out) {
  @autoreleasepool {
    const int64_t K = x.shape.back(), M = x.numel() / K, N = w.shape[0];
    if (M < 1) throw std::runtime_error("linear_nd: empty input");
    if (w.shape[1] != K) throw std::runtime_error("linear_nd: shape mismatch");
    if (!out.defined()) {
      std::vector<int64_t> s = x.shape;
      s.back() = N;
      out = c.empty(x.dtype, s);
    }
    MPSNDArray* ax = nd_contiguous(x, @[ @(M), @(K) ], @[ @(K), @1 ]);
    MPSNDArray* ay = nd_contiguous(out, @[ @(M), @(N) ], @[ @(N), @1 ]);
    MPSNDArrayDescriptor* wd = [MPSNDArrayDescriptor descriptorWithDataType:mps_dtype(w.dtype) shape:@[ @(N), @(K) ]];
    wd.preferPackedRows = YES;
    [wd transposeDimension:0 withDimension:1];
    MPSNDArray* aw = [[[MPSNDArray alloc] initWithBuffer:w.buf() offset:w.offset descriptor:wd] autorelease];
    static thread_local MPSNDArrayMatrixMultiplication* k2 = nil;
    static thread_local MPSNDArrayMatrixMultiplication* k3 = nil;
    MPSNDArrayMatrixMultiplication* k;
    NSArray* srcs;
    if (bias) {
      if (!k3) k3 = [[MPSNDArrayMatrixMultiplication alloc] initWithDevice:c.device() sourceCount:3];
      k = k3;
      srcs = @[ ax, aw, nd_contiguous(*bias, @[ @(N) ], @[ @1 ]) ];
    } else {
      if (!k2) k2 = [[MPSNDArrayMatrixMultiplication alloc] initWithDevice:c.device() sourceCount:2];
      k = k2;
      srcs = @[ ax, aw ];
    }
    [k encodeToCommandEncoder:c.encoder() commandBuffer:c.cmdbuf() sourceArrays:srcs destinationArray:ay];
    c.dispatches++;
    return out;
  }
}

struct PrefillAttnParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale, softcapping;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

void attention_prefill(Context& c, const AttnView& q, const AttnView& k, const AttnView& v, const AttnView& o,
                       int B, int H, int D, int Lq, int Lk, const Tensor* mask) {
  if (q.t.dtype != DType::F16) throw std::runtime_error("attention_prefill: half only");
  // sdpa_prefill_mps with use_mpp (qL > 8, half, D in {64,96,128,256}): BQ=64, BK=32, WM=4, WN=1
  std::string name = "prefill_attention_mpp_a10_float16_bq64_bk32_bd" + std::to_string(D) + "_wm4_wn1_hm" +
                     (mask ? "1" : "0") + "_dc0_maskfloat16";
  auto p = c.pso(LIB_TORCH, name);
  PrefillAttnParams prm{};
  prm.B = B; prm.H = H; prm.D = D; prm.qL = Lq; prm.kL = Lk; prm.gqa_factor = 1;
  prm.scale = (float)(1.0 / std::sqrt((double)D));  // sdp::calculate_scale
  prm.softcapping = 1.0f;
  // kernel strides are (batch, head, seq)
  prm.Q_strides[0] = q.sb; prm.Q_strides[1] = q.sh; prm.Q_strides[2] = q.sl;
  prm.K_strides[0] = k.sb; prm.K_strides[1] = k.sh; prm.K_strides[2] = k.sl;
  prm.V_strides[0] = v.sb; prm.V_strides[1] = v.sh; prm.V_strides[2] = v.sl;
  prm.O_strides[0] = o.sb; prm.O_strides[1] = o.sh; prm.O_strides[2] = o.sl;
  auto e = c.encoder();
  [e setComputePipelineState:p];
  set_buf(e, q.t, 0);
  set_buf(e, k.t, 1);
  set_buf(e, v.t, 2);
  set_buf(e, o.t, 3);
  set_val(e, prm, 4);
  if (mask) {
    int64_t mstr[4] = {0, 0, 0, 1};  // (B, H, qL, kL): one row of Lk biases broadcast
    [e setBytes:mstr length:sizeof(mstr) atIndex:5];
    set_buf(e, *mask, 6);
  } else {
    uint32_t zero = 0;
    set_val(e, zero, 5);
    set_val(e, zero, 6);
  }
  c.dispatch_groups(p, MTLSizeMake((Lq + 63) / 64, H, B), MTLSizeMake(32, 4, 1));
}

}  // namespace idol
