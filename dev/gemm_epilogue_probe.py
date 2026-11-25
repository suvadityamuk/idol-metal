"""Can an MPP GEMM + bias + GELU epilogue reproduce F.linear(x, W, b) (MPS) followed by GELU bit for bit, and is it
faster than the two separate ops?"""
import time, torch, torch.nn.functional as F, sys, os
SRC = r'''
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
inline float gelu_h(float xf) {
  constexpr float kBeta = M_SQRT2_F * M_2_SQRTPI_F * 0.5f;
  constexpr float kKappa = 0.044715f;
  const float inner = kBeta * (xf + kKappa * xf * xf * xf);
  return 0.5f * xf * (1.0f + fast::tanh(clamp(inner, -10.0f, 10.0f)));
}
template <int TM, int TN, int MODE>
void mm(device half* X, device half* W, device half* Y, device const half* bias, uint M, uint N, uint K, uint2 tg) {
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto A = tensor<device half, dextents<int32_t,2>, tensor_inline>(X, dextents<int32_t,2>(K, M));
  auto B = tensor<device half, dextents<int32_t,2>, tensor_inline>(W, dextents<int32_t,2>(K, N));
  auto C = tensor<device half, dextents<int32_t,2>, tensor_inline>(Y, dextents<int32_t,2>(N, M));
  const int r0 = tg.y * TM, c0 = tg.x * TN;
  auto mA = A.slice(0, r0); auto mB = B.slice(0, c0); auto mC = C.slice(c0, r0);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  #pragma unroll
  for (ushort i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, mB, cT);
  auto cH = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
  #pragma unroll
  for (ushort i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto idx = cT.get_multidimensional_index(i);
      const int c = min(c0 + (int)idx[0], int(N) - 1);
      half h;
      if (MODE == 0) h = half(cT[i] + float(bias[c]));          // fp32 add, one rounding
      else if (MODE == 1) { half a = half(cT[i]); h = a + bias[c]; }   // round, then half add
      else { h = half(cT[i] + float(bias[c])); h = half(gelu_h(float(h))); }
      cH[i] = h;
    }
  }
  cH.store(mC);
}
#define K_(NAME, TM, TN, MODE) kernel void NAME(device half* X [[buffer(0)]], device half* W [[buffer(1)]], device half* Y [[buffer(2)]], device const half* b [[buffer(3)]], constant uint& M [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& K [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]]) { mm<TM, TN, MODE>(X, W, Y, b, M, N, K, tg); }
K_(mm_bias_f32add, 64, 128, 0)
K_(mm_bias_halfadd, 64, 128, 1)
K_(mm_bias_gelu, 64, 128, 2)
'''
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fast import ops
lib = torch.mps.compile_shader(SRC)
M, K, N = 12294, 1024, 4096
x = torch.randn(M, K, device='mps').half()
W = (torch.randn(N, K, device='mps') * 0.03).half(); b = (torch.randn(N, device='mps') * 0.5).half()
ref = F.linear(x, W, b)
def call(name, Y): getattr(lib, name)(x, W, Y, b, M, N, K, threads=[(N // 128) * 128, (M + 63) // 64], group_size=[128, 1])
for name in ('mm_bias_f32add', 'mm_bias_halfadd'):
    Y = torch.empty(M, N, device='mps', dtype=torch.float16); call(name, Y); torch.mps.synchronize()
    print(name, 'bit-equal to F.linear:', torch.equal(Y, ref), 'n diff', int((Y != ref).sum()))
ref_g = ops.gelu_(ref.clone())
Y = torch.empty(M, N, device='mps', dtype=torch.float16); call('mm_bias_gelu', Y); torch.mps.synchronize()
print('mm_bias_gelu bit-equal to F.linear + gelu kernel:', torch.equal(Y, ref_g), 'n diff', int((Y != ref_g).sum()))
def bench(fn, n=20):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / n * 1e3
print(f'F.linear + gelu: {bench(lambda: ops.gelu_(F.linear(x, W, b))):.2f} ms; fused MPP: {bench(lambda: call("mm_bias_gelu", Y)):.2f} ms')
