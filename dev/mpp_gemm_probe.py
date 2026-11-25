import time, torch, inspect
print(inspect.signature(torch.mps.compile_shader))
SRC = r'''
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
// C[M,N] = A[M,K] @ W[N,K]^T   (row-major, like F.linear)
template<int TM, int TN, int NSG>
void gemm_impl(device const half* A, device const half* W, device half* C, uint M, uint N, uint K, uint2 tg) {
  auto tA = tensor<device half, dextents<int32_t,2>, tensor_inline>((device half*)A, dextents<int32_t,2>(K, M));
  auto tW = tensor<device half, dextents<int32_t,2>, tensor_inline>((device half*)W, dextents<int32_t,2>(K, N));
  auto tC = tensor<device half, dextents<int32_t,2>, tensor_inline>(C, dextents<int32_t,2>(N, M));
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto mA = tA.slice(0, tg.y*TM);
  auto mW = tW.slice(0, tg.x*TN);
  auto mC = tC.slice(tg.x*TN, tg.y*TM);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mW), float>();
  #pragma unroll
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
  op.run(mA, mW, cT);
  auto cH = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mW), half>();
  #pragma unroll
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cH[i] = half(cT[i]);
  cH.store(mC);
}
#define INST(TM,TN,NSG) kernel void gemm_##TM##_##TN##_##NSG(device const half* A [[buffer(0)]], device const half* W [[buffer(1)]], device half* C [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& K [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]]) { gemm_impl<TM,TN,NSG>(A,W,C,M,N,K,tg); }
INST(64,32,4)
INST(64,64,4)
INST(128,64,4)
INST(128,128,4)
INST(64,128,4)
'''
lib = torch.mps.compile_shader(SRC)
M, K, N = 12294, 1024, 4096
A = torch.randn(M, K, device='mps', dtype=torch.float16)
W = torch.randn(N, K, device='mps', dtype=torch.float16) * 0.03
ref = A @ W.t()
for name in ['gemm_64_32_4', 'gemm_64_64_4', 'gemm_128_64_4', 'gemm_64_128_4', 'gemm_128_128_4']:
    TM, TN, NSG = map(int, name.split('_')[1:])
    C = torch.empty(M, N, device='mps', dtype=torch.float16)
    fn = getattr(lib, name)
    run = lambda: fn(A, W, C, M, N, K, threads=[((N+TN-1)//TN)*32*NSG, (M+TM-1)//TM], group_size=[32*NSG, 1])
    run(); torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(20): run()
    torch.mps.synchronize(); dt = (time.perf_counter()-t)/20
    err = (C.float()-ref.float()).abs().max().item()
    print(f'{name}: {dt*1e3:.2f} ms {2*M*N*K/dt/1e12:.1f} TFLOP/s maxerr {err:.3e}')
t = time.perf_counter()
for _ in range(20): ref = A @ W.t()
torch.mps.synchronize(); dt = (time.perf_counter()-t)/20
print(f'torch mm: {dt*1e3:.2f} ms {2*M*N*K/dt/1e12:.1f} TFLOP/s')
