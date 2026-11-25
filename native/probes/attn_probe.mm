// Probe: torch 2.15's own MPP prefill attention kernel, loaded from the metallib extracted from libtorch_cpu,
// dispatched outside PyTorch exactly like sdpa_prefill_mps (Attention.mm). Byte-compares with the dump.
// usage: attn_probe <metal_40.metallib> <dump.safetensors> <dump-prefix e.g. dit/call000/fn/sdpa04> [blhd_out]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <string>

#include "../src/safetensors.h"

struct PrefillAttnParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale, softcapping;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string lib_path = argv[1];
    st::File dump(argv[2]);
    std::string pre = argv[3];
    bool blhd_out = argc > 4;
    const auto& q = dump.get(pre + "/in0");
    const auto& k = dump.get(pre + "/in1");
    const auto& v = dump.get(pre + "/in2");
    const auto& o = dump.get(pre + "/out");
    // dumped as [B, L, H, D] (the reference permutes views to [B, H, L, D] before SDPA)
    int B = q.shape[0], qL = q.shape[1], H = q.shape[2], D = q.shape[3], kL = k.shape[1];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(lib_path.c_str())] error:&err];
    if (!lib) { fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String); return 1; }
    std::string name = "prefill_attention_mpp_a10_float16_bq64_bk32_bd" + std::to_string(D) +
                       "_wm4_wn1_hm0_dc0_maskfloat16";
    id<MTLFunction> fn = [lib newFunctionWithName:@(name.c_str())];
    if (!fn) { fprintf(stderr, "no function %s\n", name.c_str()); return 1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> cq = [dev newCommandQueue];
    auto buf = [&](const st::TensorInfo& t) {
      return [dev newBufferWithBytes:t.data length:t.nbytes() options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> bq = buf(q), bk = buf(k), bv = buf(v);
    id<MTLBuffer> bo = [dev newBufferWithLength:o.nbytes() options:MTLResourceStorageModeShared];
    PrefillAttnParams p{};
    p.B = B; p.H = H; p.D = D; p.qL = qL; p.kL = kL; p.gqa_factor = 1;
    p.scale = (float)(1.0 / std::sqrt((double)D));
    p.softcapping = 1.0f;
    int64_t sq[3] = {(int64_t)qL * H * D, D, (int64_t)H * D}, sk[3] = {(int64_t)kL * H * D, D, (int64_t)H * D};
    for (int i = 0; i < 3; i++) { p.Q_strides[i] = sq[i]; p.K_strides[i] = sk[i]; p.V_strides[i] = sk[i]; }
    if (blhd_out) { for (int i = 0; i < 3; i++) p.O_strides[i] = sq[i]; }
    else { p.O_strides[0] = (int64_t)H * qL * D; p.O_strides[1] = (int64_t)qL * D; p.O_strides[2] = D; }
    id<MTLCommandBuffer> cb = [cq commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bq offset:0 atIndex:0];
    [enc setBuffer:bk offset:0 atIndex:1];
    [enc setBuffer:bv offset:0 atIndex:2];
    [enc setBuffer:bo offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    uint32_t zero = 0;
    [enc setBytes:&zero length:4 atIndex:5];
    [enc setBytes:&zero length:4 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((qL + 63) / 64, H, B) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const uint16_t* got = (const uint16_t*)bo.contents;
    const uint16_t* ref = (const uint16_t*)o.data;
    size_t ndiff = 0;
    for (int b = 0; b < B; b++)
      for (int l = 0; l < qL; l++)
        for (int h = 0; h < H; h++)
          for (int d = 0; d < D; d++) {
            size_t ri = (((size_t)b * qL + l) * H + h) * D + d;
            size_t gi = blhd_out ? ri : (((size_t)b * H + h) * qL + l) * D + d;
            if (got[gi] != ref[ri]) ndiff++;
          }
    printf("%s L=%d kL=%d H=%d D=%d out=%s: %zu / %zu differ\n", pre.c_str(), qL, kL, H, D,
           blhd_out ? "BLHD" : "BHLD", ndiff, (size_t)B * qL * H * D);
  }
  return 0;
}
