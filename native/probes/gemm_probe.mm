// Probe: can MPSNDArrayMatrixMultiplication, called outside PyTorch exactly the way torch 2.15's
// _mps_linear_nograph calls it (Linear.mm), reproduce the reference's linear outputs byte for byte?
// usage: gemm_probe <dump.safetensors> <weights.safetensors> <dump-prefix> <weight-prefix> <f16|f32> [bias]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdio>
#include <string>
#include <vector>

#include "../src/safetensors.h"

static MPSNDArray* strided2d(id<MTLBuffer> buf, MPSDataType dt, NSUInteger rows, NSUInteger cols) {
  // getMPSNDArray(t, sizes, strides) for a contiguous 2D tensor: packed-rows descriptor, arrayView with the
  // (sorted) strides, identity permutation (a no-op view).
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:dt shape:@[ @(rows), @(cols) ]];
  d.preferPackedRows = YES;
  MPSNDArray* a = [[[MPSNDArray alloc] initWithBuffer:buf offset:0 descriptor:d] autorelease];
  a = [a arrayViewWithShape:@[ @(rows), @(cols) ] strides:@[ @(cols), @1 ]];
  MPSNDArrayDescriptor* d2 = [a descriptor];
  return [a arrayViewWithCommandBuffer:nil descriptor:d2 aliasing:MPSAliasingStrategyShallAlias];
}

static MPSNDArray* strided1d(id<MTLBuffer> buf, MPSDataType dt, NSUInteger n) {
  MPSNDArrayDescriptor* d = [MPSNDArrayDescriptor descriptorWithDataType:dt shape:@[ @(n) ]];
  d.preferPackedRows = YES;
  MPSNDArray* a = [[[MPSNDArray alloc] initWithBuffer:buf offset:0 descriptor:d] autorelease];
  a = [a arrayViewWithShape:@[ @(n) ] strides:@[ @1 ]];
  MPSNDArrayDescriptor* d2 = [a descriptor];
  return [a arrayViewWithCommandBuffer:nil descriptor:d2 aliasing:MPSAliasingStrategyShallAlias];
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 6) {
      fprintf(stderr, "usage: %s dump weights dump_prefix weight_prefix f16|f32 [bias]\n", argv[0]);
      return 2;
    }
    st::File dump(argv[1]), weights(argv[2]);
    std::string dp = argv[3], wp = argv[4], dtype = argv[5];
    bool has_bias = argc > 6 && std::string(argv[6]) == "bias";
    bool half = dtype == "f16";
    MPSDataType dt = half ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
    size_t es = half ? 2 : 4;

    std::string in_key = getenv("IN_KEY") ? getenv("IN_KEY") : dp + "/in0";
    const auto& x = dump.get(in_key);
    const auto& y = dump.get(dp + "/out");
    const auto& w = weights.get(wp + ".weight");
    NSUInteger K = x.shape.back(), M = x.numel() / K, N = w.shape[0];
    printf("M=%lu K=%lu N=%lu dtype=%s bias=%d\n", (unsigned long)M, (unsigned long)K, (unsigned long)N,
           dtype.c_str(), has_bias);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> bx = [dev newBufferWithBytes:x.data length:x.nbytes() options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [dev newBufferWithLength:M * N * es options:MTLResourceStorageModeShared];
    // weights: fp32 checkpoint, converted like load_state_dict's copy_ into fp16 parameters (RNE)
    std::vector<uint8_t> wbytes(N * K * es), bbytes(N * es);
    const float* wf = (const float*)w.data;
    for (size_t i = 0; i < N * K; i++) {
      if (half) ((__fp16*)wbytes.data())[i] = (__fp16)wf[i];
      else ((float*)wbytes.data())[i] = wf[i];
    }
    id<MTLBuffer> bw = [dev newBufferWithBytes:wbytes.data() length:wbytes.size() options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = nil;
    if (has_bias) {
      const float* bf = (const float*)weights.get(wp + ".bias").data;
      for (size_t i = 0; i < N; i++) {
        if (half) ((__fp16*)bbytes.data())[i] = (__fp16)bf[i];
        else ((float*)bbytes.data())[i] = bf[i];
      }
      bb = [dev newBufferWithBytes:bbytes.data() length:bbytes.size() options:MTLResourceStorageModeShared];
    }

    MPSNDArray* ax = strided2d(bx, dt, M, K);
    MPSNDArray* ay = strided2d(by, dt, M, N);
    MPSNDArrayDescriptor* wd = [MPSNDArrayDescriptor descriptorWithDataType:dt shape:@[ @(N), @(K) ]];
    wd.preferPackedRows = YES;
    [wd transposeDimension:0 withDimension:1];
    MPSNDArray* aw = [[[MPSNDArray alloc] initWithBuffer:bw offset:0 descriptor:wd] autorelease];

    MPSNDArrayMatrixMultiplication* k =
        [[[MPSNDArrayMatrixMultiplication alloc] initWithDevice:dev sourceCount:(has_bias ? 3 : 2)] autorelease];
    NSArray* srcs = has_bias ? @[ ax, aw, strided1d(bb, dt, N) ] : @[ ax, aw ];
    for (int rep = 0; rep < 3; rep++) {
      MPSCommandBuffer* cb = [MPSCommandBuffer commandBufferFromCommandQueue:q];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [k encodeToCommandEncoder:enc commandBuffer:cb sourceArrays:srcs destinationArray:ay];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      const uint8_t* got = (const uint8_t*)by.contents;
      const uint8_t* ref = y.data;
      size_t ndiff = 0, first = (size_t)-1;
      for (size_t i = 0; i < M * N; i++) {
        if (memcmp(got + i * es, ref + i * es, es) != 0) {
          if (first == (size_t)-1) first = i;
          ndiff++;
        }
      }
      printf("rep %d: %zu / %lu elements differ", rep, ndiff, (unsigned long)(M * N));
      if (ndiff) {
        double g = half ? (double)((const __fp16*)got)[first] : ((const float*)got)[first];
        double r = half ? (double)((const __fp16*)ref)[first] : ((const float*)ref)[first];
        printf(" (first at %zu: got %.9g ref %.9g)", first, g, r);
      }
      printf("\n");
    }
  }
  return 0;
}
