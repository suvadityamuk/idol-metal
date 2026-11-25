// Native Metal runtime for the Idol port: device/queue, a single serial compute encoder per command buffer (as
// PyTorch's MPS stream coalesces kernels), a caching buffer pool, contiguous tensors, kernel libraries, and the
// MPS / PyTorch-kernel primitives the parity build calls with exactly PyTorch 2.15's arguments.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace idol {

enum class DType : uint8_t { F32, F16, BF16, I64, I32, U8 };

inline size_t dsize(DType t) {
  switch (t) {
    case DType::F32: case DType::I32: return 4;
    case DType::F16: case DType::BF16: return 2;
    case DType::I64: return 8;
    case DType::U8: return 1;
  }
  return 0;
}
const char* dname(DType t);

class Context;

// A buffer from the pool; returned to the pool when the last Tensor referencing it is destroyed.
struct Storage {
  id<MTLBuffer> buf = nil;
  size_t bytes = 0;
  Context* ctx = nullptr;
  ~Storage();
};

// Contiguous (row-major) tensor view into a Storage.
struct Tensor {
  std::shared_ptr<Storage> st;
  size_t offset = 0;  // bytes
  DType dtype = DType::F32;
  std::vector<int64_t> shape;

  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  size_t nbytes() const { return (size_t)numel() * dsize(dtype); }
  int64_t size(int d) const { return shape[d < 0 ? shape.size() + d : d]; }
  int dim() const { return (int)shape.size(); }
  id<MTLBuffer> buf() const { return st->buf; }
  bool defined() const { return (bool)st; }
  // host pointer (shared storage); only valid after the GPU work producing it completed
  void* host() const { return (uint8_t*)st->buf.contents + offset; }
  template <typename T> T* ptr() const { return (T*)host(); }
  // views (no copy)
  Tensor view(std::vector<int64_t> s) const;
  Tensor slice_rows(int64_t row0, int64_t rows) const;  // along dim 0 of a 2D view [rows, cols]
};

std::string shape_str(const std::vector<int64_t>& s);

class Context {
 public:
  explicit Context(const std::string& torch_metallib_dir = "");
  ~Context();

  id<MTLDevice> device() const { return dev_; }

  // encoding ------------------------------------------------------------------------------------------------
  MPSCommandBuffer* cmdbuf();
  id<MTLComputeCommandEncoder> encoder();
  void commit();     // end encoding, commit (no wait)
  void synchronize();  // commit and wait for everything queued
  uint64_t dispatches = 0;

  // memory --------------------------------------------------------------------------------------------------
  Tensor empty(DType t, std::vector<int64_t> shape);
  Tensor zeros(DType t, std::vector<int64_t> shape);  // host memset after sync (shared storage)
  Tensor from_host(DType t, std::vector<int64_t> shape, const void* data);
  void release(Storage* s);
  size_t pool_bytes() const { return pool_bytes_; }
  size_t peak_bytes() const { return peak_bytes_; }
  void reset_peak() { peak_bytes_ = live_bytes_; }
  void empty_pool();

  // kernels -------------------------------------------------------------------------------------------------
  // my kernels: runtime-compiled with MTLMathModeSafe + precise math functions (parity) or fast math (fast)
  void add_source_library(const std::string& name, const std::string& source, bool fast_math);
  id<MTLComputePipelineState> pso(const std::string& lib, const std::string& fn);
  bool has_torch_libs() const { return torch40_ != nil; }

  // dispatch helpers
  void dispatch_threads(id<MTLComputePipelineState> p, size_t n, size_t tg = 256);
  void dispatch_groups(id<MTLComputePipelineState> p, MTLSize groups, MTLSize tg);

 private:
  id<MTLDevice> dev_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  MPSCommandBuffer* cb_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
  std::vector<MPSCommandBuffer*> inflight_;
  std::map<std::string, id<MTLLibrary>> libs_;
  id<MTLLibrary> torch40_ = nil, torchbasic_ = nil;
  std::map<std::string, id<MTLComputePipelineState>> psos_;
  std::multimap<size_t, id<MTLBuffer>> free_;
  size_t pool_bytes_ = 0, live_bytes_ = 0, peak_bytes_ = 0;
};

// helpers to bind args on the current encoder
inline void set_buf(id<MTLComputeCommandEncoder> e, const Tensor& t, int idx) {
  [e setBuffer:t.buf() offset:t.offset atIndex:idx];
}
template <typename T>
inline void set_val(id<MTLComputeCommandEncoder> e, const T& v, int idx) {
  [e setBytes:&v length:sizeof(T) atIndex:idx];
}

// fp32 <-> fp16 on the host (round to nearest even, as PyTorch's CPU copy_)
void f32_to_f16(const float* src, uint16_t* dst, size_t n);
float bf16_to_f32(uint16_t v);

}  // namespace idol
