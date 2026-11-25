#include "runtime.h"

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace idol {

const char* dname(DType t) {
  switch (t) {
    case DType::F32: return "float";
    case DType::F16: return "half";
    case DType::BF16: return "bfloat";
    case DType::I64: return "long";
    case DType::I32: return "int";
    case DType::U8: return "uchar";
  }
  return "?";
}

std::string shape_str(const std::vector<int64_t>& s) {
  std::ostringstream o;
  o << "[";
  for (size_t i = 0; i < s.size(); i++) o << (i ? ", " : "") << s[i];
  o << "]";
  return o.str();
}

Storage::~Storage() {
  if (ctx && buf) ctx->release(this);
}

Tensor Tensor::view(std::vector<int64_t> s) const {
  Tensor t = *this;
  int64_t n = 1, neg = -1;
  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] == -1) neg = i;
    else n *= s[i];
  }
  if (neg >= 0) s[neg] = numel() / n;
  t.shape = s;
  if (t.numel() != numel()) throw std::runtime_error("view: numel mismatch " + shape_str(shape) + " -> " + shape_str(s));
  return t;
}

Tensor Tensor::slice_rows(int64_t row0, int64_t rows) const {
  Tensor t = *this;
  int64_t cols = numel() / shape[0];
  t.offset = offset + (size_t)(row0 * cols) * dsize(dtype);
  t.shape[0] = rows;
  return t;
}

Context::Context(const std::string& torch_metallib_dir) {
  dev_ = MTLCreateSystemDefaultDevice();
  if (!dev_) throw std::runtime_error("no Metal device");
  queue_ = [dev_ newCommandQueue];
  if (!torch_metallib_dir.empty()) {
    NSError* err = nil;
    std::string p40 = torch_metallib_dir + "/metal_40.metallib", pb = torch_metallib_dir + "/metal_basic.metallib";
    torch40_ = [dev_ newLibraryWithURL:[NSURL fileURLWithPath:@(p40.c_str())] error:&err];
    if (!torch40_) throw std::runtime_error("cannot load " + p40);
    torchbasic_ = [dev_ newLibraryWithURL:[NSURL fileURLWithPath:@(pb.c_str())] error:&err];
    if (!torchbasic_) throw std::runtime_error("cannot load " + pb);
    libs_["torch40"] = torch40_;
    libs_["torchbasic"] = torchbasic_;
  }
}

Context::~Context() {
  synchronize();
  free_.clear();
}

MPSCommandBuffer* Context::cmdbuf() {
  if (!cb_) {
    cb_ = [[MPSCommandBuffer commandBufferFromCommandQueue:queue_] retain];
  }
  return cb_;
}

id<MTLComputeCommandEncoder> Context::encoder() {
  if (!enc_) {
    enc_ = [[cmdbuf() computeCommandEncoder] retain];  // serial dispatch, like PyTorch's stream encoder
  }
  return enc_;
}

void Context::commit() {
  if (enc_) {
    [enc_ endEncoding];
    [enc_ release];
    enc_ = nil;
  }
  if (cb_) {
    [cb_ commit];
    inflight_.push_back(cb_);
    cb_ = nil;
  }
}

void Context::synchronize() {
  commit();
  for (auto* c : inflight_) {
    [c waitUntilCompleted];
    if (c.status == MTLCommandBufferStatusError) {
      std::string msg = c.error ? c.error.localizedDescription.UTF8String : "unknown";
      [c release];
      inflight_.clear();
      throw std::runtime_error("command buffer failed: " + msg);
    }
    [c release];
  }
  inflight_.clear();
}

Tensor Context::empty(DType t, std::vector<int64_t> shape) {
  Tensor x;
  x.dtype = t;
  x.shape = shape;
  size_t bytes = std::max<size_t>(x.nbytes(), 16);
  bytes = (bytes + 255) & ~size_t(255);
  id<MTLBuffer> b = nil;
  auto it = free_.lower_bound(bytes);
  // reuse a pooled buffer of at most 2x the request
  if (it != free_.end() && it->first <= bytes * 2) {
    b = it->second;
    bytes = it->first;
    free_.erase(it);
  } else {
    b = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("out of memory allocating " + std::to_string(bytes) + " bytes");
    pool_bytes_ += bytes;
  }
  live_bytes_ += bytes;
  peak_bytes_ = std::max(peak_bytes_, live_bytes_);
  auto st = std::make_shared<Storage>();
  st->buf = b;
  st->bytes = bytes;
  st->ctx = this;
  x.st = st;
  return x;
}

void Context::release(Storage* s) {
  live_bytes_ -= s->bytes;
  free_.emplace(s->bytes, s->buf);  // the pool keeps the reference
  s->buf = nil;
}

void Context::empty_pool() {
  for (auto& kv : free_) pool_bytes_ -= kv.first;
  for (auto& kv : free_) [kv.second release];
  free_.clear();
}

Tensor Context::from_host(DType t, std::vector<int64_t> shape, const void* data) {
  // a fresh buffer (never one the GPU may still be using), filled on the host
  Tensor x;
  x.dtype = t;
  x.shape = shape;
  size_t bytes = std::max<size_t>(x.nbytes(), 16);
  bytes = (bytes + 255) & ~size_t(255);
  id<MTLBuffer> b = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!b) throw std::runtime_error("out of memory");
  if (x.nbytes()) std::memcpy(b.contents, data, x.nbytes());
  pool_bytes_ += bytes;
  live_bytes_ += bytes;
  peak_bytes_ = std::max(peak_bytes_, live_bytes_);
  auto st = std::make_shared<Storage>();
  st->buf = b;
  st->bytes = bytes;
  st->ctx = this;
  x.st = st;
  return x;
}

Tensor Context::zeros(DType t, std::vector<int64_t> shape) {
  std::vector<uint8_t> z(std::max<size_t>(1, (size_t)Tensor{nullptr, 0, t, shape}.nbytes()), 0);
  return from_host(t, shape, z.data());
}

void Context::add_source_library(const std::string& name, const std::string& source, bool fast_math) {
  MTLCompileOptions* opt = [[MTLCompileOptions new] autorelease];
  if (fast_math) {
    opt.mathMode = MTLMathModeFast;
    opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast;
  } else {
    opt.mathMode = MTLMathModeSafe;
    opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  }
  opt.languageVersion = MTLLanguageVersion4_0;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev_ newLibraryWithSource:@(source.c_str()) options:opt error:&err];
  if (!lib) {
    throw std::runtime_error("Metal compile failed for " + name + ": " +
                             (err ? std::string(err.localizedDescription.UTF8String) : ""));
  }
  libs_[name] = lib;
}

id<MTLComputePipelineState> Context::pso(const std::string& lib, const std::string& fn) {
  std::string key = lib + "::" + fn;
  auto it = psos_.find(key);
  if (it != psos_.end()) return it->second;
  auto li = libs_.find(lib);
  if (li == libs_.end()) throw std::runtime_error("no library " + lib);
  id<MTLFunction> f = [li->second newFunctionWithName:@(fn.c_str())];
  if (!f) throw std::runtime_error("no function " + fn + " in " + lib);
  NSError* err = nil;
  id<MTLComputePipelineState> p = [dev_ newComputePipelineStateWithFunction:f error:&err];
  [f release];
  if (!p) throw std::runtime_error("pipeline " + fn + ": " + std::string(err.localizedDescription.UTF8String));
  psos_[key] = p;
  return p;
}

void Context::dispatch_threads(id<MTLComputePipelineState> p, size_t n, size_t tg) {
  if (n == 0) return;
  tg = std::min<size_t>(tg, p.maxTotalThreadsPerThreadgroup);
  [encoder() dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  dispatches++;
}

void Context::dispatch_groups(id<MTLComputePipelineState> p, MTLSize groups, MTLSize tg) {
  [encoder() dispatchThreadgroups:groups threadsPerThreadgroup:tg];
  dispatches++;
}

void f32_to_f16(const float* src, uint16_t* dst, size_t n) {
  for (size_t i = 0; i < n; i++) {
    __fp16 h = (__fp16)src[i];
    std::memcpy(&dst[i], &h, 2);
  }
}

float bf16_to_f32(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace idol
