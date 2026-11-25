// Native Idol denoiser + rectified-flow sampler (fast build): the same restructured computation as fast/dit.py
// (context stream once per request, noise/camera refiners once per step, unconditional context collapsed to one
// token weighted L_c through an fp16 key bias split over two copies), with the fused kernels of
// kernel/idol_metal/idol_kernels.metal, MPS GEMMs and PyTorch's own prefill attention kernel.
#pragma once
#include <string>
#include <vector>

#include "ops.h"
#include "safetensors.h"

namespace idol {

struct SampleConfig {
  int steps = 50;
  double cfg = 3.0;
  double rescale_t = 3.0;
  int start_step = 0;   // debugging / teacher forcing: begin at this step of the grid
  int run_steps = -1;   // and run this many steps (-1: to the end)
};

struct SampleResult {
  std::vector<float> latent;  // [8192, 16]
  std::vector<float> camera;  // [5]
  std::vector<float> v_first[2];  // v_cond, v_uncond latent of the first executed step (diagnostics)
  double seconds_prepare = 0, seconds_sampling = 0;
};

class NativeDenoiser {
 public:
  NativeDenoiser(Context& c, const st::File& weights);
  ~NativeDenoiser();
  // feature1 [Lc, 1280], feature2 [Lc, 128], noise latent [8192, 16], camera [5] (host float)
  SampleResult sample(const float* feature1, const float* feature2, int64_t Lc, const float* noise_latent,
                      const float* noise_camera, const SampleConfig& cfg);

 private:
  struct Impl;
  Impl* p_;
};

}  // namespace idol
