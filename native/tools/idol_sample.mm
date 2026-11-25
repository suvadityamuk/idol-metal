// idol_sample: native (no Python) Idol denoiser sampling on Metal.
//   idol_sample --weights denoiser.safetensors --inputs in.safetensors --out out.safetensors
//               [--steps 50] [--cfg 3.0] [--rescale_t 3.0] [--kernels idol_kernels.metal] [--torch-metallib dir]
// in.safetensors: encode/feature1 [1, Lc, 1280], encode/feature2 [1, Lc, 128], sample/noise/latent [1, 8192, 16],
// sample/noise/camera [1, 1, 5] (float32; a reference dump has exactly these keys).
// out.safetensors: sample/final/latent [1, 8192, 16], sample/final/camera [1, 1, 5] (float32).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "../src/sampler.h"

using namespace idol;

static void write_safetensors(const std::string& path,
                              const std::vector<std::pair<std::string, std::pair<std::vector<int64_t>, const std::vector<float>*>>>& ts) {
  std::string hdr = "{";
  size_t off = 0;
  for (size_t i = 0; i < ts.size(); i++) {
    const auto& [name, v] = ts[i];
    size_t n = v.second->size() * 4;
    std::string shp;
    for (size_t j = 0; j < v.first.size(); j++) shp += (j ? "," : "") + std::to_string(v.first[j]);
    hdr += (i ? "," : "") + std::string("\"") + name + "\":{\"dtype\":\"F32\",\"shape\":[" + shp +
           "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + n) + "]}";
    off += n;
  }
  hdr += "}";
  while (hdr.size() % 8) hdr += ' ';
  std::ofstream f(path, std::ios::binary);
  uint64_t hl = hdr.size();
  f.write((const char*)&hl, 8);
  f.write(hdr.data(), hdr.size());
  for (const auto& t : ts) f.write((const char*)t.second.second->data(), t.second.second->size() * 4);
  if (!f) throw std::runtime_error("cannot write " + path);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string weights, inputs, out, kernels = "kernel/idol_metal/idol_kernels.metal", tlib = "third_party/torch_metallib";
    SampleConfig cfg;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--weights") weights = next();
      else if (a == "--inputs") inputs = next();
      else if (a == "--out") out = next();
      else if (a == "--steps") cfg.steps = std::stoi(next());
      else if (a == "--cfg") cfg.cfg = std::stod(next());
      else if (a == "--rescale_t") cfg.rescale_t = std::stod(next());
      else if (a == "--kernels") kernels = next();
      else if (a == "--start-step") cfg.start_step = std::stoi(next());
      else if (a == "--run-steps") cfg.run_steps = std::stoi(next());
      else if (a == "--torch-metallib") tlib = next();
      else {
        fprintf(stderr, "unknown argument %s\n", a.c_str());
        return 2;
      }
    }
    if (weights.empty() || inputs.empty() || out.empty()) {
      fprintf(stderr, "usage: idol_sample --weights denoiser.safetensors --inputs in.safetensors --out out.safetensors "
                      "[--steps N --cfg W --rescale_t R --kernels file.metal --torch-metallib dir]\n");
      return 2;
    }
    try {
      Context c(tlib);
      if (!c.has_torch_libs()) throw std::runtime_error("PyTorch metallib not found in " + tlib);
      std::ifstream ks(kernels);
      if (!ks) throw std::runtime_error("cannot read " + kernels);
      std::stringstream ss;
      ss << ks.rdbuf();
      c.add_source_library("fast", ss.str(), false);
      printf("device: %s; build: native fast (fused kernels, MPS GEMM, PyTorch prefill attention)\n",
             c.device().name.UTF8String);
      st::File w(weights), in(inputs);
      NativeDenoiser dn(c, w);
      const auto& f1 = in.get("encode/feature1");
      const auto& f2 = in.get("encode/feature2");
      const auto& nl = in.get("sample/noise/latent");
      const auto& nc = in.get("sample/noise/camera");
      for (auto* t : {&f1, &f2, &nl, &nc})
        if (t->dtype != "F32") throw std::runtime_error("inputs must be float32");
      const int64_t Lc = f1.shape[1];
      auto r = dn.sample((const float*)f1.data, (const float*)f2.data, Lc, (const float*)nl.data,
                         (const float*)nc.data, cfg);
      printf("prepare %.2f s, sampling %.2f s (%d steps, cfg %.2f, rescale_t %.2f)\n", r.seconds_prepare,
             r.seconds_sampling, cfg.steps, cfg.cfg, cfg.rescale_t);
      write_safetensors(out, {{"sample/final/latent", {{1, 8192, 16}, &r.latent}},
                              {"sample/final/camera", {{1, 1, 5}, &r.camera}},
                              {"debug/first_step/v_cond", {{1, 8192, 16}, &r.v_first[0]}},
                              {"debug/first_step/v_uncond", {{1, 8192, 16}, &r.v_first[1]}}});
      printf("wrote %s\n", out.c_str());
    } catch (const std::exception& e) {
      fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
