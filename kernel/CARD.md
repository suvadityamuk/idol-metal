---
license: apache-2.0
tags:
  - kernels
  - metal
  - mps
---

# idol-metal

Fused Metal (Apple silicon, PyTorch MPS) kernels for the [Idol](https://github.com/speridlabs/idol-sdk) image-to-3D
model's denoiser and Gaussian-decoder blocks. Each op replaces a chain of PyTorch MPS ops and keeps that chain's
rounding points (values PyTorch would store as half/float are rounded to that type in registers), so on the shapes
tested the results are byte-equal to the PyTorch chain; only memory traffic and dispatch count change.

| op | replaces |
|---|---|
| `resid_ln_mod(x, scale1p, shift, h=, gate=, repo_weight=, repo_bias=)` | `x + h * gate`; `LayerNorm32(x) * (1 + scale) + shift` (adaLN); RePo's fp32 `LayerNorm` of the same row |
| `resid_layernorm(x, h=, weight=, bias=)` | `x + h`; `LayerNorm32(x)` (plain or affine) |
| `repo_rope_qk(qkv, delta, ft, fd, axis, q_gamma, k_gamma, scale)` | RePo rotary angles + complex RoPE + `F.normalize(.float()) * gamma * scale` for q and k (head dim 64) |
| `rmsnorm_heads(src, gamma, scale, col_offset=)` | per-head `F.normalize(.float()) * gamma * scale` (no RoPE) |
| `linear_gelu(x, weight, bias)` | `gelu(F.linear(x, weight, bias), approximate='tanh')`, GELU in the GEMM epilogue (Metal 4 `matmul2d`) |
| `silu_mul(gate, content)` | `F.silu(gate) * content` (fp32) |
| `gelu_tanh_(x)` | in-place `gelu(x, approximate='tanh')` (half) |
| `gated_residual(x, h, gate)` | `x + h * gate` (half) |

```python
from kernels import get_kernel
idol = get_kernel("suvadityamuk/idol-metal", version=1, trust_remote_code=True)
y = idol.linear_gelu(x, w1, b1)
```

## Tested hardware and limits

- **Tested on one machine only:** Apple M5 Pro, 64 GB, macOS 26.6.2, PyTorch 2.15.0. Nothing else was tested.
- **Metal 4 is required.** `linear_gelu` uses MetalPerformancePrimitives (`matmul2d`), which needs macOS 26 or later.
- **Variants:** the package is built for torch 2.10 to 2.15 (`torch21x-metal-aarch64-darwin`). Only the torch 2.15
  variant was loaded and tested (34/34 tests, exact equality to the PyTorch op chains, with one documented half-ulp
  tolerance for `repo_rope_qk` at 36-38 rows), from a local build and again from this repository with `kernels`
  0.18.3. The torch 2.10 to 2.14 variants are built but have never been loaded.
- Inputs must be contiguous MPS tensors of the dtypes stated in each docstring; arguments are validated.
- The ops are specific to Idol's shapes (width 1024, head dimension 64).

## Benchmarks

End to end, seconds per request (image path in, Gaussians out; load, warm-up and the PLY write excluded), for the
source repository's pipeline built on these kernels against the official pipeline on MPS:

| config | official | official, `torch.compile` | parity (bit-exact) | fast |
|---|---|---|---|---|
| robot 262k, round 1 | 168.9 | 132.0 | 118.2 | **96.3, 1.75x** |
| robot 262k, round 2 | 175.4 | 139.0 | 122.3 | **94.9, 1.85x** |
| avocado 262k | 174.6 | 133.8 | 124.4 | **98.1, 1.78x** |
| robot 1M | 185.2 | 143.2 | 126.2 | **98.5, 1.88x** |
| avocado 1M \* | 165.8 | 120.6 | 107.6 | **82.8, 2.00x** |

Apple M5 Pro, 64 GB, macOS 26.6.2, PyTorch 2.15.0, idol-sdk 8db7003; seed 42, 50 steps, cfg 3.0. One process per row,
order flipped between rounds and images, one measurement per row; speed-up against the official pipeline in the same
row. \* Ran later in a cooler state of the machine. Wall time drifted by up to 14% between sessions, so compare
within a row. Peak process footprint: 10.0-11.0 GB (official: 10.4 GB).

## Using them for the whole model

The source repository, [github.com/suvadityamuk/idol-metal](https://github.com/suvadityamuk/idol-metal), has:
- `scripts/infer.py`: one command from an image to a `.ply`. The fast build is 1.75-2.00x faster end to end than
  the official pipeline on MPS on the tested machine; `--build parity` writes a `.ply` byte-identical (sha256) to the
  official pipeline's on the four fixed configs (two example images, 262,144 and 1,048,576 Gaussians).
- `scripts/run_native.sh`: experimental. Only the denoiser sampling loop is native (Objective-C++/Metal); the front
  end and the decoders are Python, it needs PyTorch 2.15's metallib for attention, and it has no parity mode.

Speed, closeness and exactly what was verified are in that repository's README and REPORT.md.

## Source and licence

Apache-2.0. Kernels for [Idol](https://huggingface.co/speridlabs/idol) by Speri Labs
([idol-sdk](https://github.com/speridlabs/idol-sdk), Apache-2.0); not affiliated with Speri Labs. Portions of the
Metal source mirror PyTorch's MPS kernels (BSD-3-Clause, Copyright (c) Meta Platforms, Inc. and affiliates).
