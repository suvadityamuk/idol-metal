# Idol on Metal (MPS): port report

Source commit of this directory: 328bea6 (plus this file). Nothing has been published.

**Summary.** The fast build is 1.75-2.00x faster end to end than the official pipeline on MPS and is closer to it
than the official `torch.compile` path is. The parity mode's `.ply` is byte-identical (sha256) to the reference on
all four configs and is 1.40-1.54x faster than the reference. The kernel package builds with kernel-builder and
passes 34/34 tests. **The native script is incomplete: only the sampling loop is native, and it has no parity mode.**

Setup: M5 Pro, 64 GB, macOS 26.6.2, torch 2.15.0. Reference: `speridlabs/idol-sdk` at 8db7003, unmodified.
Seed 42, 50 steps, cfg 3.0, rescale_t 3.0; `fig-robot-cute.webp` (synthetic) and `food-avocado.webp` (real);
262,144 and 1,048,576 Gaussians.

## Baselines and profile (docs/profile.md)

- Reference as shipped (eager on MPS; the upstream optimize path is CUDA-only): 153-185 s per request. Sampling is
  98.5% of it: 50 steps x 2 CFG passes x 1.37 s per forward. BiRefNet 0.33 s, DINOv3 0.44 s, VAE 0.31 s; decode
  1.1 s at 262k and 6.7-7.2 s at 1M; load 4-6 s; peak footprint 10.4 GB.
- The reference is deterministic on MPS: two fresh-process runs give byte-identical PLYs. Wall time drifts by up to
  14% between sessions (thermal state).
- `torch.compile` works on MPS with no graph breaks: one forward drops from 1,370 to 970 ms. Inductor fuses the
  elementwise chains into 30 Metal kernels; linears stay `addmm`, attention stays the MPS SDPA op. The official
  optimize path forced onto MPS (compile + batched CFG) takes 120-143 s per request.
- Inside one forward: SDPA 39%, about 45% elementwise glue, the rest GEMMs. MPS GEMMs run at about 32 TFLOP/s; a
  Metal 4 `matmul2d` GEMM reaches the same and is bit-identical to torch's. PyTorch's flash attention kernel runs
  at about 26-28 TFLOP/s.

## Changes kept (docs/ab-log.md)

Measured teacher-forced against the reference, interleaved in one process. A reference step (both CFG passes) is
2.9-3.0 s.

| # | change | speed | error vs reference |
|---|---|---|---|
| 1a | Context stream once per request; noise refiner, camera refiner, time embedding shared by both CFG passes | 1.03x | bit-equal |
| 1b | Uncond pass carries 1 context token instead of 4,101 identical ones (fp16 key bias split over two copies) | 1.28x cumulative | v_cond bit-equal; v_uncond rel 3.8e-4 to 5.3e-4; guided v rel 7.8e-4 to 1.2e-3 (Inductor: 5e-3) |
| 2 | Fused Metal kernels keeping PyTorch's rounding points | 1.81x cumulative (1,604 vs 2,905 ms) | v_cond bit-equal |
| 3 | fc1 GEMM with bias + GELU in the epilogue (Metal 4 `matmul2d`) | 1.04x | bit-equal |
| 4 | Fused Gaussian decoder | 1.33x at 262k, 1.08x at 1M (decoder only) | bit-equal |
| 5 | Kernels through the built package instead of `compile_shader` | 1.05x | bit-equal |

Change 1b is the only one that alters arithmetic. Steady-state step: 1,405 ms; estimated floor about 1,250 ms.

Rejected: fp32 attention mask (0.35x, leaves the fast attention kernel); single fp16 log-weight (0.4% weight
error); restructured forward + Inductor instead of hand kernels (1,761 ms/step and less close, 5e-3); attention
tile retuning (no gain); MPP `relaxed_precision` (slower); plain MPP GEMM in place of MPS (equal).

Not attempted: dead rows in the last block (about 1%), octree decoder and front end (about 1.5% together),
int8/fp8 GEMMs.

## Benchmark (docs/benchmark.md, raw rows in logs/bench_summary.json)

One process per row, warm-up excluded, total excludes load. Total seconds (peak footprint, GB).

| config | reference | torch.compile | parity | fast |
|---|---|---|---|---|
| robot 262k, round 1 | 168.9 (10.4) | 132.0 (10.5) | 118.2 (10.0) | 96.3 (10.4), 1.75x |
| robot 262k, round 2 | 175.4 | 139.0 | 122.3 (11.0) | 94.9, 1.85x |
| avocado 262k | 174.6 | 133.8 | 124.4 | 98.1, 1.78x |
| robot 1M | 185.2 | 143.2 | 126.2 | 98.5, 1.88x |
| avocado 1M * | 165.8 | 120.6 | 107.6 | 82.8 (11.0), 2.00x |

\* Run in a later session on a cooler machine: compare within that row only. The other rows are one session.
The fast build is 1.36-1.46x faster than `torch.compile`. Load is 4.2-5.0 s for every variant. One measurement per
1M config.

## Closeness to the reference dump

| config | variant | latent rel L2 | Chamfer (Gaussian positions) |
|---|---|---|---|
| robot 262k | fast | 1.02e-2 (mean abs 2.1e-3, max 1.68) | 1.72e-3 |
| robot 262k | torch.compile | 3.69e-2 (mean abs 1.77e-2, max 2.92) | 1.78e-3 |
| avocado 262k | fast | 6.13e-3 | 1.76e-3 |
| avocado 262k | torch.compile | 7.58e-3 | 1.76e-3 |
| robot 1M | fast / compile | - | 1.02e-3 / 1.08e-3 |
| avocado 1M | fast / compile | - | 1.10e-3 / 1.10e-3 |

Camera max abs error is about 1.5e-3 for both. Anchor-cell overlap is 0.22-0.61 for both (the octree resampling
is discontinuous, so it is low for any non-identical latent). Stated tolerance: the reference's own
`torch.compile` spread; the fast build is inside it on every metric measured. Rendered-view PSNR/SSIM was not
measured (no 3DGS renderer on MPS).

## Parity

- Tier 1: the fast build with change 1b switched off.
- Verified: PLY sha256 equals the reference's on robot and avocado at 262k and 1M; latent, camera and Gaussians
  byte-equal.
- Regression check: `tools/parity_check.sh` exits non-zero on mismatch; it passed.
- Flag: `scripts/run_kernels.py --parity`. `scripts/run_native.sh --parity` refuses (see below).

## Kernel package (kernel/)

- Ops (8, all validating arguments): `resid_ln_mod`, `resid_layernorm`, `repo_rope_qk`, `rmsnorm_heads`,
  `linear_gelu`, `silu_mul`, `gelu_tanh_`, `gated_residual`.
- `check-config` ok; real `build-and-copy` exits 0 with 6 variants (torch 2.10-2.15) from clean commit 934907d;
  `check-builds` exits 0.
- Tests: 34/34 against the builder's torch 2.15 output, exact equality to the torch op chains, with one documented
  tolerance: `repo_rope_qk` is half an ulp off on a few values at 36-38 rows (PyTorch's normalize takes a different
  reduction path there); exact at the model's sizes.
- kernel-builder compiles `.metal` with fast math, which broke exactness; fixed in source with
  `#pragma METAL fp math_mode(safe)` and `contract(on)`.
- Per-op speed vs the torch chain: `linear_gelu` 1.17x, `repo_rope_qk` 14.2x, `resid_ln_mod` 6.2x,
  `rmsnorm_heads` 13.1x.

## Inference scripts (each run once in a fresh shell, robot 262k)

| script | time | result vs reference |
|---|---|---|
| `run_kernels.py --local` | 79.9 s | latent rel L2 1.02e-2, Chamfer 1.72e-3 |
| `run_kernels.py --local --parity` | 107.9 s | PLY sha256 equal to the reference's (byte-identical) |
| `run_native.sh --decode` | 92.8 s native sampling | latent rel L2 1.09e-2, Chamfer 1.72e-3 |

**The native script is incomplete.** No Python runs in the sampling process, but the front end (BiRefNet, DINOv3,
VAE) and the decoders are Python (`tools/native_io.py`). It also needs PyTorch 2.15's metallib for the attention
kernel.

**The native script has no parity mode.** Its glue ops (time embedding, 1-row linears, head, camera MLP) are not
bit-matched to PyTorch. Teacher-forced: step 0 v_cond rel 5.5e-4 (max 3.7e-3), v_uncond rel 4.9e-4; steps 1-2
guided v rel 2.5e-3 and 3.7e-3. Tried: positional embedding is bit-equal; t_mod has 2 of 6,144 values one ulp off;
routing 1-row linears through MPSNDArray matmul changed nothing; cos/sin on the GPU changed nothing; torch's CPU
`exp` differs in 14 of 128 frequencies (not fixed). The remaining source was not located.

## Not done or not verified

- Rendered-view PSNR/SSIM; the published quality metrics (upstream eval set is unnamed).
- Native front end, native decoders, native parity.
- The torch 2.10-2.14 package variants were built but never loaded.
- `kernels benchmark` (needs a Hub repo).
- Any hardware other than M5 Pro / macOS 26.6.2; `linear_gelu` needs Metal 4.

## Release preparation (2026-11-25; nothing published)

- **Kernel version 1.** `build.toml` went from version 0 to 1 (the upload branch is `v{version}` from each variant's
  `metadata.json`), with CARD.md, the test fixture and `run_kernels.py` updated to match. The package was rebuilt
  from clean commit 5b860e2: id `_idol_metal_metal_5b860e2`, version 1, not dirty, kernel sources unchanged since
  934907d. `check-config` and `check-builds` exit 0; the 34 tests pass on the torch 2.15 variant (project venv,
  `kernels` 0.18.3 `get_local_kernel`). Upload manifest: `../release/idol-metal-kernel-manifest.md`.
- **`scripts/infer.py`**: one command from image to `.ply` (`--build fast|parity`, `--local-kernels`, `--check`,
  `--dump`). Robot, synthetic, seed 42, 50 steps, cfg 3.0, 262,144 Gaussians, one run each, kernels from
  `--local-kernels kernel` through `kernels` 0.18.3 `get_local_kernel` in the project venv (logs/infer_*.log):

  | build | request | wall | result |
  |---|---|---|---|
  | parity | 109.4 s | 118.2 s | sha256 d5a02236... equal to the reference run's `.ply`: byte-identical |
  | fast | 86.4 s | 92.4 s | sha256 4dcb89dd... equal to the benchmark's fast rows; latent rel L2 1.02e-2, camera max abs 1.76e-3, Chamfer 1.72e-3 vs the reference dump (logs/infer_fast_closeness.json) |

  Error paths checked: Hub kernel repo not found (it is not published) gives a message pointing to
  `--local-kernels`; no token with an empty HF cache gives idol-sdk's gated-encoder explanation; non-Apple platform,
  macOS < 26 and no MPS (simulated by patching `platform`/`torch`) give explanations. The Hub path itself is untested.
- **README** rewritten for the public repository; media in `assets/` (benchmark chart from the round-1 robot rows;
  point renderings of this Mac's `.ply` files; two gsplat turntables from the CUDA port, captioned as such).
- **Export:** `tools/export_release.sh DEST` (staged at `../release/idol-metal/`).
