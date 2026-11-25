# A/B log (fast build, Metal / MPS)

Method: `tools/ab_dit.py` runs the fast denoiser and the reference denoiser in one process, interleaved (order flipped
each repetition, first repetition discarded as warm-up), on the reference's own inputs at each recorded step
(teacher-forced, `dumps/ref_dev_A.safetensors`, robot), and compares v_cond, v_uncond and the guided velocity
v = 4 v_cond - 3 v_uncond with the reference's. "step" = one sampling step = both CFG passes. Times are medians
(min in brackets) over the repetitions. Errors are on the latent velocity (|v| up to ~5) at steps 0 and 1;
"rel" = relative L2. End-to-end checks of kept changes are in section E.

Reference spread for scale (same harness): the reference's own compile path (Inductor) differs from eager by
rel 1.4-1.8e-3 per call on v_cond and 5e-3 on the guided velocity (`--impl torch --compile`, below).

## Kept

| # | change | step time fast vs ref (ms) | speed-up | error vs reference |
|---|---|---|---:|---|
| 1a | context stream once per request; noise refiner, camera refiner, t-embedding once per step and shared by both passes (pure reuse) | 2824 (2608) vs 2898 (2718) | 1.03x | **bit-equal** (v_cond, v_uncond, guided) |
| 1b | + unconditional pass with ONE context token: attention weight L_c = 4101 through an fp16 additive key bias split over two copies of the token (b = 8.3125, 3.2773; exp sum = 4101 to 2e-6 rel) | 2325 (2140) vs 2984 (2771) | 1.28x | v_cond bit-equal; v_uncond rel 3.8e-4 / 5.3e-4, max 2.5e-3 / 6.6e-3; guided rel 7.8e-4 / 1.2e-3 |
| 2 | + fused Metal kernels (`native/kernels/idol_kernels.metal`, via `torch.mps.compile_shader`): residual+LayerNorm32+adaLN+RePo-LayerNorm in one pass; RePo angles+RoPE+RMSNorm for q and k in one pass; in-place GELU; MPS GEMMs and SDPA unchanged | 1604 (1424) vs 2905 (2785) | 1.81x | identical to 1b: v_cond still **bit-equal** to the reference (the kernels keep every rounding point) |
| 3 | + fc1 GEMM with bias + GELU fused in the epilogue (`linear_bias_gelu_f16`, Metal 4 `matmul2d` with an fp32 accumulator, `half(acc + float(bias))`, then the GELU of `exact_gelu_tanh<half>`) | in-process toggle (`tools/ab_toggle.py fast.ops FUSE_GELU False True`): 1910 (1606) vs 1982 (1730) ms/step, measured while a Nix build loaded the CPU | 1.04x (min 1.08x) | **bit-equal** to #2 (v_cond and v_uncond byte-compared); the kernel alone is bit-equal to `F.linear` + GELU at M = 37, 4101, 8192, 8195, 12294 and takes 3.25 vs 4.09 ms at 12294x1024x4096 |
| 4 | Gaussian decoder (16 TransformerCrossBlocks) with the fused kernels: residual + LayerNorm32 (plain or affine) in one pass, head RMSNorm kernel for q/k (self and cross), fused fc1+GELU; MPS GEMMs and SDPA (`fast/decoder.py`) | teacher-forced on the reference anchors + latent (`tools/ab_decoder.py`): 476 vs 631 ms at 8,192 anchors (262,144 Gaussians); 4,907 vs 5,284 ms at 32,768 anchors (1,048,576) | 1.33x / 1.08x | **bit-equal** decoder output (and the reference decoder reproduces its dump bit for bit) |
| 5 | kernels dispatched through the kernel-builder package (`fast.ops.use_package`, what `scripts/run_kernels.py` runs) instead of `torch.mps.compile_shader` | in-process interleaved (`dev/ab_pkg_vs_shader.py`): 1576 (1383) vs 1649 (1441) ms/step | 1.05x | **bit-equal** (velocities byte-compared; dev end-to-end run byte-identical PLY/latent) |

Parity mode (Amendment 3): the fast build with change 1b (collapsed unconditional context) switched off; all other
changes are bit-equal. Teacher-forced: v_cond, v_uncond and guided v **bit-equal** at steps 0 and 1, 2112 (1995) vs
3023 (2941) ms/step (1.43x). End to end (dev config) the PLY is byte-identical to the reference (sha256
2844ebf2...); `tools/parity_check.sh` is the regression check. Full configs: see docs/benchmark.md.

Steady-state step (8 steps back to back, no per-step sync, `dev/steady_step.py`): 1,405 ms. Estimated floor with
MPS GEMM + SDPA at their measured rates: ~1,250 ms.

## Rejected / negative

| idea | result | why dropped |
|---|---|---|
| collapsed uncond context with an fp32 log(4101) mask | 7,914 ms/step (0.35x) | fp32 masks leave the MPP attention kernel (253 ms vs 11.5 ms per call at L = 8,194) |
| single fp16 log(4101) bias | not run end to end | fp16 spacing at 8.3 is 0.0078: 0.4% error in the weight; replaced by the two-key split |
| restructured forward (1b) + Inductor (`torch.compile` of the block) instead of hand kernels | 1761 (1678) ms/step | slower than #2 (1604 / 1424) and less close (guided rel 5.0e-3 vs 1.2e-3, v_cond no longer bit-equal) |
| re-tuned attention tiles (torch's MPP flash-attention template re-instantiated with BQ 32/64/128 x BK 32/64/128) | best 21.86 ms vs torch 21.61 ms at L = 12,294; 9.78 vs 9.74 ms at 8,195; BK = 128 is 3.6x slower | no tile beats the shipped one beyond noise (`dev/attn_tiles_probe.py`) |
| MPP `matmul2d` GEMM, `relaxed_precision = true` | 28.1 vs 31.8 TFLOP/s (strict) | no faster; GEMMs already at the rate of torch's `mm` |
| hand-written MPP GEMM instead of MPS `F.linear` | 31.8 vs 31.9 TFLOP/s (12294x1024x4096), bit-identical | no gain on its own; only useful to fuse epilogues (see below) |
