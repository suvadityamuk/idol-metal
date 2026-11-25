# Profile: official Idol pipeline on MPS (M5 Pro, 64 GB, macOS 26.6.2, torch 2.15.0)

All numbers synchronised (`torch.mps.synchronize()` around each timed region). Reference = `reference/idol-sdk` at
8db7003, unmodified, fp16 blocks / fp32 norms / bf16 encoders as shipped. On MPS `optimize=True` is a no-op upstream
(the compile/batched-CFG/cuDNN path is CUDA-only), so "as shipped" is eager.

## 1. Whole request, by stage (robot, 262,144 Gaussians, 50 steps, warm process)

Source: `runs/base_robot_262144_default/result.json` (previous agent, same machine, no dump hooks).

| stage | time (s) | share |
|---|---:|---:|
| preprocess (resize, BiRefNet fp16, crop, composite) | 0.33 | 0.2% |
| encode: DINOv3 ViT-H+/16 (bf16) | 0.44 | 0.3% |
| encode: FLUX.2 VAE encoder (bf16) | 0.31 | 0.2% |
| **sample: 50 Euler steps x 2 denoiser passes (CFG)** | **150.89** | **98.5%** |
| decode: octree (8 levels, 4-block decoder) | 0.42 | 0.3% |
| decode: Gaussian decoder (16 blocks, L = 8,192 anchors) | 0.72 | 0.5% |
| to_representation + PLY write | 0.03 | 0.0% |
| **total** | **153.13** | |

Model load (not in total): 3.9-6.3 s. Peak process footprint 10.3 GB, MPS driver allocation 9.4 GB.
At 1,048,576 Gaussians the Gaussian decoder grows to ~6.7 s (32,768 anchors; self-attention over all anchors) and
the octree to ~0.9 s (dump-hooked run; clean numbers in the benchmark).

Run-to-run spread of the reference: two fresh processes give byte-identical PLYs (sha256 equal), so on this device
the reference is deterministic; wall time drifts by up to ~14% between sessions (149.5 s vs 171.2 s sampling in
`logs/ref_robot_262144_{A,B}.log`), which is why all comparisons below are interleaved in one session.

## 2. Inside the sampler: one denoiser forward (batch 1, joint sequence L = 8192 + 4101 + 1 = 12,294)

Eager reference forward: 1,370 ms (3 reps). Sectioned with sync around each module type (`tools/profile_dit.py`;
sync inflates the total to 1,491 ms, used for ranking):

| section | ms | share |
|---|---:|---:|
| SDPA (incl. permutes) | 578 | 38.8% |
| FFN (fc1 + GELU + fc2) | 202 | 13.5% |
| q/k RMSNorm (`F.normalize(x.float())*gamma*scale`, fp32 round trip) | 160 | 10.7% |
| RePo layers (LayerNorm32, 2 fp32 GEMMs, SiLU*mul, fp32 GEMM, angles, polar) | 118 | 7.9% |
| LayerNorm32 (x.float() -> LN -> half) | 77 | 5.1% |
| apply_rotary_emb (complex64 view_as_complex round trip) | 71 | 4.8% |
| qkv GEMM | 71 | 4.8% |
| out GEMM | 28 | 1.9% |
| rest (modulation, gated residuals, cat, head) | ~186 | ~12% |

Op-level ceilings on this GPU (`tools/microbench.py`, `dev/mpp_gemm_probe.py`, `dev/attn_tiles_probe.py`):
- fp16 GEMM through MPS: 24-31 TFLOP/s; a hand-written Metal 4 `matmul2d` (MetalPerformancePrimitives) kernel peaks at
  31.8 TFLOP/s, i.e. the same as torch's `mm` (and bit-identical to it): GEMMs already run at the hardware rate.
- SDPA: PyTorch 2.15's MPS "prefill" attention is already an MPP (`matmul2d`) flash-attention kernel: 21.6-24.7 ms
  at L = 12,294 (~26-28 TFLOP/s). Re-instantiating its template with other tiles (BQ 32/64/128, BK 32/64/128) gives
  nothing better than +-5%. With the prefill kernel disabled (MPSGraph path) the same call takes 602 ms.
- SDPA with an fp32 additive mask leaves the fast kernel (253 ms vs 11.5 ms at L = 8,194); fp16/bool masks stay fast.

So per forward, GEMMs (~260 ms) and attention (~540 ms) are at the hardware rate and **~45% of the eager forward is
elementwise glue** (fp32 round trips, complex RoPE, separate norm/modulation/residual kernels).

## 3. torch.compile (Inductor) on MPS

Works with torch 2.15 (no failure). `compile_for_sampling` (the reference's own compile path: blocks + RePo
compiled, real-valued RoPE) on a batch-1 forward: 970 ms vs 1,370 ms eager (`tools/compile_probe.py`). Generated code
(`logs/inductor/output_code.log`): no graph breaks; Inductor emits 30 Metal kernels that fuse the elementwise chains
(LN+modulation, RoPE+RMSNorm, SiLU*mul, GELU) and calls `extern_kernels.addmm/mm` for every linear and
`aten._scaled_dot_product_attention_math_for_mps` for attention. Gaps left by Inductor:
- GEMM epilogues are not fused (bias inside addmm, GELU and gated residual as separate kernels);
- it does nothing about the model-level redundancy below (it compiles per block);
- its arithmetic differs from eager (reassociated fp32 reductions, fused casts): per-call v error ~1.5e-3 relative.

The official optimize path forced on MPS (compile + batched CFG, `tools/run_variant.py --variant compile`) samples in
143.4 s vs ~150 s eager: on MPS the batch-2 compiled forward is not faster than two batch-1 eager passes by much.

## 4. Redundancy in the request (model level, exact)

- The context stream (cond_embedder + 2 context-refiner blocks over 4,101 tokens) depends only on the image, not on
  the step: recomputed 100 times by the reference.
- The noise refiner (2 blocks over 8,192 tokens), camera refiner and time embedding do not depend on the condition:
  computed twice per step (cond and uncond) by the reference.
- With the condition zeroed (CFG's unconditional pass) all 4,101 context tokens are identical in every block
  (verified: `dev/uncond_rows_check.py`, 24/24 blocks), so the unconditional pass does 50% more token work and
  2.25x more attention work than needed.

## 5. Ranked gap list (what the fast build attacks, in order of share)

| # | gap | share of request | approach | status |
|---|---|---:|---|---|
| 1 | unconditional pass carries 4,101 identical context tokens | ~33% of sampling | one token, attention weight 4,101 via fp16 log-bias split over two keys | done |
| 2 | elementwise glue ~45% of each forward | ~40% of sampling | fused Metal kernels keeping the reference's rounding points | done |
| 3 | context stream recomputed each pass | ~3% | computed once per request | done |
| 4 | noise/camera refiner computed twice per step | ~3% | shared by both passes | done |
| 5 | attention at ~26-28 of ~32 TFLOP/s | 60% of fast step | tile retuning: no gain; left as is | negative |
| 6 | GELU / bias epilogues not fused into GEMMs | ~2% | MPP GEMM with epilogue | see ab-log |
| 7 | dead rows in the last joint block (context queries) | ~1% | compute only latent+camera rows | see ab-log |
| 8 | Gaussian decoder at 1,048,576 (L = 32,768) | ~4% at 1M | same fused-kernel treatment | see ab-log |
| 9 | front end (BiRefNet, DINOv3, VAE), octree, PLY | ~1.5% | not worth the risk at this share | not done |
