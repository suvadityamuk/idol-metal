#!/bin/bash
# Native Metal run of the Idol denoiser sampling loop (no Python in the sampling process).
#
#   scripts/run_native.sh [--image robot|avocado|PATH] [--seed 42] [--steps 50] [--cfg 3.0] [--rescale_t 3.0]
#                         [--num_gaussians 262144] [--inputs FILE] [--out DIR] [--decode]
#
# INCOMPLETE native port, stated plainly: only the denoiser + CFG/Euler loop (98% of the request's time) runs in
# the native binary (native/build/idol_sample: Objective-C++, Metal, MPS GEMMs, the fused kernels of
# kernel/idol_metal/idol_kernels.metal, PyTorch's prefill attention kernel from its metallib).
#   - its inputs (encoder features + noise) come from the official Python front end: `tools/native_io.py prepare`
#     (resize, BiRefNet, crop, DINOv3, FLUX.2 VAE, seeded noise), run once per image unless --inputs is given;
#   - --decode turns the native latent into a PLY with the official Python decoders (`tools/native_io.py decode`).
# Neither of those is native.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=robot; SEED=42; STEPS=50; CFG=3.0; RESCALE=3.0; NG=262144; INPUTS=""; OUT=runs/native; DECODE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE=$2; shift 2;; --seed) SEED=$2; shift 2;; --steps) STEPS=$2; shift 2;;
    --cfg) CFG=$2; shift 2;; --rescale_t) RESCALE=$2; shift 2;; --num_gaussians) NG=$2; shift 2;;
    --inputs) INPUTS=$2; shift 2;; --out) OUT=$2; shift 2;; --decode) DECODE=1; shift;;
    --parity) echo "error: no native parity mode: the native glue (time embedding, 1-row linears, head) is not bit-matched to PyTorch (v_cond rel 5.5e-4 per step). Use: python scripts/run_kernels.py --local --parity" >&2; exit 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "error: needs macOS on Apple silicon (Metal)" >&2; exit 1; }
xcrun -f metal >/dev/null 2>&1 || { echo "error: Xcode with the Metal toolchain is required (xcodebuild -downloadComponent MetalToolchain)" >&2; exit 1; }
PY=${PYTHON:-.venv/bin/python}
mkdir -p "$OUT"
# 1. build the native binary if missing
[ -x native/build/idol_sample ] || make -C native build/idol_sample
# 2. PyTorch's metallib (attention kernel) and the weights
TL=third_party/torch_metallib
if [ ! -f $TL/metal_40.metallib ]; then
  echo "error: $TL/metal_40.metallib missing (PyTorch 2.15 Metal library with the attention kernel, taken from a PyTorch 2.15 build; it is not shipped with this repository)" >&2; exit 1
fi
W=$(ls -d ~/.cache/huggingface/hub/models--speridlabs--idol/snapshots/*/ 2>/dev/null | head -1)denoiser.safetensors
if [ ! -f "$W" ]; then hf download speridlabs/idol denoiser.safetensors config.json >/dev/null; W=$(ls -d ~/.cache/huggingface/hub/models--speridlabs--idol/snapshots/*/ | head -1)denoiser.safetensors; fi
# 3. inputs (Python front end, not native)
if [ -z "$INPUTS" ]; then
  INPUTS=$OUT/inputs_$(basename "$IMAGE")_s$SEED.safetensors
  [ -f "$INPUTS" ] || $PY tools/native_io.py prepare --image "$IMAGE" --seed "$SEED" --out "$INPUTS"
fi
# 4. native sampling
native/build/idol_sample --weights "$W" --inputs "$INPUTS" --out "$OUT/latent.safetensors" --steps "$STEPS" \
  --cfg "$CFG" --rescale_t "$RESCALE" --kernels kernel/idol_metal/idol_kernels.metal --torch-metallib $TL
# 5. optional decode to PLY (Python decoders, not native)
if [ $DECODE = 1 ]; then
  $PY tools/native_io.py decode --latent "$OUT/latent.safetensors" --inputs "$INPUTS" --num_gaussians "$NG" --out "$OUT/gaussians"
fi
