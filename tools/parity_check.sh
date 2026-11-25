#!/bin/bash
# Parity regression check: the parity mode of the fast build must write a PLY byte-identical (sha256) to the
# reference's on the dev config (robot, seed 42, 4 steps, 32,768 Gaussians; reference sha from runs/ref_dev_A).
# Exits non-zero on mismatch. Usage: tools/parity_check.sh [package|shader]
set -euo pipefail
cd "$(dirname "$0")/.."
REF=2844ebf2cd067da8b2dadd51ddfd5c530069b38c57daf2180f7e376c7dedd618
.venv/bin/python tools/run_variant.py --variant parity --kernels "${1:-package}" --steps 4 --num_gaussians 32768 \
  --warmup none --out runs/parity_check > runs/parity_check.log 2>&1
GOT=$(shasum -a 256 runs/parity_check/run0.ply | cut -d' ' -f1)
if [ "$GOT" = "$REF" ]; then echo "PARITY OK: PLY byte-identical to the reference ($GOT)"; else echo "PARITY FAIL: $GOT != $REF"; exit 1; fi
