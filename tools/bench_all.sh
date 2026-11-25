#!/bin/bash
# Final benchmark: one process per (variant, image, config), all in one session, order flipped between rounds and
# images. Each process: load, 2-step 32,768-Gaussian warm-up (shader/graph compilation, Inductor compile), then one
# timed request (stage timers with sync), outputs saved for tools/closeness.py.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { # variant image ng tag
  local out=runs/bench/$4_$1_$2_$3
  $PY tools/run_variant.py --variant $1 --image $2 --num_gaussians $3 --out $out --save dumps/bench/$4_$1_$2_$3.safetensors \
    > logs/bench/$4_$1_$2_$3.log 2>&1
  echo "[$(date +%H:%M:%S)] $4 $1 $2 $3: $(grep '^run:' logs/bench/$4_$1_$2_$3.log | cut -c1-260)"
}
mkdir -p runs/bench dumps/bench logs/bench
for v in ref compile parity fast; do run $v robot 262144 r1; done
for v in fast parity compile ref; do run $v avocado 262144 r1; done
for v in fast parity compile ref; do run $v robot 262144 r2; done
for v in ref compile parity fast; do run $v robot 1048576 r1; done
echo done
