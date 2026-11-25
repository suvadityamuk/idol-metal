#!/bin/bash
# 15 fresh-process runs of preprocess + encoders per image, each under the shared GPU lock.
cd "$(dirname "$0")/.."
for i in $(seq -w 1 15); do
  for img in robot avocado; do
    ../gpu_run.sh .venv/bin/python tools/enc_noise.py --image $img --out dumps/enc_noise/${img}_$i.safetensors > logs/enc_noise_${img}_$i.log 2>&1 || echo "FAILED $img $i"
  done
done
echo done
