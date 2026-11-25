#!/bin/bash
# Stage 1 batch: every GPU run goes through the shared lock wrapper, one at a time.
cd "$(dirname "$0")/.."
G=../gpu_run.sh
PY=.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1. RNG replay self-check: the reference replaying its own random numbers must reproduce its dump
log replay
$G $PY tools/ref_run.py --image robot --steps 4 --num_gaussians 32768 --warmup none --out runs/ref_dev_replay \
  --dump dumps/ref_dev_replay.safetensors --replay dumps/ref_dev_A.safetensors > logs/ref_dev_replay.log 2>&1
$PY tools/compare.py dumps/ref_dev_A.safetensors dumps/ref_dev_replay.safetensors --show 5 > logs/cmp_dev_replay.txt 2>&1
log "replay: $(head -1 logs/cmp_dev_replay.txt)"

# 2. baseline timings, as shipped: default and --no_optimize, one process per (config, variant), warm-up first
for img in robot avocado; do for ng in 262144 1048576; do for v in default no_optimize; do
  flag=""; [ $v = no_optimize ] && flag="--no_optimize"
  $G $PY tools/ref_run.py --image $img --steps 50 --num_gaussians $ng $flag --repeat 1 \
    --out runs/base_${img}_${ng}_$v > logs/base_${img}_${ng}_$v.log 2>&1
  log "base $img $ng $v: $(grep '^run 0' logs/base_${img}_${ng}_$v.log | cut -c1-400)"
done; done; done

# 3. reference spread on MPS (configurations the reference code supports), robot @ 262144
$G $PY tools/ref_run.py --image robot --steps 50 --num_gaussians 262144 --warmup none --real_valued_rope \
  --out runs/spread_realrope --dump dumps/spread_realrope.safetensors > logs/spread_realrope.log 2>&1
log "spread realrope: $(grep '^run 0' logs/spread_realrope.log | cut -c1-300)"
$G $PY tools/ref_run.py --image robot --steps 50 --num_gaussians 262144 --warmup none --dinov3_dtype fp32 \
  --out runs/spread_dinofp32 --dump dumps/spread_dinofp32.safetensors > logs/spread_dinofp32.log 2>&1
log "spread dinofp32: $(grep '^run 0' logs/spread_dinofp32.log | cut -c1-300)"
PYTORCH_MPS_DISABLE_PREFILL_ATTENTION=1 $G $PY tools/ref_run.py --image robot --steps 50 --num_gaussians 262144 \
  --warmup none --out runs/spread_graphsdpa --dump dumps/spread_graphsdpa.safetensors > logs/spread_graphsdpa.log 2>&1
log "spread graphsdpa: $(grep '^run 0' logs/spread_graphsdpa.log | cut -c1-300)"

# 4. front-end noise floor (15 fresh processes per image)
tools/run_enc_noise.sh > logs/enc_noise_driver.log 2>&1
log "enc noise done"

# 5. op-level MPS profile of a short run (operation stats with GPU time, CPU fallback stats)
PYTORCH_MPS_LOG_PROFILE_INFO=$((16|64|128)) $G $PY tools/ref_run.py --image robot --steps 2 --num_gaussians 262144 \
  --warmup none --out runs/profile_s2 > logs/mps_profile_s2.log 2>&1
log "profile done"
