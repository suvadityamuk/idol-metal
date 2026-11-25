#!/bin/bash
cd "$(dirname "$0")"
L=../../third_party/torch_metallib/metal_40.metallib; D=../../dumps/ref_dev_A.safetensors
for s in sdpa00 sdpa02 sdpa04; do ./attn_probe $L $D dit/call000/fn/$s; ./attn_probe $L $D dit/call000/fn/$s blhd; done
