#!/bin/bash
# Runs the GEMM parity probe on four linears of the first DiT call (dev dump).
cd "$(dirname "$0")"
W=$(ls -d ~/.cache/huggingface/hub/models--speridlabs--idol/snapshots/*/)denoiser.safetensors
D=../../dumps/ref_dev_A.safetensors
IN_KEY=dit/call000/blocks.0.attn/in0 ./gemm_probe $D $W dit/call000/blocks.0.attn.qkv blocks.0.attn.qkv f16 bias
IN_KEY=dit/call000/repo_layers.0.norm/out ./gemm_probe $D $W dit/call000/repo_layers.0.gate_map repo_layers.0.gate_map f32
IN_KEY=dit/call000/blocks.0.mlp/in0 ./gemm_probe $D $W dit/call000/blocks.0.mlp.mlp.0 blocks.0.mlp.mlp.0 f16 bias
IN_KEY=sample/noise/latent ./gemm_probe $D $W dit/call000/input_layer input_layer f32 bias
