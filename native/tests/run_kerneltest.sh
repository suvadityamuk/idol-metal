#!/bin/bash
# Native kernel parity test against the dev reference dump (first DiT call).
cd "$(dirname "$0")/.."
W=$(ls -d ~/.cache/huggingface/hub/models--speridlabs--idol/snapshots/*/)denoiser.safetensors
exec build/kerneltest ../kernel/idol_metal/idol_kernels.metal ../third_party/torch_metallib ../dumps/ref_dev_A.safetensors "$W"
