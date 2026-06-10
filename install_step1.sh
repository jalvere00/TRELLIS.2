#!/bin/bash
# Step 1: Install nvcc + remaining basic deps
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

echo "=== Installing cuda-nvcc 12.4 via conda ==="
conda install -c "nvidia/label/cuda-12.4.0" cuda-nvcc -y

echo "=== Verifying nvcc ==="
which nvcc && nvcc --version || echo "nvcc still not found after conda install"

echo "=== Setting CUDA_HOME ==="
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME=$CUDA_HOME"

echo "=== Installing utils3d ==="
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

echo "=== Done with step 1 ==="
