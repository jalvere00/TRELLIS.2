#!/bin/bash
# Step 2: Build flash-attn 2.7.3 from source
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

# Install cuda libs needed for compilation
echo "=== Installing cuda-libraries-dev ==="
conda install -c "nvidia/label/cuda-12.4.0" cuda-libraries-dev -y 2>/dev/null || echo "skipped (may already be ok)"

echo "=== Building flash-attn 2.7.3 (targeting sm_89 only, MAX_JOBS=4) ==="
echo "=== This will take ~30-60 minutes. Go get a coffee. ==="
MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="8.9" pip install flash-attn==2.7.3 --no-build-isolation

echo "=== flash-attn installed ==="
python -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"
