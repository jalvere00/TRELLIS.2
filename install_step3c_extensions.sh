#!/bin/bash
# Step 3c: Install cumesh, flexgemm, o-voxel (nvdiffrast + nvdiffrec already done)
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH
# Limit parallel compile jobs to avoid OOM (exit code 137)
export MAX_JOBS=1

WORKDIR=/mnt/c/Users/jamee/Documents/GitHub/TRELLIS.2
mkdir -p /tmp/trellis_extensions

echo "=== [1/3] Installing CuMesh (MAX_JOBS=1 to avoid OOM) ==="
rm -rf /tmp/trellis_extensions/CuMesh/build 2>/dev/null || true
if [ ! -d /tmp/trellis_extensions/CuMesh ]; then
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/trellis_extensions/CuMesh --recursive
fi
pip install /tmp/trellis_extensions/CuMesh --no-build-isolation
echo "CuMesh done"

echo "=== [2/3] Installing FlexGEMM (MAX_JOBS=1) ==="
if [ ! -d /tmp/trellis_extensions/FlexGEMM ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/trellis_extensions/FlexGEMM --recursive
fi
pip install /tmp/trellis_extensions/FlexGEMM --no-build-isolation
echo "FlexGEMM done"

echo "=== [3/3] Installing o-voxel (MAX_JOBS=1) ==="
rm -rf /tmp/trellis_extensions/o-voxel 2>/dev/null || true
cp -r $WORKDIR/o-voxel /tmp/trellis_extensions/o-voxel
pip install /tmp/trellis_extensions/o-voxel --no-build-isolation
echo "o-voxel done"

echo ""
echo "=== All extensions installed! ==="
pip list | grep -Ei "nvdiff|cumesh|flexgemm|ovoxel|o.voxel|flash"
