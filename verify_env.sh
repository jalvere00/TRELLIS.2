#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

echo "=== All installed extensions ==="
pip list | grep -Ei "flash|nvdiff|cumesh|flex|voxel|nvrtc"

echo ""
echo "=== Import checks ==="
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python -c "import nvdiffrast; print('nvdiffrast OK')"
python -c "import nvdiffrec_render; print('nvdiffrec_render OK')"
python -c "import cumesh; print('cumesh OK')"
python -c "import flex_gemm; print('flex_gemm OK')" 2>/dev/null || python -c "import flexgemm; print('flexgemm OK')" 2>/dev/null || echo "flex_gemm: import name unknown, checking pip..."
pip show flex-gemm flex_gemm flexgemm 2>/dev/null | grep -E "Name|Version" || echo "flex_gemm not found in pip"
python -c "import o_voxel; print('o_voxel OK')"

echo ""
echo "=== PyTorch + CUDA sanity check ==="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
x = torch.randn(4,4).cuda()
print('GPU tensor OK:', x.shape)
"
