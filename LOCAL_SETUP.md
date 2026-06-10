# TRELLIS.2 Local Setup Guide

This guide documents how to set up and run TRELLIS.2 (4B-parameter image-to-3D) locally on a Linux machine or Windows 11 with WSL2.

## Hardware Requirements

| Component | Minimum | Tested |
|---|---|---|
| GPU VRAM | 12 GB | 12 GB (RTX 4070) |
| System RAM | 24 GB | 32 GB |
| Disk (weights + build) | ~30 GB | — |

> **Note:** The full pipeline offloads model stages to CPU between steps (`low_vram=True` is enabled automatically). 12 GB VRAM is sufficient at `pipeline_type='512'` resolution.

---

## 1. Windows / WSL2 Prerequisites

If running on Windows, all steps below must be executed inside **WSL2 Ubuntu**.

### 1a. WSL2 Memory Configuration

WSL2 defaults to ~50% of physical RAM. TRELLIS.2 needs ~20 GB during inference. Create or edit `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=8GB
```

Then restart WSL: `wsl --shutdown` in PowerShell, then reopen your terminal.

### 1b. NVIDIA Drivers

Install the latest NVIDIA GPU driver on **Windows** (not inside WSL). WSL2 forwards the GPU automatically. Inside WSL, verify:

```bash
nvidia-smi
```

---

## 2. Conda Environment

### 2a. Install Miniconda (if not present)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
```

### 2b. Create the Environment

```bash
conda create -n trellis2 python=3.10 -y
conda activate trellis2
```

### 2c. Install PyTorch 2.6 + CUDA 12.4

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2d. Install Base Python Dependencies

```bash
pip install transformers==5.5.0 diffusers accelerate safetensors huggingface_hub
pip install trimesh[easy] imageio imageio-ffmpeg pillow einops kornia timm lpips
pip install gradio==6.0.1 psutil opencv-python-headless openexr
```

> **Transformers version pinning:** `transformers>=5.6` uses `torch.float8_e8m0fnu` which does not exist in PyTorch 2.6. Pin to `5.5.0`.

---

## 3. CUDA Extensions

All CUDA extensions require `nvcc` to be on `PATH`. Install it into the conda env:

```bash
bash install_step1.sh
```

This installs `cuda-nvcc 12.4` via conda and `utils3d` from source.

### 3a. Flash Attention (build from source, ~30–60 min)

No prebuilt wheel exists for torch 2.6 + CUDA 12.4 + Python 3.10. Build from source:

```bash
bash install_step2_flash.sh
```

This targets `sm_89` (RTX 4070 / Ada Lovelace). Adjust `TORCH_CUDA_ARCH_LIST` for other GPUs:
- RTX 30xx (Ampere): `"8.6"`
- RTX 40xx (Ada): `"8.9"`
- A100: `"8.0"`

### 3b. Remaining CUDA Extensions (CuMesh, FlexGEMM, o-voxel)

```bash
bash install_step3c_extensions.sh
```

Key environment variables set inside the script:
- `LIBRARY_PATH=$CONDA_PREFIX/lib/stubs` — provides `libcuda.so` stub for linking
- `MAX_JOBS=1` — prevents OOM (exit 137) during compilation on machines with limited RAM

### 3c. nvdiffrast and nvdiffrec_render

These ship with the repo. Install them from the local directories:

```bash
conda activate trellis2
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH
export MAX_JOBS=1

pip install extensions/nvdiffrast --no-build-isolation
pip install extensions/nvdiffrec_render --no-build-isolation
```

### 3d. Verify All Extensions

```bash
bash verify_env.sh
```

Expected output: all six extensions import without errors (flash_attn, nvdiffrast, nvdiffrec_render, cumesh, flex_gemm, o_voxel).

---

## 4. HuggingFace Gated Model Access

TRELLIS.2 uses two gated HuggingFace models. You must request access before running:

| Model | URL | Notes |
|---|---|---|
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m | Requires manual review (~1 day) |
| `briaai/RMBG-2.0` | https://huggingface.co/briaai/RMBG-2.0 | Accept license on the page (instant) |

Once approved, set your HuggingFace token so it persists across sessions. The recommended approach is to add it to your conda env's activate script:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export HF_TOKEN=hf_YOUR_TOKEN_HERE' > $CONDA_PREFIX/etc/conda/activate.d/hf_token.sh
chmod 600 $CONDA_PREFIX/etc/conda/activate.d/hf_token.sh
```

**Do not commit this file or your token to git.**

Download RMBG-2.0 to the HF cache ahead of time (optional but avoids download during inference):

```bash
bash download_extra_deps.sh
```

---

## 5. Download Pretrained Weights

The TRELLIS.2-4B checkpoint is ~16 GB.

```bash
bash download_weights.sh
```

This downloads to `pretrained/TRELLIS.2-4B/` inside the repo directory. The `pretrained/` directory is gitignored.

---

## 6. Run Inference

### Quick Test (included example image)

```bash
bash test_example.sh
```

This runs the full pipeline on `assets/example_image/T.png` at 512³ resolution and writes:
- `sample.mp4` — PBR-rendered turntable video (15 fps)
- `sample.glb` — textured mesh (WebP textures, ~1M faces after decimation)

### Runtime Notes

- `pipeline_type='512'` is the lowest resolution / VRAM mode. Options: `'512'`, `'1024'`.
- `low_vram=True` is enabled by default in the pipeline when VRAM < 16 GB — it offloads model stages to CPU between steps.
- Peak VRAM during inference at 512³: ~11–12 GB.
- Peak system RAM during inference: ~18–22 GB (weights + activations).

### Custom Image

```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2, imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("pretrained/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open("your_image.png")   # RGBA or RGB, any size
mesh = pipeline.run(image, pipeline_type='512')[0]
mesh.simplify(16777216)

video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("output.mp4", video, fps=15)

glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
    coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], decimation_target=1000000,
    texture_size=1024, remesh=True, remesh_band=1, remesh_project=0, verbose=True
)
glb.export("output.glb", extension_webp=True)
```

---

## 7. Code Fix Applied (transformers 5.5.0 compatibility)

`trellis2/modules/image_feature_extractor.py` line 86 was patched to fix an `AttributeError` with transformers 5.5.0:

```python
# Before (broken with transformers 5.5.0):
for i, layer_module in enumerate(self.model.layer):

# After:
for i, layer_module in enumerate(self.model.model.layer):
```

In transformers 5.5.0, the `DINOv3ViTModel` encoder layers are nested one level deeper under `model.model` instead of `model`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `cannot find -lcuda` | `export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH` before `pip install` |
| Build process killed (exit 137) | `export MAX_JOBS=1` — reduces parallel compilation to avoid RAM OOM |
| `torch.float8_e8m0fnu` attribute error | Pin `transformers==5.5.0` |
| `AttributeError: 'DINOv3ViTModel' has no attribute 'layer'` | Already fixed in this repo; ensure `transformers==5.5.0` |
| WSL OOM during inference | Increase WSL memory in `C:\Users\<you>\.wslconfig`; needs 24 GB+ |
| 401/403 on gated model download | Accept license on HuggingFace page and set `HF_TOKEN` |
