#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
cd /mnt/c/Users/jamee/Documents/GitHub/TRELLIS.2

python - <<'EOF'
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

print(f"VRAM before load: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline from local weights
print("Loading pipeline from local weights...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("pretrained/TRELLIS.2-4B")
pipeline.cuda()
print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 3. Run at 1024^3 via cascade (starts at 512, refines to 1024)
print("Running inference at 1024_cascade...")
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image, pipeline_type='1024_cascade')[0]
mesh.simplify(16777216)
print(f"Inference done! VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 4. Render Video
print("Rendering video...")
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)
print("Saved sample.mp4")

# 5. Export to GLB
print("Exporting GLB...")
glb = o_voxel.postprocess.to_glb(
    vertices          = mesh.vertices,
    faces             = mesh.faces,
    attr_volume       = mesh.attrs,
    coords            = mesh.coords,
    attr_layout       = mesh.layout,
    voxel_size        = mesh.voxel_size,
    aabb              = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target = 1000000,
    texture_size      = 2048,
    remesh            = True,
    remesh_band       = 1,
    remesh_project    = 0,
    verbose           = True
)
glb.export("sample.glb", extension_webp=True)
print("Saved sample.glb")
print("All done!")
EOF
