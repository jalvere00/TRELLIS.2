#!/usr/bin/env python3
"""
Standalone FLUX image generation helper — called as a subprocess by infer.py
so its CUDA context is fully isolated from TRELLIS.2.

Usage: python _flux_generate.py <output_image> <model_id> <prompt>
       <steps> <guidance> <width> <height> <seed>
"""
import sys
import os

# Must be set before torch is imported
os.environ["TORCHDYNAMO_DISABLE"] = "1"

output_path = sys.argv[1]
model_id    = sys.argv[2]
prompt      = sys.argv[3]
steps       = int(sys.argv[4])
guidance    = float(sys.argv[5])
width       = int(sys.argv[6])
height      = int(sys.argv[7])
seed        = int(sys.argv[8])

import torch
from diffusers import FluxPipeline

token = os.environ.get("HF_TOKEN")

print(f"[flux] Loading {model_id}...")
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=token)

# enable_model_cpu_offload() moves the whole 12B-param transformer (~24GB bfloat16)
# to GPU at once — too large for 12GB VRAM, causes WSL2 "CUDA driver error: device
# not ready". enable_sequential_cpu_offload() hooks individual layers (~100-500MB
# each) so no single transfer exceeds available VRAM.
pipe.enable_sequential_cpu_offload()

full_prompt = (
    f"{prompt}, single object, clean white background, "
    "product photography, studio lighting, no shadow"
)
print(f"[flux] Prompt: {full_prompt}")

generator = torch.Generator("cpu").manual_seed(seed)
image = pipe(
    prompt=full_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance,
    width=width,
    height=height,
    generator=generator,
).images[0]

image.save(output_path)
print(f"[flux] Saved {output_path}")
