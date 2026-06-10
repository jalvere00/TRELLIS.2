#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 image-to-3D inference. "
                    "Accepts either an image file or a text prompt (which generates an image first).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "image",
        nargs="?",
        help="Path to input image (JPEG, PNG, WEBP, etc.).",
    )
    input_group.add_argument(
        "--prompt",
        help="Text description of the object to generate (runs text-to-image first, then 3D).",
    )

    # --- Output ---
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file prefix (e.g. 'my_model' → my_model.mp4 + my_model.glb). "
             "Defaults to input filename stem or a slug of the prompt.",
    )

    # --- 3D pipeline ---
    parser.add_argument(
        "-p", "--pipeline",
        default="1024_cascade",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        help="Resolution mode. '512' is fastest/lowest VRAM; "
             "'1024_cascade' is the recommended default; "
             "'1536_cascade' requires the most memory.",
    )
    parser.add_argument(
        "--weights",
        default="pretrained/TRELLIS.2-4B",
        help="Path to pretrained TRELLIS.2 weights directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (applied to both image generation and 3D inference).",
    )

    # --- Text-to-image options (only used with --prompt) ---
    t2i = parser.add_argument_group("text-to-image options (--prompt only)")
    t2i.add_argument(
        "--t2i-model",
        default="black-forest-labs/FLUX.1-schnell",
        help="HuggingFace model ID for text-to-image generation.",
    )
    t2i.add_argument(
        "--t2i-steps",
        type=int,
        default=4,
        help="Number of diffusion steps for image generation.",
    )
    t2i.add_argument(
        "--t2i-guidance",
        type=float,
        default=0.0,
        help="Guidance scale (0.0 = distilled/schnell mode; use 3.5 for FLUX.1-dev).",
    )
    t2i.add_argument(
        "--t2i-width",
        type=int,
        default=1024,
        help="Generated image width in pixels.",
    )
    t2i.add_argument(
        "--t2i-height",
        type=int,
        default=1024,
        help="Generated image height in pixels.",
    )
    t2i.add_argument(
        "--save-image",
        default=None,
        help="Save the generated image to this path (e.g. 'generated.png'). "
             "Useful for inspecting what was sent to TRELLIS.2.",
    )

    # --- GLB / video export ---
    parser.add_argument(
        "--envmap",
        default="assets/hdri/forest.exr",
        help="Path to HDR environment map (.exr) for PBR video render.",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=2048,
        choices=[512, 1024, 2048, 4096],
        help="GLB texture atlas resolution in pixels.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=1_000_000,
        help="Target face count after mesh decimation for GLB export.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for the output MP4.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rendering the turntable MP4.",
    )
    parser.add_argument(
        "--no-glb",
        action="store_true",
        help="Skip exporting the GLB file.",
    )

    args = parser.parse_args()

    # positional 'image' inside a mutually-exclusive group needs a manual check
    if args.image is None and args.prompt is None:
        parser.error("Provide either an image path or --prompt.")

    return args


def slugify(text: str, max_len: int = 40) -> str:
    import re
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_len]


def generate_image(args):
    """
    Run FLUX text-to-image in a subprocess so its CUDA context is fully
    isolated from TRELLIS.2's sparse-conv / flash-attn initialisation.
    Returns a PIL Image loaded from the saved temp file.
    """
    import subprocess
    import tempfile
    from PIL import Image

    # Write to a temp file; caller may copy it to args.save_image afterward
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    tmp_path = tmp.name

    helper = Path(__file__).parent / "_flux_generate.py"
    cmd = [
        sys.executable, str(helper),
        tmp_path,
        args.t2i_model,
        args.prompt,
        str(args.t2i_steps),
        str(args.t2i_guidance),
        str(args.t2i_width),
        str(args.t2i_height),
        str(args.seed),
    ]

    print(f"Generating image via {args.t2i_model} (subprocess)...")
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        print("ERROR: image generation subprocess failed.", file=sys.stderr)
        sys.exit(1)

    image = Image.open(tmp_path).copy()
    os.unlink(tmp_path)
    print("  Image generation done.")
    return image


def main():
    args = parse_args()

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from PIL import Image

    device = "cuda"
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Resolve input image ---
    # IMPORTANT: trellis2 / o_voxel imports are deferred until AFTER image
    # generation so their CUDA sparse-conv initialisation doesn't corrupt the
    # CUDA context that FLUX's accelerate hooks expect.
    if args.prompt:
        output_stem = args.output if args.output else slugify(args.prompt)
        image = generate_image(args)
        if args.save_image:
            image.save(args.save_image)
            print(f"  Generated image saved to {args.save_image}")
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: image not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        output_stem = args.output if args.output else image_path.stem
        image = Image.open(image_path)

    # Trellis2 imports happen here — after FLUX is done and VRAM is freed
    import cv2
    import imageio
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    import o_voxel

    output_mp4 = Path(f"{output_stem}.mp4")
    output_glb = Path(f"{output_stem}.glb")

    print(f"\nPipeline: {args.pipeline}")
    print(f"Output:   {output_stem}.{{mp4,glb}}")
    print(f"Seed:     {args.seed}")
    print(f"Image:    {image.size[0]}x{image.size[1]} {image.mode}\n")

    # --- Load envmap before TRELLIS (small, stays on GPU throughout) ---
    if not args.no_video:
        envmap_path = Path(args.envmap)
        if not envmap_path.exists():
            print(f"ERROR: envmap not found: {envmap_path}", file=sys.stderr)
            sys.exit(1)
        raw = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device=device,
        ))

    # --- TRELLIS.2 ---
    print("Loading TRELLIS.2...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.cuda()

    print(f"Running 3D inference ({args.pipeline})...")
    mesh = pipeline.run(image, pipeline_type=args.pipeline, seed=args.seed)[0]
    mesh.simplify(16_777_216)
    print(f"  Done — VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if not args.no_video:
        print("Rendering video...")
        frames = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap)
        )
        imageio.mimsave(str(output_mp4), frames, fps=args.fps)
        print(f"  Saved {output_mp4}")

    if not args.no_glb:
        print("Exporting GLB...")
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation,
            texture_size=args.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(str(output_glb), extension_webp=True)
        print(f"  Saved {output_glb}")

    print("\nDone!")


if __name__ == "__main__":
    main()
