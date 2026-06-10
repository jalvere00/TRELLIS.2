#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 image-to-3D inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        help="Path to input image (JPEG, PNG, WEBP, etc.)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file prefix (e.g. 'my_model' → my_model.mp4 + my_model.glb). "
             "Defaults to the input filename stem.",
    )
    parser.add_argument(
        "-p", "--pipeline",
        default="1024_cascade",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        help="Pipeline resolution mode. '512' is fastest/lowest VRAM; "
             "'1024_cascade' is the recommended default; "
             "'1536_cascade' requires most memory.",
    )
    parser.add_argument(
        "--weights",
        default="pretrained/TRELLIS.2-4B",
        help="Path to pretrained weights directory.",
    )
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
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible results.",
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
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import cv2
    import imageio
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import EnvMap
    from trellis2.utils import render_utils
    import o_voxel

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    output_stem = args.output if args.output else image_path.stem
    output_mp4 = Path(f"{output_stem}.mp4")
    output_glb = Path(f"{output_stem}.glb")

    print(f"Image:    {image_path}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Output:   {output_stem}.{{mp4,glb}}")
    print(f"Seed:     {args.seed}")
    print()

    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if not args.no_video:
        envmap_path = Path(args.envmap)
        if not envmap_path.exists():
            print(f"ERROR: envmap not found: {envmap_path}", file=sys.stderr)
            sys.exit(1)
        raw = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device="cuda",
        ))

    print("Loading pipeline...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.cuda()

    print("Loading image...")
    image = Image.open(image_path)
    print(f"  Size: {image.size}, mode: {image.mode}")

    print(f"Running inference ({args.pipeline})...")
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
