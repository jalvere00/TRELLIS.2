#!/bin/bash
# TRELLIS.2 inference launcher
# Usage: bash infer.sh <image> [options]
# Examples:
#   bash infer.sh assets/example_image/T.png
#   bash infer.sh my_photo.jpg -o my_model --pipeline 1024_cascade
#   bash infer.sh my_photo.jpg --pipeline 512 --no-video
#   bash infer.sh my_photo.jpg --texture-size 4096 --decimation 2000000 --seed 42

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
cd /mnt/c/Users/jamee/Documents/GitHub/TRELLIS.2

exec python infer.py "$@"
