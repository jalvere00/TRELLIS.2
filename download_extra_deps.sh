#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

python - <<'EOF'
from huggingface_hub import snapshot_download
import os

token = os.environ.get("HF_TOKEN")
print(f"Token present: {'yes' if token else 'NO - not set!'}")

print("=== Downloading briaai/RMBG-2.0 (background removal) ===")
path = snapshot_download(
    repo_id="briaai/RMBG-2.0",
    token=token,
)
print(f"Cached at: {path}")
print("Done.")
EOF
