#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

echo "=== Downloading microsoft/TRELLIS.2-4B via huggingface_hub Python API ==="

python - <<'EOF'
from huggingface_hub import snapshot_download
import sys

try:
    path = snapshot_download(
        repo_id="microsoft/TRELLIS.2-4B",
        local_dir="/mnt/c/Users/jamee/Documents/GitHub/TRELLIS.2/pretrained/TRELLIS.2-4B",
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {path}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "=== Download complete ==="
du -sh /mnt/c/Users/jamee/Documents/GitHub/TRELLIS.2/pretrained/TRELLIS.2-4B 2>/dev/null
ls /mnt/c/Users/jamee/Documents/GitHub/TRELLIS.2/pretrained/TRELLIS.2-4B 2>/dev/null
