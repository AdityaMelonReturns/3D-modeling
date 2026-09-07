#!/usr/bin/env bash
# Run this ON the RunPod GPU pod. Sets up a conda env with CUDA PyTorch +
# LightGlue + SAM2 + Depth Pro, and downloads their checkpoints. Idempotent.
#
# Assumes /workspace/miniconda3 already exists from the COLMAP-pipeline setup
# (Geometry/scripts/remote_setup_colmap_gpu.sh) - reuses the same miniconda
# install, just adds a new env, so it doesn't duplicate the ~500MB base.
#
# Usage (on the pod):
#   bash remote_setup_pipeline2_gpu.sh

set -euo pipefail

echo "== GPU check =="
nvidia-smi || { echo "nvidia-smi failed - no GPU visible in this pod" >&2; exit 1; }

CONDA_DIR="/workspace/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
  echo "== Installing Miniconda (not found from prior setup) =="
  curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^geo2 "; then
  echo "== Creating geo2 conda env (Python 3.11) =="
  conda create -y -n geo2 --override-channels -c conda-forge python=3.11
fi
conda activate geo2

echo "== Installing base deps (skips if already present) =="
python -c "import cv2, open3d, trimesh, huggingface_hub, anthropic, scipy" 2>/dev/null \
  || pip install --quiet opencv-python open3d trimesh numpy pillow huggingface_hub anthropic scipy

echo "== Checking CUDA PyTorch =="
# This GPU class (Blackwell, sm_120+) needs a cu130+ build - cu126 has no
# kernels for it and silently reports cuda.is_available()==True while every
# real kernel launch fails ("no kernel image is available for execution on
# the device"). Only (re)install if torch is missing OR a real GPU matmul
# actually fails - never blindly reinstall (each attempt is a multi-GB
# download against this pod's storage quota, which we've already hit once).
if ! python -c "
import torch
assert torch.cuda.is_available()
x = torch.randn(64, 64, device='cuda')
(x @ x).sum().item()
" 2>/dev/null; then
  echo "   -> (re)installing torch cu130 build"
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
else
  echo "   -> torch already working on this GPU, skipping"
fi

echo "== Installing LightGlue + SuperPoint =="
python -c "import lightglue" 2>/dev/null || pip install --quiet "git+https://github.com/cvg/LightGlue.git"

echo "== Installing SAM2 =="
python -c "import sam2" 2>/dev/null || pip install --quiet "git+https://github.com/facebookresearch/sam2.git"

echo "== Installing Depth Pro =="
python -c "import depth_pro" 2>/dev/null || pip install --quiet "git+https://github.com/apple/ml-depth-pro.git"
# depth-pro's setup.py pins numpy<2 but runs fine on numpy 2.x (verified
# locally) - opencv-python needs numpy>=2, so re-assert the newer one wins.
python -c "import numpy; assert numpy.__version__.startswith('2')" 2>/dev/null || pip install --quiet "numpy>=2"

# Keep the quota-limited /workspace volume lean - pip's wheel cache
# accumulates multi-GB per torch reinstall attempt and isn't needed after.
pip cache purge >/dev/null 2>&1 || true

mkdir -p /workspace/geo2_checkpoints/sam2
if [ ! -f /workspace/geo2_checkpoints/depth_pro.pt ]; then
  echo "== Downloading Depth Pro checkpoint (~1.9GB) =="
  python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='apple/DepthPro', filename='depth_pro.pt', local_dir='/workspace/geo2_checkpoints')
"
fi
if [ ! -f /workspace/geo2_checkpoints/sam2/sam2.1_hiera_large.pt ]; then
  echo "== Downloading SAM2 checkpoint (~898MB) =="
  curl -sL -o /workspace/geo2_checkpoints/sam2/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

echo "== Verifying CUDA is visible to torch =="
python -c "import torch; print('cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "Setup complete. Activate with: conda activate geo2"
