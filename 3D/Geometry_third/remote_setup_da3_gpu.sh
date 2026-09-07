#!/usr/bin/env bash
# Run this ON the RunPod GPU pod. Adds DA3 + xformers to the EXISTING `geo2`
# conda env from Geometry_Second (already has torch cu130 verified working on
# this Blackwell GPU, plus SAM2, Depth Pro, opencv, open3d, trimesh, etc).
#
# Reusing geo2 instead of building a fresh env is a deliberate time-budget
# call: we only have a 1-hour pod window today, and a from-scratch env
# (conda create + torch + opencv/open3d/trimesh + SAM2 + Depth Pro) costs
# several extra minutes we don't have to spend twice. The trade-off is a
# real risk - xformers or DA3's own install could silently upgrade/downgrade
# torch and re-break the Blackwell (sm_120) fix we already fought for. This
# script guards against exactly that: xformers and DA3 are installed with
# --no-deps (so pip can't pull in a conflicting torch on its own), and torch's
# version + a real CUDA op are checked BEFORE and AFTER to prove nothing moved.
#
# Assumes /workspace/miniconda3 and the `geo2` env already exist (from the
# Geometry_Second setup). If geo2 doesn't exist for some reason, this falls
# back to building it fresh (slower, but still correct).
#
# Usage (on the pod):
#   bash remote_setup_da3_gpu.sh

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

if conda env list | grep -q "^geo2 "; then
  echo "== Reusing existing geo2 env (has torch/SAM2/Depth Pro already) =="
  conda activate geo2
else
  echo "== geo2 not found - creating it fresh (slower path) =="
  conda create -y -n geo2 --override-channels -c conda-forge python=3.11
  conda activate geo2
  pip install --quiet opencv-python open3d trimesh numpy pillow huggingface_hub anthropic scipy
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu130
  pip install --quiet "git+https://github.com/facebookresearch/sam2.git"
  pip install --quiet "git+https://github.com/apple/ml-depth-pro.git"
fi

export HF_HOME=/workspace/geo3_hf_cache
mkdir -p "$HF_HOME"

echo "== Baseline: confirm torch + CUDA work BEFORE touching anything =="
BEFORE=$(python -c "
import torch
assert torch.cuda.is_available()
x = torch.randn(64, 64, device='cuda')
(x @ x).sum().item()
print(torch.__version__)
")
echo "   torch $BEFORE, CUDA op OK"

echo "== Installing xformers (--no-deps, so it can't silently change torch) =="
# Capped at 5 minutes: if pip can't find a prebuilt wheel for this torch/CUDA
# combo it may silently fall back to compiling xformers from source, which
# can run 30-60+ minutes - not acceptable against a 1-hour pod budget. Fail
# fast instead and fall through to the documented no-xformers path.
if ! python -c "import xformers" 2>/dev/null; then
  if ! timeout 300 pip install --quiet --no-deps xformers; then
    echo "   !! xformers install failed. DA3's own repo has a note for this exact"
    echo "   !! situation: 'Older GPUs without XFormers support: see Issue #11'"
    echo "   !! (https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11)."
    echo "   !! Proceeding without xformers - DA3 may still run with a slower"
    echo "   !! attention fallback, or may fail at inference time. Don't treat"
    echo "   !! this as fatal yet, but flag it clearly if inference errors out."
  fi
else
  echo "   -> xformers already installed, skipping"
fi

echo "== Installing Depth Anything 3 (--no-deps, then filling in small missing pieces only) =="
DA3_SRC=/workspace/geo3_da3_src
if [ ! -d "$DA3_SRC" ]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_SRC"
fi
python -c "import depth_anything_3" 2>/dev/null || pip install --quiet --no-deps -e "$DA3_SRC"

# --no-deps means small lightweight deps DA3 actually needs (einops etc) may
# be missing - these are pure-python, no risk to torch, safe to install
# normally. Try the real import; if it's missing a specific module, install
# just that one and retry, instead of guessing the whole list upfront.
for attempt in 1 2 3 4 5; do
  MISSING=$(python -c "from depth_anything_3.api import DepthAnything3" 2>&1 | grep "No module named" | sed -E "s/.*No module named '([^']+)'.*/\1/") || true
  if [ -z "$MISSING" ]; then
    break
  fi
  echo "   -> installing missing dep: $MISSING"
  pip install --quiet "$MISSING"
done

echo "== Verifying torch is UNCHANGED after xformers + DA3 install =="
AFTER=$(python -c "
import torch
assert torch.cuda.is_available()
x = torch.randn(64, 64, device='cuda')
(x @ x).sum().item()
print(torch.__version__)
")
echo "   torch $AFTER, CUDA op OK"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "   !! WARNING: torch version changed ($BEFORE -> $AFTER) despite --no-deps."
  echo "   !! CUDA op above still passed, so it may be fine, but flag this explicitly"
  echo "   !! rather than silently trusting it - this is exactly the failure mode"
  echo "   !! that broke things before (cu126 vs cu130 on this GPU)."
fi

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

echo "== Smoke-testing DA3 import (not a full inference test yet) =="
python -c "from depth_anything_3.api import DepthAnything3; print('DA3 import OK')"

echo "Setup complete. Activate with: conda activate geo2 && export HF_HOME=/workspace/geo3_hf_cache"
