#!/usr/bin/env bash
# Run this ON the RunPod GPU pod. Adds Qwen3-VL to the EXISTING `geo2` conda
# env (already has torch cu130 verified working on this Blackwell GPU, plus
# SAM2 and DA3 from Geometry_third's setup). Same time-budget reasoning as
# Geometry_third's setup script: reusing geo2 instead of a fresh env saves
# re-downloading torch/opencv/open3d/SAM2/DA3, all of which already work.
#
# The one new, real risk: Qwen3-VL needs transformers built from source (not
# yet in a stable pip release as of this writing) - see DESIGN.md's risk
# list. That's a bigger, riskier dependency change than xformers was for DA3,
# because SAM2/DA3 may themselves import transformers internally. This
# script verifies torch AND a plain import of sam2/depth_anything_3 both
# before and after, not just torch, so any breakage shows up immediately
# here rather than mid-run.
#
# Usage (on the pod):
#   bash remote_setup_final_gpu.sh

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
  echo "== Reusing existing geo2 env (torch/SAM2/DA3 already there) =="
  conda activate geo2
else
  echo "== geo2 not found - this script assumes Geometry_third's setup already ran on this"
  echo "   pod/volume. If this is a genuinely fresh volume, run"
  echo "   Geometry_third/remote_setup_da3_gpu.sh first, then re-run this script."
  exit 1
fi

# Same persistent cache Geometry_third used - DA3-BASE's checkpoint is
# already sitting here from prior runs, no need to re-download it.
export HF_HOME=/workspace/geo3_hf_cache
mkdir -p "$HF_HOME"

echo "== Baseline: confirm torch + SAM2 + DA3 all work BEFORE touching anything =="
python -c "
import torch
assert torch.cuda.is_available()
(torch.randn(64,64,device='cuda') @ torch.randn(64,64,device='cuda')).sum().item()
print('torch OK', torch.__version__)
import sam2
print('sam2 import OK')
import depth_anything_3
print('depth_anything_3 import OK')
"

echo "== Installing transformers from source (needed for Qwen3-VL - not yet in a stable release) =="
if ! python -c "from transformers import Qwen3VLMoeForConditionalGeneration" 2>/dev/null; then
  pip install --quiet git+https://github.com/huggingface/transformers
  pip install --quiet accelerate
else
  echo "   -> Qwen3-VL support already present, skipping"
fi

echo "== Verifying torch + SAM2 + DA3 are all STILL working after the transformers upgrade =="
python -c "
import torch
assert torch.cuda.is_available()
(torch.randn(64,64,device='cuda') @ torch.randn(64,64,device='cuda')).sum().item()
print('torch OK', torch.__version__)
import sam2
print('sam2 import OK')
import depth_anything_3
print('depth_anything_3 import OK')
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
print('Qwen3-VL classes import OK')
"

pip cache purge >/dev/null 2>&1 || true

echo "== Pre-downloading Qwen3-VL-4B-Instruct (~8GB bf16) =="
# NOT the 30B-A3B MoE variant DESIGN.md originally planned - this pod's
# /workspace has a real ~40GB storage quota (separate from the "195TB free"
# df shows, which is the whole shared cluster). Confirmed live: two 30B-A3B
# download attempts both failed with "Disk quota exceeded (os error 122)"
# partway through, even after freeing ~28GB of old envs/leftover folders.
# See DESIGN.md's updated risk list.
python -c "
from transformers import AutoModelForImageTextToText, AutoProcessor
ckpt = 'Qwen/Qwen3-VL-4B-Instruct'
print('downloading/loading processor...')
AutoProcessor.from_pretrained(ckpt)
print('downloading/loading model...')
model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype='auto', device_map='auto')
print('Qwen3-VL loaded OK on', model.device)
"

# geo2_checkpoints already has SAM2 + Depth Pro's checkpoint from earlier
# folders (Depth Pro's isn't used here, but SAM2's is - no need to touch it).
mkdir -p /workspace/geo2_checkpoints/sam2
if [ ! -f /workspace/geo2_checkpoints/sam2/sam2.1_hiera_large.pt ]; then
  echo "== Downloading SAM2 checkpoint (~898MB) =="
  curl -sL -o /workspace/geo2_checkpoints/sam2/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

echo "Setup complete. Activate with: conda activate geo2 && export HF_HOME=/workspace/geo3_hf_cache"
