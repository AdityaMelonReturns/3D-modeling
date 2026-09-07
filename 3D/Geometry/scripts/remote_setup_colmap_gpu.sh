#!/usr/bin/env bash
# Run this ON the remote GPU pod (RunPod, etc). Installs a CUDA-enabled COLMAP
# via conda-forge, no build-from-source needed. Idempotent - safe to re-run.
#
# Usage (on the pod):
#   bash remote_setup_colmap_gpu.sh

set -euo pipefail

echo "== GPU check =="
nvidia-smi || { echo "nvidia-smi failed - no GPU visible in this pod" >&2; exit 1; }

CONDA_DIR="/workspace/miniconda3"

if ! command -v conda >/dev/null 2>&1 && [ ! -d "$CONDA_DIR" ]; then
  echo "== Installing Miniconda =="
  curl -sL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm /tmp/miniconda.sh
fi

# shellcheck disable=SC1090
source "$CONDA_DIR/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^colmap "; then
  echo "== Creating colmap conda env (CUDA build) =="
  conda create -y -n colmap --override-channels -c conda-forge "colmap=4.1.1=cuda*"
fi

conda activate colmap

# The conda-forge colmap=4.1.1 cuda build is missing a couple of runtime libs
# it links against (libfaiss, openimageio) - install them explicitly.
if ! colmap -h >/tmp/colmap_check.log 2>&1; then
  echo "== Patching missing runtime deps (libfaiss/openimageio) =="
  conda install -y -n colmap --override-channels -c conda-forge "libfaiss=1.10.0=cuda129*" openimageio
fi

echo "== Verifying COLMAP sees the GPU =="
colmap -h | head -3
echo "Setup complete. Activate with: conda activate colmap"
