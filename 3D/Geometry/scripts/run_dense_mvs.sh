#!/usr/bin/env bash
# Dense MVS (patch_match_stereo + stereo_fusion). REQUIRES a CUDA-capable NVIDIA GPU
# that the colmap-cuda build actually supports (see tools/colmap-cuda). On this
# machine the only GPU (MX130, compute capability 5.0) is too old for the prebuilt
# CUDA binaries ("no kernel image is available for execution on the device") and
# COLMAP has no CPU fallback for dense stereo - it hard-errors:
#   "Dense stereo reconstruction requires CUDA, which is not available on your system."
#
# Run this only on a machine/instance with a modern NVIDIA GPU (Pascal/compute
# capability 6.1+ recommended) and enough VRAM for the image resolution.
#
# Usage:
#   ./run_dense_mvs.sh <name>
# Expects workspace/<name>/dense to already exist (produced by run_sparse_sfm.sh).

set -euo pipefail

NAME="${1:?Usage: run_dense_mvs.sh <name>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOMETRY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COLMAP="$GEOMETRY_DIR/tools/colmap-cuda/bin/colmap.exe"
WORK="$GEOMETRY_DIR/workspace/$NAME"

if [ ! -x "$COLMAP" ]; then
  echo "colmap-cuda not found at $COLMAP" >&2
  exit 1
fi
if [ ! -d "$WORK/dense" ]; then
  echo "$WORK/dense not found - run run_sparse_sfm.sh $NAME first" >&2
  exit 1
fi

cd "$WORK"

echo "== Patch match stereo (GPU) =="
"$COLMAP" patch_match_stereo --workspace_path dense

echo "== Stereo fusion =="
"$COLMAP" stereo_fusion \
  --workspace_path dense \
  --output_path dense/fused.ply

echo "Done. Dense point cloud: $WORK/dense/fused.ply"
