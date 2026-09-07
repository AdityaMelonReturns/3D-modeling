#!/usr/bin/env bash
# Sparse SfM pipeline (CPU-only, no GPU needed) for one defect capture zip.
#
# Usage:
#   ./run_sparse_sfm.sh <path-to-zip> [name]
#
# Example:
#   ./run_sparse_sfm.sh ../../gaps.zip gaps
#
# Produces, under Geometry/workspace/<name>/:
#   images/         - extracted photos
#   metadata/       - extracted *.json sensor sidecars (angles/gps/barometer/etc.)
#   database.db     - COLMAP feature/match database
#   sparse/0/       - binary reconstruction (cameras, poses, points3D) + points.ply
#   sparse/0_txt/   - human-readable TXT export of the same reconstruction
#   dense/          - undistorted images + MVS workspace, staged and ready for the
#                     dense step (patch_match_stereo + stereo_fusion), which requires
#                     a CUDA GPU with compute capability supported by the colmap-cuda
#                     build (see tools/colmap-cuda). Not run automatically here.

set -euo pipefail

ZIP_PATH="${1:?Usage: run_sparse_sfm.sh <path-to-zip> [name]}"
NAME="${2:-$(basename "$ZIP_PATH" .zip)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOMETRY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COLMAP="$GEOMETRY_DIR/tools/colmap-nocuda/bin/colmap.exe"
WORK="$GEOMETRY_DIR/workspace/$NAME"

if [ ! -x "$COLMAP" ]; then
  echo "COLMAP not found at $COLMAP" >&2
  exit 1
fi

echo "== Setting up workspace: $WORK =="
mkdir -p "$WORK/images" "$WORK/metadata"
unzip -o "$ZIP_PATH" -d "$WORK/images" > /dev/null
mv "$WORK"/images/*.json "$WORK/metadata/" 2>/dev/null || true

cd "$WORK"

echo "== Feature extraction (CPU) =="
"$COLMAP" feature_extractor \
  --image_path images \
  --database_path database.db \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --FeatureExtraction.use_gpu 0

echo "== Exhaustive matching (CPU) =="
"$COLMAP" exhaustive_matcher \
  --database_path database.db \
  --FeatureMatching.use_gpu 0

echo "== Incremental mapping (sparse reconstruction) =="
mkdir -p sparse
"$COLMAP" mapper \
  --image_path images \
  --database_path database.db \
  --output_path sparse

if [ -d sparse/0 ]; then
  "$COLMAP" model_analyzer --path sparse/0
  "$COLMAP" model_converter --input_path sparse/0 --output_path sparse/0/points.ply --output_type PLY
  mkdir -p sparse/0_txt
  "$COLMAP" model_converter --input_path sparse/0 --output_path sparse/0_txt --output_type TXT

  echo "== Staging dense workspace (undistortion only, CPU) =="
  mkdir -p dense
  "$COLMAP" image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path dense \
    --output_type COLMAP

  echo "Done. Sparse point cloud: $WORK/sparse/0/points.ply"
  echo "Dense MVS workspace staged at $WORK/dense (run patch_match_stereo + stereo_fusion on a CUDA GPU to finish)."
else
  echo "Mapper produced no reconstruction (sparse/0 missing) - check images for enough overlap." >&2
  exit 1
fi
