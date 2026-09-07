#!/usr/bin/env bash
# Orchestrator run LOCALLY (Windows/git-bash). Ships a defect's staged "dense/"
# workspace (from run_sparse_sfm.sh) up to a RunPod GPU pod over SSH, runs the
# CUDA dense MVS steps there, and pulls fused.ply + depth maps back down.
#
# Usage:
#   ./run_dense_on_runpod.sh <name> <ssh_host> <ssh_port> <ssh_user> <key_path>
#
# Example:
#   ./run_dense_on_runpod.sh roof 123.45.67.89 40001 root ~/.ssh/runpod_key
#
# Prereqs on the pod (one-time): scp remote_setup_colmap_gpu.sh up and run it,
# or let this script do it automatically (default behavior below).

set -euo pipefail

NAME="${1:?Usage: run_dense_on_runpod.sh <name> <ssh_host> <ssh_port> <ssh_user> <key_path>}"
HOST="${2:?missing ssh_host}"
PORT="${3:?missing ssh_port}"
USERNAME="${4:?missing ssh_user}"
KEY="${5:?missing key_path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOMETRY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$GEOMETRY_DIR/workspace/$NAME"
REMOTE_DIR="/workspace/geo_$NAME"

SSH_OPTS=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new)

if [ ! -d "$WORK/dense" ]; then
  echo "$WORK/dense not found - run run_sparse_sfm.sh $NAME first" >&2
  exit 1
fi

echo "== One-time COLMAP+CUDA setup on pod (skips if already installed) =="
scp "${SCP_OPTS[@]}" "$SCRIPT_DIR/remote_setup_colmap_gpu.sh" "$USERNAME@$HOST:/tmp/remote_setup_colmap_gpu.sh"
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" "bash /tmp/remote_setup_colmap_gpu.sh"

echo "== Packing and uploading dense workspace =="
tar --owner=0 --group=0 --numeric-owner -C "$WORK" -czf /tmp/"$NAME"_dense.tar.gz dense
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" "mkdir -p $REMOTE_DIR"
scp "${SCP_OPTS[@]}" /tmp/"$NAME"_dense.tar.gz "$USERNAME@$HOST:$REMOTE_DIR/dense.tar.gz"
rm /tmp/"$NAME"_dense.tar.gz

echo "== Running dense MVS on the GPU pod =="
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" bash -s <<EOF
set -euo pipefail
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate colmap
cd $REMOTE_DIR
tar --no-same-owner -xzf dense.tar.gz
colmap patch_match_stereo --workspace_path dense
colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply
EOF

echo "== Downloading result =="
scp "${SCP_OPTS[@]}" "$USERNAME@$HOST:$REMOTE_DIR/dense/fused.ply" "$WORK/dense/fused.ply"

echo "Done. Dense point cloud: $WORK/dense/fused.ply"
