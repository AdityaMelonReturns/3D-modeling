#!/usr/bin/env bash
# Orchestrator run LOCALLY. Ships this pipeline's code + one defect's
# images/metadata up to the RunPod GPU pod, runs it there, pulls back
# output/<defect_id>/. Same shape as Geometry_third's equivalent script.
#
# Usage:
#   ./run_defect_on_runpod.sh <defect_id> <ssh_host> <ssh_port> <ssh_user> <key_path> [defect_type]
#
# Example:
#   ./run_defect_on_runpod.sh hole 198.13.252.100 44017 root ~/.ssh/id_ed25519 hole_or_rot

set -euo pipefail

DEFECT_ID="${1:?Usage: run_defect_on_runpod.sh <defect_id> <ssh_host> <ssh_port> <ssh_user> <key_path> [defect_type]}"
HOST="${2:?missing ssh_host}"
PORT="${3:?missing ssh_port}"
USERNAME="${4:?missing ssh_user}"
KEY="${5:?missing key_path}"
DEFECT_TYPE="${6:-hole_or_rot}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOMETRY_BASE="$(cd "$SCRIPT_DIR/../Geometry/workspace" && pwd)"
DEFECT_DIR="$GEOMETRY_BASE/$DEFECT_ID"
REMOTE_DIR="/workspace/geofinal_$DEFECT_ID"

SSH_OPTS=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new)

if [ ! -d "$DEFECT_DIR/images" ]; then
  echo "$DEFECT_DIR/images not found - run Geometry/scripts/run_sparse_sfm.sh $DEFECT_ID first" \
       "(this pipeline reuses its already-extracted images/, metadata/ is optional here)" >&2
  exit 1
fi

echo "== One-time GPU env setup on pod (skips already-installed pieces) =="
scp "${SCP_OPTS[@]}" "$SCRIPT_DIR/remote_setup_final_gpu.sh" "$USERNAME@$HOST:/tmp/remote_setup_final_gpu.sh"
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" "bash /tmp/remote_setup_final_gpu.sh"

echo "== Packing pipeline code + this defect's images/metadata + prompts =="
TMP_TAR=/tmp/geofinal_"$DEFECT_ID".tar.gz
STAGE=/tmp/geofinal_stage_"$DEFECT_ID"
rm -rf "$STAGE" && mkdir -p "$STAGE/geofinal_code/prompts" "$STAGE/defect_data"
cp -r "$SCRIPT_DIR/pipeline" "$SCRIPT_DIR/run_defect.py" "$STAGE/geofinal_code/"
[ -f "$SCRIPT_DIR/prompts/$DEFECT_ID.json" ] && cp "$SCRIPT_DIR/prompts/$DEFECT_ID.json" "$STAGE/geofinal_code/prompts/" || true
cp -r "$DEFECT_DIR/images" "$STAGE/defect_data/"
[ -d "$DEFECT_DIR/metadata" ] && cp -r "$DEFECT_DIR/metadata" "$STAGE/defect_data/" || true
tar --owner=0 --group=0 --numeric-owner -C "$STAGE" -czf "$TMP_TAR" .
rm -rf "$STAGE"

ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" "mkdir -p $REMOTE_DIR"
scp "${SCP_OPTS[@]}" "$TMP_TAR" "$USERNAME@$HOST:$REMOTE_DIR/bundle.tar.gz"
rm -f "$TMP_TAR"

echo "== Running pipeline on the GPU pod =="
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" bash -s <<EOF
set -euo pipefail
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate geo2
export HF_HOME=/workspace/geo3_hf_cache
cd $REMOTE_DIR
tar --no-same-owner -xzf bundle.tar.gz
mkdir -p Geometry/workspace/$DEFECT_ID
cp -r defect_data/images Geometry/workspace/$DEFECT_ID/
[ -d defect_data/metadata ] && cp -r defect_data/metadata Geometry/workspace/$DEFECT_ID/ || true
ln -sf /workspace/geo2_checkpoints geofinal_code/checkpoints
cd geofinal_code
python -u run_defect.py $DEFECT_ID --defect-type $DEFECT_TYPE --device cuda \
  --geometry-base $REMOTE_DIR/Geometry/workspace
EOF

echo "== Downloading results =="
mkdir -p "$SCRIPT_DIR/output/$DEFECT_ID"
scp "${SCP_OPTS[@]}" -r "$USERNAME@$HOST:$REMOTE_DIR/geofinal_code/output/$DEFECT_ID/." "$SCRIPT_DIR/output/$DEFECT_ID/"

echo "Done. Output: $SCRIPT_DIR/output/$DEFECT_ID/"
