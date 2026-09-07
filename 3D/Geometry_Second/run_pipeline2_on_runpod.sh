#!/usr/bin/env bash
# Orchestrator run LOCALLY. Ships this pipeline's code + one defect's
# images/metadata up to the RunPod GPU pod, runs it there (real GPU, not the
# ~5min/image CPU crawl), and pulls back output/<defect_id>/.
#
# Usage:
#   ./run_pipeline2_on_runpod.sh <defect_id> <ssh_host> <ssh_port> <ssh_user> <key_path> [defect_type]
#
# Example:
#   ./run_pipeline2_on_runpod.sh ceiling 198.13.252.100 26859 root ~/.ssh/id_ed25519 hole_or_rot
#
# Notes learned from the COLMAP-pod scripts (Geometry/scripts/):
#   - Use the DIRECT ssh/scp endpoint (host:port), not the ssh.runpod.io
#     proxy - the proxy only supports interactive shells, scp/exec fail on it.
#   - tar must use --owner=0 --group=0 on creation and --no-same-owner on
#     extraction, or root's tar aborts trying to chown to a bogus Windows uid.
#   - scp uses -P (capital) for port; ssh uses -p (lowercase). Mixing these
#     up silently breaks scp (it interprets -p as "preserve mtimes").

set -euo pipefail

DEFECT_ID="${1:?Usage: run_pipeline2_on_runpod.sh <defect_id> <ssh_host> <ssh_port> <ssh_user> <key_path> [defect_type]}"
HOST="${2:?missing ssh_host}"
PORT="${3:?missing ssh_port}"
USERNAME="${4:?missing ssh_user}"
KEY="${5:?missing key_path}"
DEFECT_TYPE="${6:-hole_or_rot}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOMETRY_BASE="$(cd "$SCRIPT_DIR/../Geometry/workspace" && pwd)"
DEFECT_DIR="$GEOMETRY_BASE/$DEFECT_ID"
REMOTE_DIR="/workspace/geo2_$DEFECT_ID"

SSH_OPTS=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new)

if [ ! -d "$DEFECT_DIR/images" ]; then
  echo "$DEFECT_DIR/images not found - run Geometry/scripts/run_sparse_sfm.sh $DEFECT_ID first" \
       "(this pipeline reuses its already-extracted images/+metadata/, no need to re-unzip)" >&2
  exit 1
fi
PROMPTS_FILE="$SCRIPT_DIR/prompts/$DEFECT_ID.json"
if [ ! -f "$PROMPTS_FILE" ]; then
  echo "$PROMPTS_FILE not found - add box prompts (see pipeline/segment.py:load_prompts) first" >&2
  exit 1
fi

echo "== One-time GPU env setup on pod (skips already-installed pieces) =="
scp "${SCP_OPTS[@]}" "$SCRIPT_DIR/remote_setup_pipeline2_gpu.sh" "$USERNAME@$HOST:/tmp/remote_setup_pipeline2_gpu.sh"
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" "bash /tmp/remote_setup_pipeline2_gpu.sh"

echo "== Packing pipeline code + this defect's images/metadata + prompts =="
TMP_TAR=/tmp/geo2_"$DEFECT_ID".tar.gz
STAGE=/tmp/geo2_stage_"$DEFECT_ID"
# NOTE: prompts/ must live INSIDE geo2_code/ - run_defect.py resolves its
# prompts path relative to its own file location (os.path.dirname(__file__)),
# not relative to the bundle root. A top-level prompts/ next to geo2_code/
# silently produces "no box prompts found" for every shot (hit this on the
# first real run - ceiling/gaps/hole all failed this way before the fix).
rm -rf "$STAGE" && mkdir -p "$STAGE/geo2_code/prompts" "$STAGE/defect_data"
cp -r "$SCRIPT_DIR/pipeline" "$SCRIPT_DIR/run_defect.py" "$STAGE/geo2_code/"
cp "$PROMPTS_FILE" "$STAGE/geo2_code/prompts/"
cp -r "$DEFECT_DIR/images" "$DEFECT_DIR/metadata" "$STAGE/defect_data/"
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
cd $REMOTE_DIR
tar --no-same-owner -xzf bundle.tar.gz
mkdir -p Geometry/workspace/$DEFECT_ID
cp -r defect_data/images defect_data/metadata Geometry/workspace/$DEFECT_ID/
ln -sf /workspace/geo2_checkpoints geo2_code/checkpoints
cd geo2_code
python -u run_defect.py $DEFECT_ID --defect-type $DEFECT_TYPE --device cuda \
  --geometry-base $REMOTE_DIR/Geometry/workspace
EOF

echo "== Downloading results =="
mkdir -p "$SCRIPT_DIR/output/$DEFECT_ID"
scp "${SCP_OPTS[@]}" -r "$USERNAME@$HOST:$REMOTE_DIR/geo2_code/output/$DEFECT_ID/." "$SCRIPT_DIR/output/$DEFECT_ID/"

echo "Done. Output: $SCRIPT_DIR/output/$DEFECT_ID/"
