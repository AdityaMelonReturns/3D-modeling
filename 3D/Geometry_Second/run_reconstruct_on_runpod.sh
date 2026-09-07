#!/usr/bin/env bash
# Orchestrator run LOCALLY - ships reconstruct_object.py + a plain image
# folder (no sensor metadata needed, unlike run_pipeline2_on_runpod.sh) to
# the GPU pod, runs it, pulls back output/<name>/.
#
# Usage:
#   ./run_reconstruct_on_runpod.sh <name> <image_dir> <prompts.json> <ssh_host> <ssh_port> <ssh_user> <key_path>
#
# Example:
#   ./run_reconstruct_on_runpod.sh bottle ../Bottle prompts/bottle.json 198.13.252.100 44017 root ~/.ssh/id_ed25519

set -euo pipefail

NAME="${1:?Usage: run_reconstruct_on_runpod.sh <name> <image_dir> <prompts.json> <ssh_host> <ssh_port> <ssh_user> <key_path>}"
IMAGE_DIR="${2:?missing image_dir}"
PROMPTS_FILE="${3:?missing prompts.json}"
HOST="${4:?missing ssh_host}"
PORT="${5:?missing ssh_port}"
USERNAME="${6:?missing ssh_user}"
KEY="${7:?missing key_path}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="/workspace/geo2_$NAME"

SSH_OPTS=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new)

[ -d "$IMAGE_DIR" ] || { echo "$IMAGE_DIR not found" >&2; exit 1; }
[ -f "$PROMPTS_FILE" ] || { echo "$PROMPTS_FILE not found" >&2; exit 1; }

echo "== One-time GPU env setup on pod (skips already-installed pieces) =="
scp "${SCP_OPTS[@]}" "$SCRIPT_DIR/remote_setup_pipeline2_gpu.sh" "$USERNAME@$HOST:/tmp/remote_setup_pipeline2_gpu.sh"
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" "bash /tmp/remote_setup_pipeline2_gpu.sh"

echo "== Packing reconstruct_object.py + pipeline/ + images + prompts =="
TMP_TAR=/tmp/geo2_"$NAME".tar.gz
STAGE=/tmp/geo2_stage_"$NAME"
rm -rf "$STAGE" && mkdir -p "$STAGE/geo2_code" "$STAGE/images"
cp -r "$SCRIPT_DIR/pipeline" "$SCRIPT_DIR/reconstruct_object.py" "$STAGE/geo2_code/"
cp "$PROMPTS_FILE" "$STAGE/geo2_code/prompts.json"
cp -r "$IMAGE_DIR"/* "$STAGE/images/"
tar --owner=0 --group=0 --numeric-owner -C "$STAGE" -czf "$TMP_TAR" .
rm -rf "$STAGE"

ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" "mkdir -p $REMOTE_DIR"
scp "${SCP_OPTS[@]}" "$TMP_TAR" "$USERNAME@$HOST:$REMOTE_DIR/bundle.tar.gz"
rm -f "$TMP_TAR"

echo "== Running reconstruction on the GPU pod =="
ssh "${SSH_OPTS[@]}" "$USERNAME@$HOST" bash -s <<EOF
set -euo pipefail
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate geo2
cd $REMOTE_DIR
tar --no-same-owner -xzf bundle.tar.gz
ln -sf /workspace/geo2_checkpoints geo2_code/checkpoints
cd geo2_code
python -u reconstruct_object.py $NAME $REMOTE_DIR/images prompts.json cuda
EOF

echo "== Downloading results =="
mkdir -p "$SCRIPT_DIR/output/$NAME"
scp "${SCP_OPTS[@]}" -r "$USERNAME@$HOST:$REMOTE_DIR/geo2_code/output/$NAME/." "$SCRIPT_DIR/output/$NAME/"

echo "Done. Output: $SCRIPT_DIR/output/$NAME/"
