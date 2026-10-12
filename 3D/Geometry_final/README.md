# Geometry_final

3D defect measurement from a handful of phone photos. Fourth iteration of this
project - see `DESIGN.md` for what changed and why, `WALKTHROUGH.md` for what
each stage actually does and what file it produces.

**Models used (exactly one per job, see DESIGN.md):**
- Qwen3-VL-30B-A3B-Instruct - reads photos, draws defect boxes, checks masks, writes the final description
- SAM2 - turns a box into a precise mask
- DA3 (DA3-BASE) - depth + real camera position, computed jointly across all photos of one defect

## Folder layout

```
Geometry_final/
  DESIGN.md              architecture + why (start here)
  README.md              this file
  WALKTHROUGH.md          stage-by-stage: what runs, what it produces
  pipeline/
    common.py            load a defect set's photos (+ optional sensor metadata)
    gravity.py            sensor-derived "which way is down", for the tilt measurement only
    vlm_qwen.py           Qwen3-VL: quality gate, auto-box, mask check, severity description
    segment.py             SAM2 box-prompt segmentation
    geometry_da3.py        DA3 multi-view depth + camera pose
    scale.py                ruler-in-frame scale calibration
    fusion.py               back-projection + multi-view point cloud merge
    plane.py                reference-plane fit
    measure.py              depth/width/length/area/volume/tilt from the fused points
    mesh_export.py          point cloud + mesh (alpha-shape) export
    interpret.py            rule-based severity thresholds
  run_defect.py             orchestrator - runs one defect end to end
  remote_setup_final_gpu.sh   one-time GPU pod environment setup
  run_defect_on_runpod.sh      ships code + one defect's photos to the pod, runs it, pulls results back
  prompts/                   optional manual box prompts (same format as Geometry_third) - if present for
                              a defect, used instead of Qwen's auto-box (see DESIGN.md's rollout safety net)
```

## Running one defect

Once the pod is up (`remote_setup_final_gpu.sh` has been run there once):

```bash
./run_defect_on_runpod.sh <defect_id> <ssh_host> <ssh_port> <ssh_user> <key_path> [defect_type]
# example:
./run_defect_on_runpod.sh hole 198.13.252.100 44017 root ~/.ssh/id_ed25519 hole_or_rot
```

Reuses the same `../Geometry/workspace/<defect_id>/images/` (+ optional
`metadata/`) that every previous folder in this project has used - no
re-unzipping needed.

Output lands in `output/<defect_id>/`:
- `measurements.json` - the numbers (depth/width/length/area/volume/tilt), written early so a later stage failing never loses them
- `interpretation.json` - rule-based severity + Qwen's plain-English description
- `defect_pointcloud.ply` - the real 3D point cloud, written before the mesh is attempted
- `mesh.ply` / `mesh.glb` - best-effort mesh (may be missing if that specific defect's points didn't support a clean surface - this is a known, disclosed limitation, not a bug to chase silently)
- `debug/` - saved masks, per-photo Qwen judgments, camera positions, everything needed to audit a run afterward

## Status

Code written, not yet run on a pod. First real run will tell us whether the
transformers-from-source install for Qwen3-VL plays nicely with the existing
`geo2` environment - see DESIGN.md's risk list.
