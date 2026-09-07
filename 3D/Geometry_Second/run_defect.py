#!/usr/bin/env python
"""Orchestrator: runs stages A-J for one defect set.

Usage:
    python run_defect.py <defect_id> [--geometry-base PATH] [--defect-type TYPE] [--device cuda|cpu]

Reuses the images/ + metadata/ already extracted by the COLMAP pipeline in
../Geometry/workspace/<defect_id>/ - no need to re-unzip anything.

Requires prompts/<defect_id>.json (box prompt per shot for SAM2 - see
pipeline/segment.py:load_prompts for the format). Skips shots with no entry.

Example:
    python run_defect.py ceiling --defect-type hole_or_rot
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.common import load_defect_set, load_image_rgb
from pipeline.pose import (
    sensor_rotation_camera_to_world, gravity_world, match_all_pairs,
    pose_confidence, fuse_translation,
)
from pipeline.depth import DepthEstimator
from pipeline.scale import detect_ruler, cross_check_against_depth
from pipeline.segment import DefectSegmenter, load_prompts
from pipeline.fusion import backproject, to_world, refine_and_merge
from pipeline.plane import fit_reference_plane
from pipeline.measure import measure_defect
from pipeline.mesh_export import build_mesh, export_all
from pipeline.interpret import rule_based_severity, claude_vision_interpret

import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("defect_id")
    ap.add_argument("--geometry-base",
                     default=os.path.join(os.path.dirname(__file__), "..", "Geometry", "workspace"))
    ap.add_argument("--defect-type", default="hole_or_rot",
                     choices=["hole_or_rot", "crack_or_gap", "inclination"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-fusion-icp", action="store_true",
                     help="use raw pose-based alignment only, skip colored ICP refinement")
    args = ap.parse_args()

    base_dir = os.path.join(args.geometry_base, args.defect_id)
    out_dir = os.path.join(os.path.dirname(__file__), "output", args.defect_id)
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    print(f"=== [A] Loading defect set: {args.defect_id} ===")
    shots = load_defect_set(base_dir, args.defect_id)
    print(f"{len(shots)} shots loaded")

    prompts_path = os.path.join(os.path.dirname(__file__), "prompts", f"{args.defect_id}.json")
    prompts = load_prompts(prompts_path)
    shots_with_prompts = [s for s in shots if s.name in prompts]
    if not shots_with_prompts:
        print(f"ERROR: no box prompts found in {prompts_path} for any of "
              f"{[s.name for s in shots]}. Add entries and rerun.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(shots_with_prompts)}/{len(shots)} shots have a defect box prompt "
          f"- only these are used (others skipped)")

    print(f"\n=== [B] Pose estimation ({len(shots_with_prompts)} shots) ===")
    pair_matches, _ = match_all_pairs(shots_with_prompts, device="cpu")
    confidence, frac = pose_confidence(pair_matches, len(shots_with_prompts))
    print(f"pose_confidence: {confidence} ({frac*100:.0f}% of pairs had >=30 matches)")

    rotations = [sensor_rotation_camera_to_world(s) for s in shots_with_prompts]
    gravities = [gravity_world(s) for s in shots_with_prompts]
    gravity_mean = np.mean(gravities, axis=0)
    gravity_mean /= np.linalg.norm(gravity_mean)
    gravity_agreement = np.mean([np.dot(g, gravity_mean) for g in gravities])
    print(f"gravity-vector cross-shot agreement: {gravity_agreement:.3f} "
          f"(1.0 = perfect; see Geometry/REPORT.md Sec.4 for why this matters)")

    print(f"\n=== [C] Monocular depth (Depth Pro, device={args.device or 'auto'}) ===")
    depth_est = DepthEstimator(device=args.device)
    depths, scale_infos = [], []
    for shot in shots_with_prompts:
        depth_result = depth_est.infer(shot)
        depths.append(depth_result["depth_m"])

    print(f"\n=== [D] Scale calibration ===")
    for shot, depth_m in zip(shots_with_prompts, depths):
        ruler = detect_ruler(shot.image_path)
        check = cross_check_against_depth(ruler, depth_m, shot.K)
        scale_infos.append(check)
        print(f"{shot.name}: {check['detail']}")

    print(f"\n=== [E] Segmentation (SAM2) ===")
    segmenter = DefectSegmenter(device=args.device)
    masks_dir = os.path.join(debug_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    seg_results = []
    for shot in shots_with_prompts:
        rgb = load_image_rgb(shot)
        p = prompts[shot.name]
        seg = segmenter.segment(rgb, p["box"], neg_points=p.get("neg_points"))
        seg_results.append(seg)
        np.save(os.path.join(masks_dir, f"{shot.name}_mask.npy"), seg["mask"])
        print(f"{shot.name}: SAM2 score={seg['score']:.3f}, "
              f"{seg['mask'].sum()}px core / {seg['mask_dilated'].sum()}px dilated")

    median_depths = [float(np.median(d[s["mask_dilated"]])) if s["mask_dilated"].sum() else float(np.median(d))
                      for d, s in zip(depths, seg_results)]
    centers = fuse_translation(shots_with_prompts, rotations, median_depths)

    print(f"\n=== [F] Back-projection + fusion ===")
    defect_pcds, surrounding_pcds = [], []
    for shot, depth_m, seg, check, R, center in zip(
        shots_with_prompts, depths, seg_results, scale_infos, rotations, centers
    ):
        depth_corrected = depth_m * check["correction_scalar"]
        rgb = load_image_rgb(shot)
        core_pcd = backproject(depth_corrected, shot.K, seg["mask"], rgb)
        surround_mask = seg["mask_dilated"] & ~seg["mask"]
        surround_pcd = backproject(depth_corrected, shot.K, surround_mask, rgb)
        defect_pcds.append(to_world(core_pcd, R, center))
        surrounding_pcds.append(to_world(surround_pcd, R, center))

    icp_dist = 0.0 if args.skip_fusion_icp else 0.05
    fused_defect = refine_and_merge(defect_pcds, icp_max_corr_dist=icp_dist if icp_dist else 1e9)
    fused_surrounding = refine_and_merge(surrounding_pcds, icp_max_corr_dist=icp_dist if icp_dist else 1e9)
    print(f"fused defect points: {len(fused_defect.points)}, "
          f"surrounding points: {len(fused_surrounding.points)}")

    print(f"\n=== [G] Reference plane fit ===")
    normal, point_on_plane, inlier_ratio = fit_reference_plane(fused_surrounding)
    print(f"plane normal={normal}, inlier_ratio={inlier_ratio:.2f}")

    print(f"\n=== [H] Measurements ===")
    measurements = measure_defect(
        np.asarray(fused_defect.points), normal, point_on_plane, gravity_mean,
        device_pitch_deg=shots_with_prompts[0].pitch_deg,
        device_roll_deg=shots_with_prompts[0].roll_deg,
    )
    measurements["defect_id"] = args.defect_id
    measurements["shots_used"] = len(shots_with_prompts)
    measurements["pose_confidence"] = confidence
    measurements["scale_source"] = "card_verified" if any(s["reliable"] for s in scale_infos) else "model_estimate"
    print(json.dumps(measurements, indent=2, default=str))

    print(f"\n=== [I] Meshing + export ===")
    full_pcd = fused_defect + fused_surrounding
    mesh, d = build_mesh(full_pcd, normal, point_on_plane)
    paths = export_all(mesh, full_pcd, out_dir)
    print(f"exported: {paths}")

    print(f"\n=== [J] Interpretation ===")
    severity, flags = rule_based_severity(args.defect_type, measurements)
    interpretation = {"rule_based_severity": severity, "rule_based_flags": flags}
    claude_result = claude_vision_interpret(
        shots_with_prompts[0].image_path, measurements, args.defect_id
    )
    if claude_result is None:
        interpretation["claude_vision"] = None
        interpretation["note"] = "ANTHROPIC_API_KEY not set - skipped vision confirmation, rule-based only"
    else:
        interpretation["claude_vision"] = claude_result

    with open(os.path.join(out_dir, "measurements.json"), "w") as f:
        json.dump(measurements, f, indent=2, default=str)
    with open(os.path.join(out_dir, "interpretation.json"), "w") as f:
        json.dump(interpretation, f, indent=2, default=str)
    with open(os.path.join(debug_dir, "poses.json"), "w") as f:
        json.dump({
            "pose_confidence": confidence, "pair_matches": {str(k): v for k, v in pair_matches.items()},
            "gravity_agreement": float(gravity_agreement),
            "scale_infos": [{k: v for k, v in s.items() if k != "detail"} | {"detail": s["detail"]} for s in scale_infos],
        }, f, indent=2, default=str)

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
