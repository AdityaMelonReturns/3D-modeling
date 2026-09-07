#!/usr/bin/env python
"""Orchestrator: DA3 (multi-view geometry) + SAM2 (defect mask) + Depth Pro
(local crop cross-check) -> 3D defect measurement. See PLAN.md.

Usage:
    python run_defect.py <defect_id> [--geometry-base PATH] [--defect-type TYPE] [--device cuda|cpu]

Reuses the images/ + metadata/ already extracted by the COLMAP pipeline in
../Geometry/workspace/<defect_id>/ - no need to re-unzip anything. metadata/
is optional here (see pipeline/common.py).

Requires prompts/<defect_id>.json (box prompt per shot for SAM2 - identical
format/files to Geometry_Second, see pipeline/segment.py:load_prompts).

Example:
    python run_defect.py hole --defect-type hole_or_rot
"""
import argparse
import json
import os
import signal
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.common import load_defect_set, load_image_rgb
from pipeline.geometry_da3 import DA3Geometry, check_camera_layout
from pipeline.gravity import average_gravity
from pipeline.scale import detect_ruler, cross_check_against_depth
from pipeline.segment import DefectSegmenter, load_prompts
from pipeline.fusion import backproject, to_world, refine_and_merge
from pipeline.plane import fit_reference_plane
from pipeline.measure import measure_defect
from pipeline.mesh_export import build_mesh, export_pointcloud, export_mesh
from pipeline.interpret import rule_based_severity, claude_vision_interpret


def local_depth_pro_crosscheck(depth_est, shot, box_xyxy, da3_depth_m, da3_K, mask, margin_pct=0.25):
    """Stage [C']: crop tight around the defect box (with margin), run Depth
    Pro on JUST that crop, and compare its median depth in the masked region
    against DA3's own median depth there. This is a cross-check, not a
    correction - we report both and flag disagreement rather than blend them
    (see pipeline/depth_local.py docstring)."""
    x0, y0, x1, y1 = box_xyxy
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * margin_pct), int(h * margin_pct)
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(shot.width, x1 + mx), min(shot.height, y1 + my)

    rgb = load_image_rgb(shot)
    crop = rgb[cy0:cy1, cx0:cx1]
    mask_crop = mask[cy0:cy1, cx0:cx1]
    if mask_crop.sum() < 10:
        return None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        crop_path = tf.name
    try:
        Image.fromarray(crop).save(crop_path)
        f_px = float(da3_K[0, 0])  # focal length in pixels doesn't change when cropping
        depth_pro_crop = depth_est.infer_crop(crop_path, f_px=f_px)
    finally:
        os.remove(crop_path)

    da3_median = float(np.median(da3_depth_m[cy0:cy1, cx0:cx1][mask_crop]))
    dp_median = float(np.median(depth_pro_crop[mask_crop])) if depth_pro_crop.shape == mask_crop.shape else None
    if dp_median is None:
        return {"da3_median_m": da3_median, "depth_pro_median_m": None,
                "note": "crop shape mismatch, skipped comparison"}

    disagreement_pct = abs(da3_median - dp_median) / max(da3_median, 1e-6) * 100
    return {
        "da3_median_m": da3_median,
        "depth_pro_median_m": dp_median,
        "disagreement_pct": disagreement_pct,
        "agrees": disagreement_pct < 30,  # loose threshold, both are estimates
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("defect_id")
    ap.add_argument("--geometry-base",
                     default=os.path.join(os.path.dirname(__file__), "..", "Geometry", "workspace"))
    ap.add_argument("--defect-type", default="hole_or_rot",
                     choices=["hole_or_rot", "crack_or_gap", "inclination"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--da3-checkpoint", default="depth-anything/DA3-BASE")
    ap.add_argument("--skip-local-crosscheck", action="store_true",
                     help="skip the optional Depth Pro local crop cross-check (stage C')")
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
    names = [s.name for s in shots_with_prompts]

    print(f"\n=== [B+C] Multi-view geometry (DA3, device={args.device or 'auto'}) ===")
    t0 = time.time()
    da3 = DA3Geometry(device=args.device, checkpoint=args.da3_checkpoint)
    print(f"[geometry_da3] {args.da3_checkpoint} loaded in {time.time()-t0:.1f}s")
    image_paths = [s.image_path for s in shots_with_prompts]
    target_sizes = [(s.height, s.width) for s in shots_with_prompts]
    t0 = time.time()
    geom_results = da3.infer_multiview(image_paths, target_sizes=target_sizes)
    print(f"[geometry_da3] inferred {len(geom_results)} views jointly in {time.time()-t0:.1f}s")
    for s, g in zip(shots_with_prompts, geom_results):
        print(f"  {s.name}: depth range {g['depth_m'].min():.2f}-{g['depth_m'].max():.2f} m "
              f"(DA3's own scale, pre ruler-correction), mean conf {g['conf'].mean():.2f}")
    check_camera_layout(geom_results, names)

    gravity_mean, gravity_agreement = average_gravity(shots_with_prompts)
    if gravity_mean is not None:
        print(f"gravity-vector cross-shot agreement: {gravity_agreement:.3f} "
              f"(1.0 = perfect; sensor-derived, used only for the tilt measurement now)")
    else:
        print("no sensor metadata for this set - inclination measurement will be unavailable")

    print(f"\n=== [D] Scale calibration (ruler vs DA3's own depth) ===")
    scale_infos = []
    for shot, g in zip(shots_with_prompts, geom_results):
        ruler = detect_ruler(shot.image_path)
        check = cross_check_against_depth(ruler, g["depth_m"], g["K"])
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

    print(f"\n=== [C'] Local Depth Pro cross-check (best-scoring shot only) ===")
    crosscheck = None
    if not args.skip_local_crosscheck:
        best_idx = int(np.argmax([s["score"] for s in seg_results]))
        best_shot, best_seg, best_g = shots_with_prompts[best_idx], seg_results[best_idx], geom_results[best_idx]
        from pipeline.depth_local import DepthEstimator
        depth_est = DepthEstimator(device=args.device)
        crosscheck = local_depth_pro_crosscheck(
            depth_est, best_shot, prompts[best_shot.name]["box"],
            best_g["depth_m"], best_g["K"], best_seg["mask"],
        )
        if crosscheck:
            print(f"{best_shot.name} (best SAM2 score): {json.dumps(crosscheck, indent=2)}")
        else:
            print("skipped (mask too small in crop)")
    else:
        print("skipped (--skip-local-crosscheck)")

    print(f"\n=== [F] Back-projection + fusion ===")
    defect_pcds, surrounding_pcds = [], []
    for shot, g, seg, check in zip(shots_with_prompts, geom_results, seg_results, scale_infos):
        depth_corrected = g["depth_m"] * check["correction_scalar"]
        rgb = load_image_rgb(shot)
        core_pcd = backproject(depth_corrected, g["K"], seg["mask"], rgb)
        surround_mask = seg["mask_dilated"] & ~seg["mask"]
        surround_pcd = backproject(depth_corrected, g["K"], surround_mask, rgb)
        defect_pcds.append(to_world(core_pcd, g["R_c2w"], g["cam_center_world"]))
        surrounding_pcds.append(to_world(surround_pcd, g["R_c2w"], g["cam_center_world"]))

    fused_defect = refine_and_merge(defect_pcds)
    fused_surrounding = refine_and_merge(surrounding_pcds)
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
    measurements["scale_source"] = "card_verified" if any(s["reliable"] for s in scale_infos) else "model_estimate"
    measurements["geometry_engine"] = args.da3_checkpoint
    if crosscheck:
        measurements["local_depth_crosscheck"] = crosscheck
    print(json.dumps(measurements, indent=2, default=str))

    # Write measurements + debug info NOW, before meshing - if Poisson meshing
    # hangs or crashes on a large/awkward point cloud (hit this live on
    # `ceiling`, 67k points), we keep the numbers regardless. Meshing is a
    # nice-to-have visualization, not the thing worth losing a whole run over.
    with open(os.path.join(out_dir, "measurements.json"), "w") as f:
        json.dump(measurements, f, indent=2, default=str)
    with open(os.path.join(debug_dir, "geometry_debug.json"), "w") as f:
        json.dump({
            "gravity_agreement": gravity_agreement,
            "scale_infos": [{k: v for k, v in s.items() if k != "detail"} | {"detail": s["detail"]} for s in scale_infos],
            "camera_centers": {n: g["cam_center_world"].tolist() for n, g in zip(names, geom_results)},
            "mean_conf_per_shot": {n: float(g["conf"].mean()) for n, g in zip(names, geom_results)},
        }, f, indent=2, default=str)

    # Point cloud export is CHEAP and can't hang (no Poisson involved) - do
    # this before attempting the mesh, not bundled after it. Losing this
    # exact ordering is why 7 of the first 8 real runs ended up with a
    # measurements.json but no .ply at all: killing a stuck meshing process
    # took the point-cloud write down with it, since the old export_all()
    # wrote both together, mesh first.
    full_pcd = fused_defect + fused_surrounding
    ply_path = export_pointcloud(full_pcd, out_dir)
    print(f"point cloud saved: {ply_path} ({len(full_pcd.points)} points)")

    # [J] Interpretation moved BEFORE [I] meshing, deliberately - same reason
    # as the point cloud above. interpretation.json used to be written after
    # meshing too, so it was ALSO getting lost every time a stuck mesh got
    # killed (7 of 8 real runs). It's cheap (no GPU/heavy compute), so do it
    # first and get it safely on disk before touching the one risky step.
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

    with open(os.path.join(out_dir, "interpretation.json"), "w") as f:
        json.dump(interpretation, f, indent=2, default=str)

    print(f"\n=== [I] Meshing + export (last, on purpose - see above) ===")

    def _mesh_timeout(signum, frame):
        raise TimeoutError("meshing exceeded 180s budget")

    try:
        # Hard wall-clock cap, not just a try/except - Open3D's normal-
        # orientation propagation can genuinely HANG (not raise) on some
        # point cloud shapes, not just error out. Saw this live on `ceiling`
        # (67k points). Everything else is already safely on disk by now
        # regardless of what happens here.
        old_handler = signal.signal(signal.SIGALRM, _mesh_timeout)
        signal.alarm(180)
        try:
            mesh, d = build_mesh(full_pcd, normal, point_on_plane)
            paths = export_mesh(mesh, out_dir)
            print(f"exported: {paths}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except Exception as e:
        print(f"!! Meshing failed/skipped ({e}) - everything else is already saved regardless.")

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
