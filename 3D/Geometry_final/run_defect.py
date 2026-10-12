#!/usr/bin/env python
"""Orchestrator: Qwen3-VL (auto-box + mask check + interpretation) + SAM2
(defect mask) + DA3 (multi-view geometry) -> 3D defect measurement.
See DESIGN.md for the architecture, WALKTHROUGH.md for what each stage does.

No Depth Pro, no Claude - see DESIGN.md's "why one model per job" section.

Usage:
    python run_defect.py <defect_id> [--geometry-base PATH] [--defect-type TYPE] [--device cuda|cpu]

Reuses the images/ (+ optional metadata/) already extracted by the COLMAP
pipeline in ../Geometry/workspace/<defect_id>/ - no need to re-unzip.

Box prompts: if prompts/<defect_id>.json has an entry for a shot, that box
is used (manual override, same format as Geometry_third). Otherwise Qwen's
own auto-box is used. See DESIGN.md's rollout safety net.

Example:
    python run_defect.py hole --defect-type hole_or_rot
"""
import argparse
import json
import os
import signal
import sys

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
from pipeline.interpret import rule_based_severity
from pipeline.vlm_qwen import QwenVLM

DEFECT_TYPE_HINTS = {
    "hole_or_rot": "a hole, missing material, or rotten/decayed wood or plaster",
    "crack_or_gap": "a crack or gap in a surface",
    "inclination": "a leaning or tilted structure (e.g. railing, post, pergola)",
}


def mask_overlay(rgb, mask, alpha=0.45):
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    overlay[mask] = (overlay[mask] * (1 - alpha) + red[mask] * alpha).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("defect_id")
    ap.add_argument("--geometry-base",
                     default=os.path.join(os.path.dirname(__file__), "..", "Geometry", "workspace"))
    ap.add_argument("--defect-type", default="hole_or_rot",
                     choices=["hole_or_rot", "crack_or_gap", "inclination"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--da3-checkpoint", default="depth-anything/DA3-BASE")
    ap.add_argument("--qwen-checkpoint", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()

    base_dir = os.path.join(args.geometry_base, args.defect_id)
    out_dir = os.path.join(os.path.dirname(__file__), "output", args.defect_id)
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    defect_hint = DEFECT_TYPE_HINTS[args.defect_type]

    print(f"=== [A] Loading defect set: {args.defect_id} ===")
    shots = load_defect_set(base_dir, args.defect_id)
    print(f"{len(shots)} shots loaded")

    prompts_path = os.path.join(os.path.dirname(__file__), "prompts", f"{args.defect_id}.json")
    manual_prompts = load_prompts(prompts_path)
    if manual_prompts:
        print(f"{len(manual_prompts)} manual box prompts found in {prompts_path} - "
              f"these override Qwen's auto-box for their shots (see DESIGN.md)")

    print(f"\n=== [V1] Qwen quality gate + auto-box (device={args.device or 'auto'}) ===")
    qwen = QwenVLM(checkpoint=args.qwen_checkpoint, device=args.device)
    print(f"[vlm_qwen] {args.qwen_checkpoint} loaded")

    v1_results = {}
    shots_with_prompts = []
    for shot in shots:
        rgb = load_image_rgb(shot)
        if shot.name in manual_prompts:
            p = manual_prompts[shot.name]
            v1_results[shot.name] = {"usable": True, "box": p["box"], "source": "manual",
                                      "neg_points": p.get("neg_points")}
            shots_with_prompts.append(shot)
            print(f"{shot.name}: manual box {p['box']}")
            continue
        v1 = qwen.quality_gate_and_box(rgb, defect_type_hint=defect_hint)
        v1["source"] = "qwen"
        v1_results[shot.name] = v1
        if v1.get("usable") and v1.get("box"):
            shots_with_prompts.append(shot)
            print(f"{shot.name}: Qwen box {v1['box']} (guess: {v1.get('defect_type_guess')}) - {v1.get('reason')}")
        else:
            print(f"{shot.name}: SKIPPED - {v1.get('reason')}")

    if not shots_with_prompts:
        print(f"ERROR: no usable shots (Qwen marked all unusable, and no manual prompts). "
              f"Check {prompts_path} or the photos themselves.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(shots_with_prompts)}/{len(shots)} shots usable")
    names = [s.name for s in shots_with_prompts]

    print(f"\n=== [B+C] Multi-view geometry (DA3) ===")
    da3 = DA3Geometry(device=args.device, checkpoint=args.da3_checkpoint)
    print(f"[geometry_da3] {args.da3_checkpoint} loaded")
    image_paths = [s.image_path for s in shots_with_prompts]
    target_sizes = [(s.height, s.width) for s in shots_with_prompts]
    geom_results = da3.infer_multiview(image_paths, target_sizes=target_sizes)
    print(f"[geometry_da3] inferred {len(geom_results)} views jointly")
    for s, g in zip(shots_with_prompts, geom_results):
        print(f"  {s.name}: depth range {g['depth_m'].min():.2f}-{g['depth_m'].max():.2f} m, "
              f"mean conf {g['conf'].mean():.2f}")
    check_camera_layout(geom_results, names)

    gravity_mean, gravity_agreement = average_gravity(shots_with_prompts)
    if gravity_mean is not None:
        print(f"gravity-vector cross-shot agreement: {gravity_agreement:.3f}")
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
    overlays_dir = os.path.join(debug_dir, "mask_overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    seg_results = []
    for shot in shots_with_prompts:
        rgb = load_image_rgb(shot)
        v1 = v1_results[shot.name]
        seg = segmenter.segment(rgb, v1["box"], neg_points=v1.get("neg_points"))
        seg_results.append(seg)
        np.save(os.path.join(masks_dir, f"{shot.name}_mask.npy"), seg["mask"])
        Image.fromarray(mask_overlay(rgb, seg["mask"])).save(
            os.path.join(overlays_dir, f"{shot.name}_overlay.png"))
        print(f"{shot.name}: SAM2 score={seg['score']:.3f}, "
              f"{seg['mask'].sum()}px core / {seg['mask_dilated'].sum()}px dilated")

    print(f"\n=== [V2] Qwen mask verification ===")
    v2_results = {}
    for shot, seg in zip(shots_with_prompts, seg_results):
        overlay = np.array(Image.open(os.path.join(overlays_dir, f"{shot.name}_overlay.png")))
        v2 = qwen.verify_mask(overlay, defect_type_hint=defect_hint)
        v2_results[shot.name] = v2
        flag = "OK" if v2.get("matches") else "FLAGGED"
        print(f"{shot.name}: [{flag}] confidence={v2.get('confidence')} - {v2.get('reason')}")

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
    measurements["vlm_flagged_masks"] = [n for n, v in v2_results.items() if not v.get("matches")]
    print(json.dumps(measurements, indent=2, default=str))

    with open(os.path.join(out_dir, "measurements.json"), "w") as f:
        json.dump(measurements, f, indent=2, default=str)
    with open(os.path.join(debug_dir, "geometry_debug.json"), "w") as f:
        json.dump({
            "gravity_agreement": gravity_agreement,
            "scale_infos": [{k: v for k, v in s.items() if k != "detail"} | {"detail": s["detail"]} for s in scale_infos],
            "camera_centers": {n: g["cam_center_world"].tolist() for n, g in zip(names, geom_results)},
            "mean_conf_per_shot": {n: float(g["conf"].mean()) for n, g in zip(names, geom_results)},
            "v1_quality_gate": v1_results,
            "v2_mask_verification": v2_results,
        }, f, indent=2, default=str)

    full_pcd = fused_defect + fused_surrounding
    ply_path = export_pointcloud(full_pcd, out_dir)
    print(f"point cloud saved: {ply_path} ({len(full_pcd.points)} points)")

    print(f"\n=== [J] Interpretation ===")
    severity, flags = rule_based_severity(args.defect_type, measurements)
    interpretation = {"rule_based_severity": severity, "rule_based_flags": flags}
    best_idx = int(np.argmax([s["score"] for s in seg_results]))
    best_shot = shots_with_prompts[best_idx]
    qwen_result = qwen.describe_severity(
        load_image_rgb(best_shot), measurements, args.defect_id
    )
    interpretation["qwen_vision"] = qwen_result
    print(json.dumps(interpretation, indent=2, default=str))
    with open(os.path.join(out_dir, "interpretation.json"), "w") as f:
        json.dump(interpretation, f, indent=2, default=str)

    print(f"\n=== [I] Meshing + export (last, on purpose - see DESIGN.md) ===")

    def _mesh_timeout(signum, frame):
        raise TimeoutError("meshing exceeded 180s budget")

    try:
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
