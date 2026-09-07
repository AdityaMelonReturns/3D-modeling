"""Standalone: general-purpose 3D reconstruction for a small set of ordinary
photos WITHOUT the sensor-metadata sidecar the main pipeline (run_defect.py)
relies on. Built for the `bottle` sanity-check set: 3 plain Samsung Galaxy A35
photos, no ruler in frame, no IMU/orientation JSON - so, unlike run_defect.py:

  - no sensor-based pose (Stage [B] there needs pitch/roll/heading JSON these
    photos don't have)
  - no scale correction (Stage [D] there needs a ruler in frame - none here,
    so this output is in Depth Pro's own UNCORRECTED units. Every ruler
    cross-check we've run elsewhere in this project found Depth Pro's raw
    absolute scale 60-99% off - so treat this model's absolute size as
    illustrative shape only, not a measurement)
  - the relative pose between the 3 views is found by point-cloud
    registration (Open3D FPFH+RANSAC global registration, then ICP refine)
    instead of IMU/heading, since there's no sensor data to seed it from.

Usage: python reconstruct_object.py <name> <image_dir> <prompts.json> [device]
Reads prompts.json as {"<image_filename>": [x0,y0,x1,y1], ...}
Writes output/<name>/{pointcloud.ply, mesh.ply, mesh.glb}
"""
import sys
import os
import json

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.common import build_intrinsics
from pipeline.depth import DepthEstimator
from pipeline.segment import DefectSegmenter
from pipeline.fusion import backproject

FOCAL_35MM_EQUIV = 25.0  # from this phone's EXIF (Samsung Galaxy A35 5G, confirmed per-shot)


class Shot:
    """Minimal stand-in for pipeline.common.ShotData - just enough fields
    for DepthEstimator.infer() (image_path, width, height, focal_px)."""
    def __init__(self, name, image_path, width, height):
        self.name = name
        self.image_path = image_path
        self.width = width
        self.height = height
        self.focal_px, self.K = build_intrinsics(width, height, FOCAL_35MM_EQUIV)


def global_register(source, target, voxel_size=0.02):
    """FPFH-feature RANSAC global registration - unlike fusion.refine_and_merge
    (which assumes sensor pose already gives a decent initial alignment),
    this needs none: it searches for the best rigid transform from scratch."""
    def prep(pcd):
        down = pcd.voxel_down_sample(voxel_size)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return down, fpfh

    src_down, src_fpfh = prep(source)
    tgt_down, tgt_fpfh = prep(target)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result.transformation


def refine_icp(source, target, trans_init, max_corr):
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_corr, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result.transformation


def main():
    name = sys.argv[1]
    image_dir = sys.argv[2]
    prompts_path = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) > 4 else "cuda"

    with open(prompts_path) as f:
        prompts = json.load(f)

    print(f"=== Loading Depth Pro + SAM2 (device={device}) ===")
    depth_est = DepthEstimator(device=device)
    segmenter = DefectSegmenter(device=device)

    clouds = []
    for shot_name, box in prompts.items():
        img_path = os.path.join(image_dir, shot_name)
        rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = rgb.shape[:2]
        shot = Shot(shot_name, img_path, w, h)

        depth_m = depth_est.infer(shot)["depth_m"]

        seg = segmenter.segment(rgb, box, dilate_pct=0.05)
        print(f"[sam2] {shot_name}: score={seg['score']:.3f}, {seg['mask'].sum()}px masked")

        # Save a mask overlay for a quick visual sanity check - big generous
        # boxes on a plain-floor background risk SAM2 grabbing floor/background
        # along with the object; better to catch that here than trust the mesh.
        overlay_dir = os.path.join(os.path.dirname(__file__), "output", name, "debug")
        os.makedirs(overlay_dir, exist_ok=True)
        overlay = rgb.copy()
        red = np.zeros_like(rgb)
        red[..., 0] = 255
        alpha = 0.45
        overlay[seg["mask"]] = (overlay[seg["mask"]] * (1 - alpha) + red[seg["mask"]] * alpha).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(overlay_dir, shot_name.replace(".jpg", "") + "_mask_overlay.png"))

        pcd = backproject(depth_m, shot.K, seg["mask"], rgb)
        clouds.append((shot_name, pcd))

    print("=== Registering views (no sensor pose available - FPFH+RANSAC then ICP) ===")
    merged = clouds[0][1]
    for shot_name, pcd in clouds[1:]:
        try:
            voxel = 0.02
            T = global_register(pcd, merged, voxel_size=voxel)
            T = refine_icp(pcd, merged, T, max_corr=voxel * 1.5)
            pcd = pcd.transform(T)
            print(f"  {shot_name}: registered")
        except Exception as e:
            print(f"  {shot_name}: registration failed ({e}), appending unregistered (will look off)")
        merged = merged + pcd

    merged = merged.voxel_down_sample(0.005)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"merged cloud: {len(merged.points)} points")

    print("=== Poisson meshing ===")
    merged.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    merged.orient_normals_consistent_tangent_plane(30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged, depth=9)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))

    out_dir = os.path.join(os.path.dirname(__file__), "output", name)
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, "pointcloud.ply"), merged)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.ply"), mesh)

    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None,
        process=False,
    )
    tm.export(os.path.join(out_dir, "mesh.glb"))
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
