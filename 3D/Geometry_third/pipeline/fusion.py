"""Stage [F]: Back-projection + Multi-view Fusion.

Each masked depth map -> colored point cloud in camera frame -> world frame
via stage [B]'s (approximate) pose -> merged, then refined with pairwise
colored ICP (Open3D) to soak up residual pose error - this is what
compensates for [B] being a best-effort estimate rather than a bundle-
adjusted solve.
"""
import numpy as np
import open3d as o3d


def backproject(depth_m, K, mask, rgb=None, max_points=200_000):
    """depth_m: HxW meters. mask: HxW bool (use the dilated defect mask).
    Returns an Open3D point cloud in CAMERA frame."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return o3d.geometry.PointCloud()
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]

    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    xs, ys, z = xs[valid], ys[valid], z[valid]

    X = (xs - K[0, 2]) * z / K[0, 0]
    Y = (ys - K[1, 2]) * z / K[1, 1]
    pts = np.stack([X, Y, z], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        colors = rgb[ys, xs].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def to_world(pcd_cam, R_c2w, t_c2w):
    """Camera frame -> world frame: X_world = R_c2w @ X_cam + t_c2w."""
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w
    return pcd_cam.transform(T)


def refine_and_merge(world_pcds, icp_max_corr_dist=0.05, voxel_size=0.005):
    """Pairwise colored ICP refinement (each cloud registered against the
    running merged cloud), then merge + voxel-downsample.

    icp_max_corr_dist is in METERS - defaults assume the scale-corrected
    (stage [D]) depth is already in real units; if scale is uncorrected
    ("model_estimate"), this threshold may need adjusting per set.
    """
    if not world_pcds:
        return o3d.geometry.PointCloud()

    def _with_normals(pcd, radius=0.02, max_nn=30):
        if not pcd.has_normals() and len(pcd.points) >= 3:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        return pcd

    merged = _with_normals(world_pcds[0])
    for pcd in world_pcds[1:]:
        if len(pcd.points) == 0 or len(merged.points) == 0:
            merged = merged + pcd
            continue
        pcd = _with_normals(pcd)
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                pcd, merged,
                max_correspondence_distance=icp_max_corr_dist,
                estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            )
            pcd = pcd.transform(result.transformation)
        except Exception as e:
            print(f"[fusion] colored ICP failed ({e}), using raw pose-based alignment")
        merged = merged + pcd
        merged = _with_normals(merged)

    if voxel_size:
        merged = merged.voxel_down_sample(voxel_size)
    return merged
