"""Stage [G]: Reference Plane Fitting.

Fit a plane to the NON-defect (surrounding, dilated-mask-minus-core-mask)
points only - this is the "what the surface should look like without the
defect" baseline every measurement in stage [H] is relative to.

This directly targets the exact segmentation gap the COLMAP-based pilot
hit on invlined_railing (Geometry/REPORT.md Sec.4): there, fitting a plane
to the WHOLE scene (deck + siding + bushes + railing all mixed) gave two
different answers (29 vs 36 deg) depending on what points were included.
Here the SAM2 mask (stage E) does that isolation up front, so the plane fit
should only ever see the one surface being measured.
"""
import numpy as np
import open3d as o3d


def fit_reference_plane(surrounding_pcd, ransac_thresh=0.01, ransac_n=3, iterations=1000):
    """RANSAC plane fit via Open3D. Returns (normal (3,) unit vector,
    point_on_plane (3,), inlier_ratio float)."""
    if len(surrounding_pcd.points) < ransac_n:
        raise ValueError(f"Not enough surrounding points to fit a plane "
                          f"({len(surrounding_pcd.points)} < {ransac_n}) - "
                          f"widen the mask dilation in stage [E].")
    plane_model, inliers = surrounding_pcd.segment_plane(
        distance_threshold=ransac_thresh, ransac_n=ransac_n, num_iterations=iterations
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    pts = np.asarray(surrounding_pcd.points)
    point_on_plane = pts[inliers].mean(axis=0) if len(inliers) else pts.mean(axis=0)
    inlier_ratio = len(inliers) / len(pts)
    return normal, point_on_plane, inlier_ratio


def signed_distance(points, normal, point_on_plane):
    return (points - point_on_plane) @ normal
