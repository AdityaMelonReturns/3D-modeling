"""Stage [H]: Measurement Extraction.

Everything here is computed relative to the reference plane from stage [G]:
  signed distance d = n . (x - p0)  for each DEFECT point x (not the
  surrounding points - those defined the plane, these are what we measure).
"""
import numpy as np
from scipy.spatial import ConvexHull

from .plane import signed_distance


def _project_to_plane_2d(points, normal, point_on_plane):
    """Drop the normal component; return 2D coords in an arbitrary but
    consistent in-plane basis (u, v)."""
    # any vector not parallel to normal
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, normal)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    centered = points - point_on_plane
    return np.stack([centered @ u, centered @ v], axis=1)


def measure_defect(defect_points_world, normal, point_on_plane, gravity_world_dir,
                    device_pitch_deg=None, device_roll_deg=None):
    """
    defect_points_world: Nx3 fused, world-frame points belonging to the
    defect (i.e. the SAM2 mask region, NOT dilated/surrounding).
    gravity_world_dir: unit vector, "down" in world frame (from stage B).

    Returns the measurements.json-shaped dict (minus defect_id/shots_used,
    added by the orchestrator).
    """
    n_pts = len(defect_points_world)
    if n_pts < 10:
        return {
            "depth_mm": {"max": None, "mean": None, "confidence": "low"},
            "width_mm": None, "length_mm": None, "area_mm2": None, "volume_mm3": None,
            "inclination_deg": {"value": None, "source_agreement": None, "confidence": "low"},
            "point_count": n_pts,
            "note": f"only {n_pts} defect points after fusion - below reliable "
                    f"measurement threshold, not fabricating a number (spec Sec.4)",
        }

    d = signed_distance(defect_points_world, normal, point_on_plane)
    depth_max_m = float(np.max(np.abs(d)))
    depth_mean_m = float(np.mean(np.abs(d)))

    xy = _project_to_plane_2d(defect_points_world, normal, point_on_plane)
    try:
        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
        area_m2 = float(hull.volume)  # 2D hull: .volume IS the area in scipy
    except Exception as e:
        hull_pts, area_m2 = xy, None

    # PCA on hull for major/minor (length/width)
    centered = hull_pts - hull_pts.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    major_axis = evecs[:, order[0]]
    minor_axis = evecs[:, order[1]]
    length_m = float((centered @ major_axis).max() - (centered @ major_axis).min())
    width_m = float((centered @ minor_axis).max() - (centered @ minor_axis).min())

    volume_m3 = area_m2 * depth_mean_m if area_m2 is not None else None

    # inclination: angle between reference-plane normal and gravity
    cos_angle = np.clip(np.dot(normal, gravity_world_dir) /
                         (np.linalg.norm(normal) * np.linalg.norm(gravity_world_dir)), -1, 1)
    angle_deg = float(np.degrees(np.arccos(abs(cos_angle))))

    # second opinion: raw device pitch/roll, if given, as an independent
    # per-shot cross-check (spec explicitly asks to report both + flag
    # disagreement, not silently trust the plane-fit version alone)
    source_agreement = None
    if device_pitch_deg is not None and device_roll_deg is not None:
        device_tilt_estimate = float(np.degrees(np.arccos(
            np.clip(np.cos(np.radians(device_pitch_deg)) * np.cos(np.radians(device_roll_deg)), -1, 1)
        )))
        source_agreement = abs(device_tilt_estimate - angle_deg) < 10  # loose, both are noisy

    if angle_deg < 2.0:
        inclination_confidence = "low"
        inclination_note = "within ~2deg sensor/plane-fit noise floor - report as 'no measurable lean', not a precise number (spec Sec.4)"
    else:
        inclination_confidence = "medium"
        inclination_note = None

    depth_confidence = "medium" if n_pts >= 200 else "low"

    return {
        "depth_mm": {"max": depth_max_m * 1000, "mean": depth_mean_m * 1000, "confidence": depth_confidence},
        "width_mm": width_m * 1000,
        "length_mm": length_m * 1000,
        "area_mm2": area_m2 * 1e6 if area_m2 is not None else None,
        "volume_mm3": volume_m3 * 1e9 if volume_m3 is not None else None,
        "inclination_deg": {
            "value": angle_deg, "source_agreement": source_agreement,
            "confidence": inclination_confidence, "note": inclination_note,
        },
        "point_count": n_pts,
    }
