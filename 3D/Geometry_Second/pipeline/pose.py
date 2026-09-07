"""Stage [B]: Camera Pose Estimation — hybrid (feature-based + sensor-based).

Two independent signals, fused:
  1. LightGlue/SuperPoint pairwise matching -> match-count confidence per pair.
     (Used only as a confidence signal here, not full triangulated pose -
     with 4-9 images and no guaranteed connectivity, a from-scratch robust
     incremental SfM is exactly the "don't spend a day on this" trap the
     spec warns about. We still report match counts so low-overlap sets are
     visibly flagged, same failure mode COLMAP hit on peeling_paint.)
  2. Sensor-based absolute rotation per image, from pitch/roll + true_heading.
     Translation is filled in later (see fuse_translation) once stage [C]
     gives us a per-image median depth to place each camera along its own
     viewing ray.

IMPORTANT CAVEAT (documented, not hidden): the device-frame -> camera-frame
axis mapping used below (`DEVICE_TO_CAMERA`) was empirically fit on ONE
defect set (invlined_railing, 7 images) in the COLMAP-based analysis, by
testing which of several candidate 90-degree rotations made the gravity
vector agree most consistently across images once transformed through each
image's COLMAP-estimated rotation. It generalizes across sets IF the phone
was held the same way (same app, same portrait/landscape convention) for
every capture - which is likely but not verified per-set here. Treat
sensor-derived rotation as "best effort / regularizer", exactly as the spec
frames it - not as validated ground truth.
"""
import itertools
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, numpy_image_to_torch

# Empirically-fit device(accelerometer/motion)-frame -> camera-frame rotation.
# (see module docstring; found via inclination.py against invlined_railing)
DEVICE_TO_CAMERA = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
])


def Rz(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rx(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def sensor_rotation_camera_to_world(shot):
    """
    Best-effort device-to-world rotation from pitch/roll/heading, then mapped
    into the camera frame via DEVICE_TO_CAMERA. Convention: Rz(heading) @
    Rx(pitch) @ Ry(roll) - a reasonable, but not Apple-verified, intrinsic
    rotation order. Flagged low-confidence by design (see docstring).
    """
    R_device_to_world = Rz(shot.true_heading_deg) @ Rx(shot.pitch_deg) @ Ry(shot.roll_deg)
    R_camera_to_world = R_device_to_world @ DEVICE_TO_CAMERA.T
    return R_camera_to_world


def gravity_world(shot):
    """World-frame gravity direction implied by this shot's own sensor data,
    using the same DEVICE_TO_CAMERA + sensor rotation. Independent sanity
    check: should point the same way for every shot in a set if the
    sensor-rotation convention above is actually correct for this data."""
    R_c2w = sensor_rotation_camera_to_world(shot)
    return R_c2w @ (DEVICE_TO_CAMERA @ shot.accel_g)


def match_all_pairs(shots, max_keypoints=1024, device="cpu"):
    """Brute-force LightGlue matching across all pairs (fine for 4-20 images,
    per spec). Returns dict[(i,j)] = num_matches, plus per-image keypoints."""
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    feats = []
    for shot in shots:
        img = numpy_image_to_torch(_load_rgb_float(shot)).to(device)
        with torch.no_grad():
            f = extractor.extract(img[None])
        feats.append(f)

    pair_matches = {}
    for i, j in itertools.combinations(range(len(shots)), 2):
        with torch.no_grad():
            m = matcher({"image0": feats[i], "image1": feats[j]})
        m = rbd(m)
        pair_matches[(i, j)] = int(m["matches"].shape[0])

    return pair_matches, feats


def _load_rgb_float(shot):
    from PIL import Image
    return np.asarray(Image.open(shot.image_path).convert("RGB")).astype(np.float32) / 255.0


def pose_confidence(pair_matches, n_images, good_match_threshold=30):
    """Fraction of all possible pairs that had >= threshold matches.
    Cheap, honest proxy for 'did this set have enough overlap' - directly
    analogous to the registration-rate finding from the COLMAP pass."""
    total_pairs = n_images * (n_images - 1) // 2
    good_pairs = sum(1 for v in pair_matches.values() if v >= good_match_threshold)
    frac = good_pairs / total_pairs if total_pairs else 0.0
    if frac >= 0.6:
        return "high", frac
    elif frac >= 0.25:
        return "medium", frac
    else:
        return "low", frac


def fuse_translation(shots, rotations, median_depths, target_point=None):
    """
    Places each camera along its own optical-axis viewing ray, at a distance
    equal to that image's median scene depth (from stage [C]), assuming all
    cameras are roughly aimed at the same defect region (true for a
    walk-around capture of one defect - not true in general).

    If target_point is None, uses image 0's ray at its own median depth as
    the reference target point every other camera is assumed to look at.
    Returns list of camera centers (world frame), same order as shots.
    """
    def viewing_dir(R_c2w):
        # camera looks down +Z in its own frame (OpenCV convention)
        return R_c2w @ np.array([0, 0, 1.0])

    dirs = [viewing_dir(R) for R in rotations]

    if target_point is None:
        # anchor camera 0 at the origin, looking at a point `median_depths[0]` ahead
        target_point = np.array([0, 0, 0]) + dirs[0] * median_depths[0]

    centers = []
    for d, depth in zip(dirs, median_depths):
        center = target_point - d * depth
        centers.append(center)
    return centers
