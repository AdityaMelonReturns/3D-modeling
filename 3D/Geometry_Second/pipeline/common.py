"""Stage [A]: Preprocess & Parse — shared utilities used by every other stage.

Loads a defect set (images + sidecar JSON), builds camera intrinsics per image,
and extracts the sensor-derived gravity vector / rotation estimate.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class ShotData:
    name: str                 # e.g. "ceiling__shot_05"
    image_path: str
    json_path: str
    width: int
    height: int
    focal_px: float           # focal length in pixels (derived from 35mm-equiv)
    K: np.ndarray             # 3x3 intrinsics matrix
    accel_g: np.ndarray       # raw accelerometer vector, device frame (gravity direction)
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    true_heading_deg: float
    heading_accuracy_deg: float
    gps: dict
    barometric_rel_alt_m: float
    raw_json: dict = field(repr=False, default=None)


def build_intrinsics(width_px, height_px, focal_35mm_equiv):
    """
    Standard 35mm-equivalent -> pixel focal length conversion:
        focal_px = (focal_35mm_equiv / 36mm) * image_width_px
    36mm is the full-frame sensor width the "35mm equivalent" convention is
    defined against. This is an approximation (COLMAP's own EXIF sensor-width
    database gave ~699.8px for this same camera on a 1920px-wide image vs.
    ~746.7px from this formula - about 7% apart, both are estimates, neither
    is a real calibration; treat focal_px as approximate, not ground truth).
    """
    focal_px = (focal_35mm_equiv / 36.0) * width_px
    cx, cy = width_px / 2.0, height_px / 2.0
    K = np.array([
        [focal_px, 0, cx],
        [0, focal_px, cy],
        [0, 0, 1],
    ])
    return focal_px, K


def load_shot(image_dir, metadata_dir, shot_name):
    """shot_name like 'ceiling__shot_05' (no extension)."""
    json_path = os.path.join(metadata_dir, shot_name + ".json")
    with open(json_path) as f:
        d = json.load(f)

    image_path = os.path.join(image_dir, d["image_file"])
    width, height = d["image"]["width_px"], d["image"]["height_px"]
    focal_35mm = d["intrinsics"]["focal_length_35mm_equiv"]
    focal_px, K = build_intrinsics(width, height, focal_35mm)

    accel = d["raw"]["accelerometer_g"]
    accel_g = np.array([accel["x"], accel["y"], accel["z"]])

    ori = d["orientation"]
    heading = d["heading"]

    return ShotData(
        name=shot_name,
        image_path=image_path,
        json_path=json_path,
        width=width,
        height=height,
        focal_px=focal_px,
        K=K,
        accel_g=accel_g,
        pitch_deg=ori["pitch_deg"],
        roll_deg=ori["roll_deg"],
        yaw_deg=ori["yaw_deg"],
        true_heading_deg=heading.get("true_heading_deg"),
        heading_accuracy_deg=heading.get("accuracy_deg"),
        gps=d["location"],
        barometric_rel_alt_m=d.get("barometric_relative_altitude_m"),
        raw_json=d,
    )


def load_defect_set(base_dir, defect_id):
    """
    base_dir: e.g. .../3D/Geometry/workspace/ceiling  (reuses the same
    images/ + metadata/ layout already extracted by the COLMAP pipeline -
    no need to re-unzip).
    Returns list[ShotData] sorted by shot index.
    """
    image_dir = os.path.join(base_dir, "images")
    metadata_dir = os.path.join(base_dir, "metadata")
    names = sorted(
        f[:-5] for f in os.listdir(metadata_dir) if f.endswith(".json")
    )
    return [load_shot(image_dir, metadata_dir, n) for n in names]


def load_image_rgb(shot: ShotData):
    return np.array(Image.open(shot.image_path).convert("RGB"))
