"""Stage [A]: Load one defect set's photos.

Unlike Geometry_Second, the sensor-metadata JSON sidecar is now OPTIONAL -
DA3 (geometry_da3.py) works out camera position from the photos themselves,
it doesn't need the phone's tilt/heading sensor for that anymore. When the
metadata IS present (all 8 real defect sets have it, from the capture app),
we still read pitch/roll/heading/accelerometer, but only as a secondary
"which way is down" signal for the inclination measurement (see gravity.py) -
never as the primary pose source. This also means a plain zip with no
metadata sidecar (e.g. a generic photo dump) loads fine here; it just won't
get the inclination cross-check.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class ShotData:
    name: str                 # filename without extension
    image_path: str
    width: int
    height: int
    focal_px: Optional[float] = None      # only set if metadata JSON had it (used as a
                                           # fallback if DA3/Depth Pro can't read real EXIF)
    # sensor fields - all None if no metadata sidecar was found for this shot
    accel_g: Optional[np.ndarray] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    true_heading_deg: Optional[float] = None
    raw_json: dict = field(repr=False, default=None)

    @property
    def has_sensor_data(self):
        return self.pitch_deg is not None and self.roll_deg is not None and self.true_heading_deg is not None


def _read_metadata(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def load_defect_set(base_dir, defect_id=None):
    """base_dir: a folder with an images/ subfolder (metadata/ optional).
    Returns list[ShotData], sorted by filename.

    Also tolerates base_dir being the images folder itself (no images/
    subfolder) - useful for ad-hoc photo dumps with no companion metadata/.
    """
    image_dir = os.path.join(base_dir, "images")
    metadata_dir = os.path.join(base_dir, "metadata")
    if not os.path.isdir(image_dir):
        image_dir = base_dir
        metadata_dir = os.path.join(base_dir, "metadata")  # usually won't exist either

    names = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS)
    )
    shots = []
    for fname in names:
        shot_name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        with Image.open(image_path) as im:
            width, height = im.size

        d = _read_metadata(os.path.join(metadata_dir, shot_name + ".json"))
        if d is None:
            shots.append(ShotData(name=shot_name, image_path=image_path, width=width, height=height))
            continue

        focal_px = None
        if "intrinsics" in d and "focal_length_35mm_equiv" in d.get("intrinsics", {}):
            focal_px = (d["intrinsics"]["focal_length_35mm_equiv"] / 36.0) * width

        accel = d.get("raw", {}).get("accelerometer_g")
        ori = d.get("orientation", {})
        heading = d.get("heading", {})
        shots.append(ShotData(
            name=shot_name,
            image_path=image_path,
            width=width, height=height,
            focal_px=focal_px,
            accel_g=np.array([accel["x"], accel["y"], accel["z"]]) if accel else None,
            pitch_deg=ori.get("pitch_deg"),
            roll_deg=ori.get("roll_deg"),
            yaw_deg=ori.get("yaw_deg"),
            true_heading_deg=heading.get("true_heading_deg"),
            raw_json=d,
        ))
    return shots


def load_image_rgb(shot: ShotData):
    return np.array(Image.open(shot.image_path).convert("RGB"))
