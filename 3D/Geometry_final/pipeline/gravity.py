"""Sensor-derived gravity direction - a NARROWED-SCOPE reuse of Geometry_Second's
pose.py. In Geometry_Second, the phone's tilt/heading sensor was used to guess
the FULL camera pose (rotation + a heuristic translation) - that's exactly the
part that came back `pose_confidence: low` on every single defect set we ran,
and is now replaced by DA3 (geometry_da3.py), which works out real camera
positions from the photos themselves.

The one thing DA3 does NOT give us is "which way is down in the real world" -
that's still useful for the inclination (tilt) measurement, and the phone's
accelerometer is a perfectly good source for just that one number. So this
module keeps ONLY that narrow piece.

If a shot has no sensor metadata (see common.py), skip it - inclination
falls back to "no gravity reference available" rather than a fabricated
number.
"""
import numpy as np

# Empirically-fit device(accelerometer/motion)-frame -> camera-frame rotation.
# (unchanged from Geometry_Second/pipeline/pose.py - found via testing candidate
# 90-degree rotations against COLMAP poses on invlined_railing)
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
    """Best-effort device-to-world rotation from pitch/roll/heading, mapped
    into the camera frame via DEVICE_TO_CAMERA. Same caveat as before: this
    convention was fit on one defect set, not Apple-verified - treat as
    best-effort, which is fine since it now only feeds one cross-check number."""
    R_device_to_world = Rz(shot.true_heading_deg) @ Rx(shot.pitch_deg) @ Ry(shot.roll_deg)
    return R_device_to_world @ DEVICE_TO_CAMERA.T


def gravity_world(shot):
    """World-frame gravity direction implied by this shot's own sensor data.
    Returns None if the shot has no sensor metadata."""
    if not shot.has_sensor_data or shot.accel_g is None:
        return None
    R_c2w = sensor_rotation_camera_to_world(shot)
    return R_c2w @ (DEVICE_TO_CAMERA @ shot.accel_g)


def average_gravity(shots):
    """Mean gravity direction across all shots that have sensor data, plus
    an agreement score (1.0 = all shots agree perfectly on "down" - the same
    diagnostic Geometry_Second printed). Returns (None, None) if no shot in
    this set has sensor data at all."""
    vecs = [gravity_world(s) for s in shots]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None, None
    mean = np.mean(vecs, axis=0)
    mean = mean / np.linalg.norm(mean)
    agreement = float(np.mean([np.dot(v, mean) for v in vecs]))
    return mean, agreement
