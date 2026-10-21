"""Angle against gravity, in THIS photo's camera frame.

Old pipelines mixed sensor-world and DA3-world frames (why walls read ~90 deg).
Here gravity never leaves the photo it was recorded with:

  1. Accelerometer (iOS device frame: x right, y toward top of phone,
     z out of the screen) at rest reads the gravity direction.
  2. The JPEG pixels are stored in the sensor's landscape layout (EXIF
     orientation 6). In that raw layout: image +x = device -y, image +y
     (down) = device -x, and the rear camera looks along device -z. So
         g_cam = (-a_y, -a_x, -a_z)       (x right, y down, z forward)
  3. Every true vertical line in the scene points at the vertical vanishing
     point v = K g_cam. So the "true vertical" direction differs per pixel -
     at pixel p it is the direction from p toward v. This is what makes the
     result correct even when the phone is pitched up or down.

Validated on all 82 test photos: of the 1842 long straight lines found, the
share lying within 2 deg of this predicted vertical is 10.9 % (true
verticals: posts, door frames, siding), versus <= 4.2 % for every one of
the other 7 possible axis mappings, and 1.7 % with the forward sign flipped.
The nominal EXIF focal (14 mm-equiv -> 747 px at 1920 wide) also fit best.
"""
import numpy as np
import cv2


def gravity_cam(accel_g):
    ax, ay, az = accel_g
    g = np.array([-ay, -ax, -az], float)
    return g / np.linalg.norm(g)


def vertical_dir_at(pts, g_cam, focal_px, cx, cy):
    """Unit image direction of 'straight down' at each pixel in pts (N,2)."""
    pts = np.asarray(pts, float)
    if abs(g_cam[2]) < 1e-6:
        d = np.tile(g_cam[:2], (len(pts), 1))
    else:
        v = np.array([focal_px * g_cam[0] / g_cam[2] + cx, focal_px * g_cam[1] / g_cam[2] + cy])
        d = (v - pts) * np.sign(g_cam[2])
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


def camera_tilt(g_cam):
    """Phone pitch (camera axis above/below horizontal) and roll, in degrees."""
    pitch = float(np.degrees(np.arcsin(np.clip(-g_cam[2], -1, 1))))   # + = looking up
    roll = float(np.degrees(np.arctan2(g_cam[0], g_cam[1])))           # 0 = upright in raw frame
    return pitch, roll


def member_angles(img_rgb, mask, accel_g, focal_px, min_len_frac=0.12):
    """Straight edges of the structure inside its mask, each compared with true
    vertical at its own position. Returns length-weighted lean of the
    near-vertical members (posts) and tilt-from-horizontal of the others
    (rails, beams), plus the raw segments for the overlay."""
    h, w = mask.shape
    g = gravity_cam(accel_g)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
    band = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    edges[~band] = 0
    ys, xs = np.nonzero(mask)
    span = np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) if len(xs) else min(h, w)
    L = cv2.HoughLinesP(edges, 1, np.pi / 720, 60, minLineLength=int(min_len_frac * span), maxLineGap=6)
    if L is None:
        return None
    L = np.asarray(L).reshape(-1, 4).astype(float)
    mid = np.stack([(L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2], 1)
    dl = np.stack([L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]], 1)
    length = np.linalg.norm(dl, axis=1)
    vert = vertical_dir_at(mid, g, focal_px, w / 2, h / 2)
    cosang = np.abs((dl * vert).sum(1)) / (length + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))        # 0 = plumb, 90 = level

    def wmedian(v, wt):
        o = np.argsort(v); c = np.cumsum(wt[o])
        return float(v[o][np.searchsorted(c, c[-1] / 2)])

    out = {"segments": [dict(xyxy=list(map(float, s)), deg_from_vertical=float(a), length_px=float(l))
                        for s, a, l in zip(L, ang, length)]}
    pv, ph = ang < 45, ang >= 45
    if pv.any():
        out["posts_lean_from_vertical_deg"] = wmedian(ang[pv], length[pv])
        out["posts_total_length_px"] = float(length[pv].sum())
    if ph.any():
        out["members_tilt_from_horizontal_deg"] = wmedian(90 - ang[ph], length[ph])
        out["members_total_length_px"] = float(length[ph].sum())
    out["camera_pitch_deg"], out["camera_roll_deg"] = camera_tilt(g)
    return out
