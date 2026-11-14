"""Depth for rot and holes (Task B of the Reference-Object Scale brief).

Three pieces, in the order the brief asks for them:

1. dark_core()      the darkness classifier: inside the defect outline, which
                    pixels are much darker than the clean surface around it -
                    the likely core of deep damage. Reported as an AREA and a
                    fraction of the defect, never as a depth: rot is darker
                    than sound wood even when flat, and flash only changes
                    brightness ~12% over 2 cm at 30 cm.
2. features()       the hand-made features the depth-band classifier will be
                    trained on once probe labels exist (brightness ratio,
                    colour difference, texture, shape).
3. ruler_depth_from_side_photo()
                    reads real depth from the field protocol's side photo
                    (ruler pushed into the hole until it stops): the hidden
                    part of the ruler = depth. These become the labels.

The classifier itself lives in train_depth_band.py and refuses to train
below MIN_LABELS. Until then depth.band comes only from a probe or a ruler
photo; depth.source = "classifier" is never written by this module.
"""
import numpy as np
import cv2
from scipy import ndimage

from config import CARD_LENGTH_CM, CARD_WIDTH_CM, PROBE_BANDS

DEPTH_DEFECTS = ("hole", "rotten_wood", "wood1", "ceiling")   # rot / hole types
MIN_LABELS = 200


def _rim(mask, width_px):
    k = np.ones((2 * width_px + 1,) * 2, np.uint8)
    outer = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    gap = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return outer & ~gap


def dark_core(img_rgb, mask, exclude=None, rel=0.6):
    """Core of deep damage = defect pixels darker than rel (60%) of the clean
    surface's brightness in a thin ring just outside the outline. The ring is
    kept thin (8-30 px) so it samples the surface right at the defect, not the
    background; if even that ring is uneven it is flagged, not hidden."""
    L = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
    ys, xs = np.nonzero(mask)
    size = max(np.ptp(xs), np.ptp(ys), 20)
    rim = _rim(mask, int(np.clip(0.05 * size, 8, 30)))
    if exclude is not None:
        rim &= ~exclude
    if rim.sum() < 50 or mask.sum() < 30:
        return None
    rim_med = float(np.median(L[rim]))
    rim_spread = float(1.4826 * np.median(np.abs(L[rim] - rim_med)) / max(rim_med, 1e-6))
    thr = rel * rim_med
    core = mask & (L < thr)
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    lab, n = ndimage.label(core)
    if n:
        sizes = ndimage.sum(core.astype(np.int32), lab, range(1, n + 1))
        core = np.isin(lab, np.flatnonzero(sizes >= max(20, 0.02 * mask.sum())) + 1)
    inside = L[mask]
    return {
        "core": core, "rim": rim,
        "core_fraction": float(core.sum() / mask.sum()),
        "core_px": int(core.sum()),
        "inside_vs_rim_brightness": float(np.median(inside) / max(rim_med, 1e-6)),
        "core_vs_rim_brightness": float(np.median(L[core]) / max(rim_med, 1e-6)) if core.any() else None,
        "threshold_L": float(thr), "rim_median_L": rim_med, "rim_spread": round(rim_spread, 3),
        "rim_uneven": bool(rim_spread > 0.25),
    }


def features(img_rgb, mask, dc, diffuse_d=None):
    """Hand-made features for the baseline classifier (brief, Phase 2 step 1)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(float)
    rim = dc["rim"]
    # colour difference from the clean surface (same idea as edges.diffuse)
    ref = np.median(lab[rim], 0)
    dE = np.linalg.norm(lab[mask] - ref, axis=1)
    # texture inside: local variance and edge density (fibres, layers)
    mu = cv2.blur(gray, (7, 7))
    var = cv2.blur(gray ** 2, (7, 7)) - mu ** 2
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120) > 0
    # shape: size, solidity, how narrow the opening is
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = float(mask.sum())
    hull = cv2.contourArea(cv2.convexHull(c)) or 1.0
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return {
        "inside_vs_rim_brightness": dc["inside_vs_rim_brightness"],
        "core_vs_rim_brightness": dc["core_vs_rim_brightness"],
        "core_fraction": dc["core_fraction"],
        "colour_diff_median": float(np.median(dE)),
        "colour_diff_p90": float(np.percentile(dE, 90)),
        "texture_var_inside_vs_rim": float(np.median(var[mask]) / (np.median(var[rim]) + 1e-6)),
        "edge_density_inside": float(edges[mask].mean()),
        "edge_density_rim": float(edges[rim].mean()),
        "solidity": float(area / hull),
        "elongation": float(max(w, h) / max(min(w, h), 1.0)),
        "opening_narrowness": float(2 * dt.max() / max(max(w, h), 1.0)),   # max thickness / length
        "area_px": area,
        # flash on/off pair not captured yet - the protocol asks for it
    }


def band_from_cm(depth_cm):
    """Map a measured depth to the four probe bands in config.PROBE_BANDS."""
    if depth_cm is None:
        return None
    if depth_cm < 1:
        return PROBE_BANDS[0]
    if depth_cm <= 3:
        return PROBE_BANDS[1]
    return PROBE_BANDS[2]          # "through" is only ever set by the worker


def ruler_depth_from_side_photo(img_rgb, card, surface_xy, tick=None):
    """Field protocol: the ruler is pushed into the hole until it stops and
    photographed from the side. The part inside the hole is hidden, so

        depth = (blue body length) - (visible length from the surface to the outer end)

    surface_xy: a click where the surface meets the ruler (manual for now).
    Scale along the ruler comes from its printed cm ticks when readable
    (card.tick_check), else from its width (2.1 cm) - coarser, flagged.
    The outer end is the end of the detected ruler farther from surface_xy."""
    q = np.array(card["rect_quad"], float)
    a, b = (q[0] + q[3]) / 2, (q[1] + q[2]) / 2           # the two ends on the centre line
    s = np.asarray(surface_xy, float)
    outer = a if np.linalg.norm(a - s) > np.linalg.norm(b - s) else b
    axis = (outer - s) / (np.linalg.norm(outer - s) + 1e-9)
    visible_px = float((outer - s) @ axis)
    if tick and tick.get("verdict") == "agrees with card outline":
        ppc, src = tick["px_per_cm_from_ticks"], "printed cm ticks"
    else:
        # ruler width: coarse. Simulated test (hide the last 0.5-6 cm of 6 real
        # rulers) gave ~1 cm median error - too coarse for the 1 cm / 3 cm band
        # edges, so the protocol asks for a close-up where the ticks are readable
        ppc, src = card["width_px"] / CARD_WIDTH_CM, "ruler width (coarse, ~1 cm error - ticks not readable)"
    visible_cm = visible_px / ppc
    depth = CARD_LENGTH_CM - visible_cm
    return {"depth_cm": round(depth, 2), "visible_cm": round(visible_cm, 2), "scale_from": src,
            "confidence": "medium" if src == "printed cm ticks" else "low",
            "band": band_from_cm(max(depth, 0)), "source": "ruler_photo",
            "valid": bool(-0.5 <= depth <= CARD_LENGTH_CM - 2),
            "note": "depends on the 17.0 cm body length (tape check pending)"}


def draw(o, dc):
    if dc is None:
        return o
    core = dc["core"]
    o[core] = (0.35 * o[core] + [0, 0, 165]).astype(np.uint8)        # deep core: dark blue
    cnts, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cnts, -1, (0, 200, 255), 2)
    return o
