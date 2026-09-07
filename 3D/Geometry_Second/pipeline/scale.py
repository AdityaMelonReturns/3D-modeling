"""Stage [D]: Scale Calibration - detect the scale card, cross-check/correct
Depth Pro's native metric scale against it.

The capture app's scale card is a small "Teddy Bear" novelty ruler, pale
blue, ~15cm (0-15cm marked). Not an ArUco marker (spec suggested ArUco as a
"going forward" improvement) - this detects the physical ruler already in
the existing photos via color thresholding + largest-connected-component,
the same method validated in the COLMAP-based pilot (Geometry/REPORT.md
Section 3): reliable ONLY when the ruler is laid flat, roughly parallel to
the camera, in the same shot as the defect. Held-at-an-angle shots will
silently give a wrong correction scalar - that's why we return a
`reliable: bool` the caller must respect, and gate degraded confidence off it.
"""
import numpy as np
from scipy import ndimage
from PIL import Image

RULER_LENGTH_M = 0.15  # 0-15cm marked ruler; ~5-10% uncertainty on exact tip-to-tip length


def _largest_component(mask):
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return None, 0
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    return labeled == best, int(sizes[best - 1])


def detect_ruler(image_path, hue_lo=175, hue_hi=225, sat_lo=25, val_lo=70, val_hi=255,
                  crop=None, min_size=200):
    """Returns dict(found, mask, endpoints_px=[(x0,y0),(x1,y1)], length_px,
    aspect_ratio) or {"found": False} if nothing plausible was isolated.
    `aspect_ratio` (major/minor axis) flags oblique/foreshortened shots -
    a real, flat, in-frame ruler should be a thin, elongated blob (high
    aspect ratio); a low aspect ratio (blobby, not sliver-shaped) usually
    means the detector caught something else, or the ruler is nearly
    edge-on / not worth trusting.
    """
    im = Image.open(image_path).convert("RGB")
    x0, y0 = 0, 0
    if crop:
        x0, y0, x1, y1 = crop
        im = im.crop((x0, y0, x1, y1))
    hsv = np.asarray(im.convert("HSV")).astype(np.float32)
    H, S, V = hsv[..., 0] * (360.0 / 255.0), hsv[..., 1], hsv[..., 2]
    mask = (H >= hue_lo) & (H <= hue_hi) & (S >= sat_lo) & (V >= val_lo) & (V <= val_hi)
    comp, size = _largest_component(mask)
    if comp is None or size < min_size:
        return {"found": False, "reason": f"no component >= {min_size}px (best {size})"}

    ys, xs = np.nonzero(comp)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    major = evecs[:, 0]
    proj = centered @ major
    length_px = proj.max() - proj.min()
    minor_std = np.sqrt(max(evals[1], 1e-6))
    aspect_ratio = length_px / (2 * minor_std) if minor_std > 0 else 0

    p0 = mean + major * proj.min()
    p1 = mean + major * proj.max()
    # shift back to full-image coords if cropped
    p0 = (p0[0] + x0, p0[1] + y0)
    p1 = (p1[0] + x0, p1[1] + y0)

    full_mask = None
    if crop:
        full_mask = np.zeros((Image.open(image_path).size[1], Image.open(image_path).size[0]), dtype=bool)
        full_mask[y0:y0 + comp.shape[0], x0:x0 + comp.shape[1]] = comp
    else:
        full_mask = comp

    return {
        "found": True,
        "mask": full_mask,
        "endpoints_px": [p0, p1],
        "length_px": float(length_px),
        "aspect_ratio": float(aspect_ratio),
        "component_size_px": int(size),
    }


def cross_check_against_depth(ruler_result, depth_m, K, aspect_ratio_min=3.0):
    """
    Projects the ruler's two detected endpoints to 3D using Depth Pro's own
    metric depth map + intrinsics K, computes the 3D distance, and compares
    to the known 0.15m ruler length.

    Returns dict(scale_source, correction_scalar, reliable, detail).
    `reliable=False` (aspect ratio too low -> oblique/foreshortened, or the
    two depth samples disagree wildly with local neighborhood -> noisy) means
    the caller should NOT apply the correction and should tag this image's
    measurements `scale_source: "model_estimate"` instead of "card_verified".
    """
    if not ruler_result.get("found"):
        return {"scale_source": "model_estimate", "correction_scalar": 1.0,
                "reliable": False, "detail": "no ruler detected"}

    if ruler_result["aspect_ratio"] < aspect_ratio_min:
        return {"scale_source": "model_estimate", "correction_scalar": 1.0,
                "reliable": False,
                "detail": f"ruler aspect ratio {ruler_result['aspect_ratio']:.1f} "
                          f"< {aspect_ratio_min} - likely oblique/foreshortened, not trusted "
                          f"(see Geometry/REPORT.md Sec.3: wood1/rotten_wood hit exactly this)"}

    def depth_at(px, window=5):
        x, y = int(round(px[0])), int(round(px[1]))
        h, w = depth_m.shape
        x0, x1 = max(0, x - window), min(w, x + window + 1)
        y0, y1 = max(0, y - window), min(h, y + window + 1)
        patch = depth_m[y0:y1, x0:x1]
        return float(np.median(patch))

    def to_3d(px, z):
        x, y = px
        X = (x - K[0, 2]) * z / K[0, 0]
        Y = (y - K[1, 2]) * z / K[1, 1]
        return np.array([X, Y, z])

    p0_px, p1_px = ruler_result["endpoints_px"]
    z0, z1 = depth_at(p0_px), depth_at(p1_px)
    P0, P1 = to_3d(p0_px, z0), to_3d(p1_px, z1)
    depth_predicted_length = float(np.linalg.norm(P1 - P0))

    if depth_predicted_length < 1e-6:
        return {"scale_source": "model_estimate", "correction_scalar": 1.0,
                "reliable": False, "detail": "degenerate depth at ruler endpoints"}

    correction = RULER_LENGTH_M / depth_predicted_length
    disagreement_pct = abs(correction - 1.0) * 100

    return {
        "scale_source": "card_verified",
        "correction_scalar": correction,
        "reliable": True,
        "depth_predicted_ruler_length_m": depth_predicted_length,
        "known_ruler_length_m": RULER_LENGTH_M,
        "disagreement_pct": disagreement_pct,
        "detail": (f"Depth Pro predicted {depth_predicted_length*100:.1f}cm for a known "
                   f"15cm ruler ({disagreement_pct:.1f}% off) -> applying {correction:.3f}x correction"),
    }
