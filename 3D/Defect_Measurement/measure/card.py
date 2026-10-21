"""Scale + rectify: find the ruler card, reject false ones, and build the
mapping from photo pixels to a straight-on, fixed cm/px view.

Why each check exists (all seen on the 82 test photos):
  - colour alone finds false "rulers": blue-ish shadows on the roof, frosted
    window glass in hole, a blue wagon in invlined_railing, sky patches.
    -> shape checks: long thin (aspect), solid (fill / solidity).
  - a candidate can be the right shape but the wrong size: implied
    camera-to-card distance from the EXIF focal length must be sane.
  - one threshold set misses rulers in shade or glare -> try a strict and a
    loose colour threshold, keep the best plausible candidate.

The ruler is ~8x longer than wide, so the 4-corner homography from it pins
down perspective ALONG the ruler well but ACROSS it weakly. We therefore
always compute both the homography and a plain affine (rotate + scale)
mapping, use the homography only when it doesn't distort the defect region
implausibly, and report how much the two disagree as part of the scale
uncertainty. A square card (credit card / A5 sheet) would remove this
weakness - see README.
"""
import numpy as np
import cv2

from config import CARD_LENGTH_CM, CARD_WIDTH_CM

# (min saturation, min value) - strict first, then loose for shaded rulers
THRESHOLDS = [(60, 90), (35, 70)]
HUE_RANGE_DEG = (185, 225)
ASPECT_RANGE = (5.0, 12.0)          # true ~7.9; foreshortening moves it either way
MIN_FILL, MIN_SOLIDITY = 0.70, 0.75
MAX_END_WIDTH_RATIO = 1.6          # beyond this the outline is cut short by glare
                                    # (peeling_paint_08 stops at the 9 cm tick) -> length unusable
DISTANCE_RANGE_CM = (12, 400)       # implied camera-to-card distance
MIN_LENGTH_PX = 200                 # < ~12 px/cm: too coarse to scale from (also
                                    # rejects a ruler-shaped blue paint strip in railing_02)


def _blue_mask(img_rgb, s_min, v_min):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int) * 2, hsv[..., 1], hsv[..., 2]
    m = ((h >= HUE_RANGE_DEG[0]) & (h <= HUE_RANGE_DEG[1]) & (s >= s_min) & (v >= v_min)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def _fit_side_lines(pts, center, axis, normal, half_len):
    """Fit a straight line to each long edge of the ruler, using contour points
    from the middle 60% of its length (the bear's overalls at one end stick
    out sideways and faked a 1.4x perspective taper on roof_03)."""
    rel = pts - center
    along, across = rel @ axis, rel @ normal
    keep = np.abs(along) < 0.6 * half_len
    lines = []
    for side in (across > 0, across < 0):
        sel = pts[keep & side]
        if len(sel) < 10:
            return None
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    return lines


def _intersect(p, d, q, e):
    A = np.array([d, -e]).T
    t = np.linalg.solve(A, q - p)
    return p + t[0] * d


def _candidate(contour, focal_px):
    area = cv2.contourArea(contour)
    (cx, cy), (w, h), ang = cv2.minAreaRect(contour)
    L, W = max(w, h), min(w, h)
    if W < 3:
        return None
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    c = dict(aspect=L / W, fill=area / (w * h), solidity=area / max(hull_area, 1),
             length_px=L, width_px=W, center=[cx, cy])
    c["implied_distance_cm"] = focal_px * CARD_LENGTH_CM / L if focal_px else None

    reasons = []
    if not ASPECT_RANGE[0] <= c["aspect"] <= ASPECT_RANGE[1]:
        reasons.append(f"aspect {c['aspect']:.1f} not ruler-like")
    if c["fill"] < MIN_FILL:
        reasons.append(f"fill {c['fill']:.2f} too ragged")
    if c["solidity"] < MIN_SOLIDITY:
        reasons.append(f"solidity {c['solidity']:.2f} too ragged")
    if L < MIN_LENGTH_PX:
        reasons.append(f"only {L:.0f} px long - too small to scale from")
    dist = c["implied_distance_cm"]
    if dist is not None and not DISTANCE_RANGE_CM[0] <= dist <= DISTANCE_RANGE_CM[1]:
        reasons.append(f"implies camera {dist:.0f} cm away - implausible ruler size")
    c["plausible"], c["reject_reasons"] = not reasons, reasons

    # geometry: long axis, then trapezoid from the two fitted long edges
    rect_pts = cv2.boxPoints(((cx, cy), (w, h), ang))
    axis = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    if h > w:
        axis = np.array([-axis[1], axis[0]])
    normal = np.array([-axis[1], axis[0]])
    center = np.array([cx, cy])
    pts = contour.reshape(-1, 2).astype(float)
    proj = (pts - center) @ axis
    end_a, end_b = center + proj.min() * axis, center + proj.max() * axis
    # rectangle corners ordered: end_a/+n, end_b/+n, end_b/-n, end_a/-n
    c["rect_quad"] = [list(end_a + normal * W / 2), list(end_b + normal * W / 2),
                      list(end_b - normal * W / 2), list(end_a - normal * W / 2)]
    lines = _fit_side_lines(pts, center, axis, normal, L / 2)
    if lines:
        (p1, d1), (p2, d2) = lines
        quad = [_intersect(p1, d1, end_a, normal), _intersect(p1, d1, end_b, normal),
                _intersect(p2, d2, end_b, normal), _intersect(p2, d2, end_a, normal)]
        c["quad"] = [list(map(float, q)) for q in quad]
        wa = np.linalg.norm(quad[0] - quad[3])
        wb = np.linalg.norm(quad[1] - quad[2])
        c["end_width_ratio"] = float(max(wa, wb) / max(min(wa, wb), 1e-6))
    else:
        c["quad"], c["end_width_ratio"] = None, None
    if c["end_width_ratio"] and c["end_width_ratio"] > MAX_END_WIDTH_RATIO:
        c["reject_reasons"].append(f"ends differ {c['end_width_ratio']:.1f}x in width - outline cut short (glare/occlusion)")
        c["plausible"] = False
    c["contour"] = contour
    c["rect_quad"] = [list(map(float, q)) for q in c["rect_quad"]]
    return c


def detect_card(img_rgb, focal_px=None, min_area=1500):
    """Returns (best plausible candidate or None, list of all candidates)."""
    cands = []
    for s_min, v_min in THRESHOLDS:
        m = _blue_mask(img_rgb, s_min, v_min)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            c = _candidate(cnt, focal_px)
            if c:
                c["threshold"] = [s_min, v_min]
                cands.append(c)
    good = [c for c in cands if c["plausible"]]
    # closer to the true aspect and more solid wins; size breaks ties
    best = max(good, key=lambda c: (c["solidity"] + c["fill"] - abs(np.log(c["aspect"] / 7.9)),
                                    c["length_px"]), default=None)
    return best, cands


def card_mask(shape, card, grow_px=6):
    m = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(m, [card["contour"]], -1, 1, -1)
    if grow_px:
        m = cv2.dilate(m, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8))
    return m.astype(bool)


def card_neg_points(card, n=5):
    """Points along the ruler's centre line - fed to SAM2 as 'not this' clicks
    so the ruler never gets absorbed into the defect mask (it did in the old
    cached masks for roof 02/05/06 and gaps 05/06/07)."""
    q = np.array(card["rect_quad"])
    a, b = (q[0] + q[3]) / 2, (q[1] + q[2]) / 2
    return [list(map(float, a + (b - a) * t)) for t in np.linspace(0.15, 0.85, n)]


def _homography(src_quad, px_per_cm):
    dst = np.array([[0, 0], [CARD_LENGTH_CM, 0], [CARD_LENGTH_CM, CARD_WIDTH_CM], [0, CARD_WIDTH_CM]]) * px_per_cm
    return cv2.getPerspectiveTransform(np.float32(src_quad), np.float32(dst))


def _local_scale(H, pt):
    """sqrt of the area magnification of H at pt (rectified px per image px)."""
    eps = 1.0
    p = np.float32([[pt, [pt[0] + eps, pt[1]], [pt[0], pt[1] + eps]]])
    q = cv2.perspectiveTransform(p, H)[0]
    J = np.array([q[1] - q[0], q[2] - q[0]]).T / eps
    return float(np.sqrt(abs(np.linalg.det(J)))), float(np.linalg.det(J))


def build_rectifier(card, defect_xy):
    """Pick the image->rectified mapping for this photo.

    Rectified view has a FIXED scale of px_per_cm, chosen to roughly match the
    photo's own resolution at the card (so warping neither throws detail away
    nor invents it). Returns dict with H, px_per_cm, method, and the
    homography-vs-affine scale disagreement at the defect (an uncertainty)."""
    px_per_cm = card["length_px"] / CARD_LENGTH_CM
    H_aff = _homography(card["rect_quad"], px_per_cm)
    out = dict(px_per_cm=px_per_cm, H=H_aff, H_affine=H_aff, method="affine (card rectangle)",
               scale_disagreement_pct=None)
    if card.get("quad") is None:
        return out
    H_hom = _homography(card["quad"], px_per_cm)
    s_hom, det_hom = _local_scale(H_hom, defect_xy)
    _, det_card = _local_scale(H_hom, card["center"])
    s_aff, _ = _local_scale(H_aff, defect_xy)
    ratio = s_hom / s_aff
    out["scale_disagreement_pct"] = float(abs(ratio - 1) * 100)
    out["H_homography"] = H_hom
    # use the homography unless it folds (horizon line between card and
    # defect -> determinant changes sign) or blows up at the defect: a thin
    # ruler's perspective can't be trusted extrapolated far from it. A plain
    # mirror (corner order) is harmless and has the same sign at both places.
    if det_hom * det_card > 0 and 0.67 < ratio < 1.5:
        out["H"], out["method"] = H_hom, "homography (card trapezoid)"
    return out


def tick_check(img_rgb, card, up=2):
    """Cross-check the card from its printed cm ticks: autocorrelation of the
    tick strip, searched only in the period window a 16-23 cm body allows
    (that window excludes the half-cm and 2-cm harmonics). Returns the
    tick-implied body length; a strong, periodic signal that disagrees with
    CARD_LENGTH_CM means the outline is wrong. Weak signal = no verdict
    (most photos: ticks too small / blurred to read reliably)."""
    q = np.float32(card["rect_quad"])
    L, W = int(card["length_px"] * up), int(card["width_px"] * up)
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [L, 0], [L, W], [0, W]]))
    g = cv2.cvtColor(cv2.warpPerspective(img_rgb, M, (L, W), flags=cv2.INTER_CUBIC), cv2.COLOR_RGB2GRAY).astype(float)
    lo_p, hi_p = L / 23.0, L / 16.0
    best = None
    for gg in (g, g[::-1]):
        for a, b in ((0.04, 0.20), (0.10, 0.30), (0.20, 0.40)):
            prof = gg[int(a * W):int(b * W)].mean(0)
            k = max(3, int(lo_p / 2)) | 1
            prof = (prof - np.convolve(prof, np.ones(k) / k, "same"))[k:-k]
            prof -= prof.mean()
            ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
            ac /= ac[0] + 1e-9
            i0, i1 = int(lo_p), int(hi_p) + 1
            if i1 + 1 >= len(ac):
                continue
            j = i0 + int(np.argmax(ac[i0:i1]))
            if j <= i0 or j >= i1 - 1:
                continue
            y0, y1, y2 = ac[j - 1], ac[j], ac[j + 1]
            jf = j + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
            harm = np.mean([ac[int(round(m * jf))] for m in (2, 3) if int(round(m * jf)) < len(ac)])
            if best is None or y1 + harm > best[1]:
                best = (jf / up, float(y1 + harm))
    if best is None:
        return {"verdict": "ticks not readable"}
    per, score = best
    body = card["length_px"] / per
    out = {"px_per_cm_from_ticks": round(per, 2), "body_cm_implied": round(body, 2), "periodicity": round(score, 2)}
    if score < 0.6:
        out["verdict"] = "ticks too faint to cross-check"
    elif abs(body / CARD_LENGTH_CM - 1) > 0.10:
        out["verdict"] = "DISAGREES with card outline - outline likely cut short or merged"
    else:
        out["verdict"] = "agrees with card outline"
    return out
