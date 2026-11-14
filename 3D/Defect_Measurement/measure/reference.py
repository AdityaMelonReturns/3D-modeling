"""Scale from objects of known size, for photos with no usable ruler
(Task A of the Reference-Object Scale brief).

Prompt (prompts_reference/<defect>.json), one per photo, hand-drawn for now:
    "07": {"object": "deck_board", "box": [x0, y0, x1, y1], "same_surface": true}
  object        key into reference_sizes.REFERENCE_SIZES
  box           an area showing the pattern (or, for a single object, both of
                its edges)
  same_surface  the person drawing the box says whether the reference lies on
                the defect's surface. A brick wall behind a rotten deck board
                does not scale the board -> false lowers confidence to low.
  size_cm       optional, overrides the standard size (lap siding)

How (mirrors card.tick_check(), which reads the ruler's cm ticks):
  1. Direction: the structure tensor of strong edges in the box gives the
     direction the repeats run in; we measure across it.
  2. Profile: sample a band of the photo along that cross direction, through
     the point we want scale at (the defect, or the ruler in the acceptance
     test), so perspective along the band is handled where it matters.
  3. repeating: autocorrelation finds the spacing, then every repeat is
     located and a smooth curve (position vs repeat number) is fitted, so the
     local spacing AT the query point is used - not the average.
     single: the two strongest parallel edges, their distance apart.
  4. Scale is known along one direction only (across the repeats). Along the
     repeats it is assumed equal - flagged in the output.
"""
import numpy as np
import cv2
from scipy.signal import find_peaks

from measure.reference_sizes import REFERENCE_SIZES

ONE_DIRECTION_PCT = 10.0   # extra uncertainty: scale along the repeats is assumed, not measured


def _dominant_normal(gray, box, exclude=None):
    """Unit vector ACROSS the pattern lines, plus how consistent the lines are.
    From LONG straight lines (Hough, length-weighted) so surface texture -
    shingle granules, wood grain - can't decide it; structure tensor only as
    a fallback when no long lines are found."""
    x0, y0, x1, y1 = box
    g8 = np.clip(gray[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    e = cv2.Canny(cv2.GaussianBlur(g8, (0, 0), 2.0), 30, 90)
    if exclude is not None:
        e[exclude[y0:y1, x0:x1]] = 0
    L = cv2.HoughLinesP(e, 1, np.pi / 360, 60, minLineLength=int(0.2 * min(g8.shape)), maxLineGap=8)
    if L is not None and len(L) >= 3:
        L = np.asarray(L).reshape(-1, 4).astype(float)
        dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
        w = np.hypot(dx, dy)
        ang2 = 2 * np.arctan2(dy, dx)
        c, s = (w * np.cos(ang2)).sum(), (w * np.sin(ang2)).sum()
        coherence = float(np.hypot(c, s) / w.sum())
        th = 0.5 * np.arctan2(s, c) + np.pi / 2          # line direction -> normal
        return np.array([np.cos(th), np.sin(th)]), coherence
    g = gray[y0:y1, x0:x1]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.hypot(gx, gy)
    sel = mag > np.percentile(mag, 80)
    if exclude is not None:
        sel &= ~exclude[y0:y1, x0:x1]
    # double-angle averaging: opposite gradients (both sides of a dark gap) agree
    ang2 = 2 * np.arctan2(gy[sel], gx[sel])
    w = mag[sel]
    c, s = (w * np.cos(ang2)).sum(), (w * np.sin(ang2)).sum()
    coherence = float(np.hypot(c, s) / (w.sum() + 1e-9))
    th = 0.5 * np.arctan2(s, c)
    return np.array([np.cos(th), np.sin(th)]), coherence


def _profile(gray, box, q, n, exclude=None, band_frac=0.15):
    """Mean brightness along the line through q in direction n, averaged over a
    band +-band_frac of the box size along the pattern lines. NaN where the
    band leaves the box or hits an excluded pixel (the ruler)."""
    x0, y0, x1, y1 = box
    t = np.array([-n[1], n[0]])
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float) - q
    s_lo, s_hi = (corners @ n).min(), (corners @ n).max()
    u_lo, u_hi = (corners @ t).min(), (corners @ t).max()
    half = band_frac * max(x1 - x0, y1 - y0)
    u_c = np.clip(0.0, u_lo + half, u_hi - half) if u_hi - u_lo > 2 * half else (u_lo + u_hi) / 2
    s = np.arange(np.floor(s_lo), np.ceil(s_hi) + 1)
    u = np.linspace(u_c - half, u_c + half, 41)
    S, U = np.meshgrid(s, u)
    X = (q[0] + S * n[0] + U * t[0]).astype(np.float32)
    Y = (q[1] + S * n[1] + U * t[1]).astype(np.float32)
    vals = cv2.remap(gray, X, Y, cv2.INTER_LINEAR, borderValue=np.nan)
    inside = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
    if exclude is not None:
        ex = cv2.remap(exclude.astype(np.uint8), X, Y, cv2.INTER_NEAREST, borderValue=1).astype(bool)
        inside &= ~ex
    vals[~inside] = np.nan
    with np.errstate(all="ignore"):
        prof = np.nanmean(vals, axis=0)
        cover = np.mean(inside, axis=0)
    prof[cover < 0.3] = np.nan
    return s, prof, float(abs(u_c)) > 1.0


def _fit_repeats(s, prof, min_period=10):
    """Repeat spacing from a brightness profile. Returns dict or None."""
    ok = ~np.isnan(prof)
    if ok.sum() < 4 * min_period:
        return None
    p = prof.copy()
    p[~ok] = np.nanmean(prof)
    min_period = max(min_period, int(0.03 * ok.sum()))
    # smooth away texture finer than a repeat (granules, grain), then remove the
    # slow light fall-off; gaps / shadow lines become peaks
    sm = cv2.GaussianBlur(p.reshape(1, -1).astype(np.float32), (0, 0), min_period / 4).ravel()
    k = int(4 * min_period) | 1
    dark = -(sm - np.convolve(sm, np.ones(k) / k, "same"))
    near_gap = np.convolve((~ok).astype(float), np.ones(2 * min_period + 1), "same") > 0
    dark[near_gap] = 0             # filled-in stretches must not create peaks at their borders
    d = dark - dark[ok].mean()
    d[~ok] = 0
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac /= ac[0] + 1e-9
    hi = int(ok.sum() / 2.5)
    if hi <= min_period + 2:
        return None
    lags = np.arange(min_period, hi)
    seg = ac[min_period:hi]
    pk, _ = find_peaks(seg)
    if len(pk) == 0:
        return None
    best = seg[pk].max()
    # the fundamental: the shortest-lag peak nearly as strong as the best one
    fund = pk[np.argmax(seg[pk] >= 0.8 * best)]
    period, strength = float(lags[fund]), float(seg[fund])
    if strength < 0.15:
        return {"strength": strength, "reason": f"no clear repeat (periodicity {strength:.2f})"}

    peaks, props = find_peaks(dark, distance=max(3, int(0.6 * period)), prominence=np.std(dark[ok]) * 0.5)
    peaks = peaks[ok[peaks]]
    if len(peaks) < 3:
        return {"strength": strength, "reason": "fewer than 3 repeats located"}
    x = s[peaks].astype(float)
    # number the repeats using the local spacing, so a missed one shows up as a gap of 2
    idx = [0]
    for a, b in zip(x[:-1], x[1:]):
        idx.append(idx[-1] + max(1, int(round((b - a) / period))))
    nidx = np.array(idx, float)
    keep = np.ones(len(x), bool)
    for _ in range(3):
        deg = 2 if keep.sum() >= 5 else 1
        coef = np.polyfit(nidx[keep], x[keep], deg)
        res = x - np.polyval(coef, nidx)
        new = np.abs(res) < 0.2 * period
        if (new == keep).all() or new.sum() < 3:
            break
        keep = new
    deg = 2 if keep.sum() >= 5 else 1
    coef = np.polyfit(nidx[keep], x[keep], deg)
    resid = float(np.sqrt(np.mean((x[keep] - np.polyval(coef, nidx[keep])) ** 2)))
    # repeat number at the query point (s = 0) and the local spacing there
    n_grid = np.linspace(nidx[keep].min() - 2, nidx[keep].max() + 2, 2001)
    n0 = n_grid[np.argmin(np.abs(np.polyval(coef, n_grid)))]
    local = float(abs(np.polyval(np.polyder(coef), n0)))
    span = (x[keep].min(), x[keep].max())
    return {"period_px_local": local, "period_px_mean": period, "strength": strength,
            "n_repeats": int(nidx[keep].max() - nidx[keep].min()), "fit_resid_px": resid,
            "query_inside": bool(span[0] - period <= 0 <= span[1] + period),
            "peaks_s": x[keep].tolist()}


def _fit_single(s, prof):
    """Distance between the two strongest edges on either side of the query."""
    ok = ~np.isnan(prof)
    if ok.sum() < 30:
        return None
    p = prof.copy()
    p[~ok] = np.nanmean(prof)
    g = np.abs(np.gradient(cv2.GaussianBlur(p.reshape(1, -1).astype(np.float32), (0, 0), 2).ravel()))
    near_gap = np.convolve((~ok).astype(float), np.ones(11), "same") > 0
    g[near_gap] = 0                # no fake edges at the box border or the masked ruler
    floor = np.median(g[ok]) + 1e-6
    # the query sits ON the object, so its edges are the NEAREST strong edges
    # on each side - not the strongest edge anywhere (that is often background)
    pk, _ = find_peaks(g, distance=5)
    strong = pk[g[pk] > max(4 * floor, 0.35 * g.max())]
    left, right = strong[s[strong] < 0], strong[s[strong] > 0]
    if len(left) == 0 or len(right) == 0:
        return None
    ia, ib = left[np.argmax(s[left])], right[np.argmin(s[right])]
    a, b = s[ia], s[ib]
    gl, gr = g[ia], g[ib]
    return {"width_px": float(b - a), "edges_s": [float(a), float(b)],
            "edge_strength": float(min(gl, gr) / floor)}


def _snap_line(gray, p1, p2, search=12):
    """Slide a clicked edge line sideways (+-search px) to where the image
    gradient across it is strongest - hand clicks are only ~10 px accurate."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = (p2 - p1) / np.linalg.norm(p2 - p1)
    nrm = np.array([-d[1], d[0]])
    ts = np.linspace(0.1, 0.9, 60)
    best, best_off = -1, 0.0
    for off in np.arange(-search, search + 0.5, 0.5):
        pts = p1 + np.outer(ts, p2 - p1) + off * nrm
        a = cv2.remap(gray, (pts[:, 0] - nrm[0] * 2).astype(np.float32).reshape(1, -1),
                      (pts[:, 1] - nrm[1] * 2).astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)
        b = cv2.remap(gray, (pts[:, 0] + nrm[0] * 2).astype(np.float32).reshape(1, -1),
                      (pts[:, 1] + nrm[1] * 2).astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)
        g = float(np.abs(np.nanmedian(a - b)))
        if g > best:
            best, best_off = g, off
    return p1 + best_off * nrm, p2 + best_off * nrm, best_off


def _width_from_edges(gray, edges, q):
    """edges = [[a1, a2], [b1, b2]]: two clicked points on each long edge of
    the object. Width = distance between the two (snapped) edge lines,
    measured through the query point, across their mean direction."""
    (a1, a2), (b1, b2) = [_snap_line(gray, *e)[:2] for e in edges]
    da, db = (a2 - a1) / np.linalg.norm(a2 - a1), (b2 - b1) / np.linalg.norm(b2 - b1)
    if da @ db < 0:
        db = -db
    t = (da + db) / np.linalg.norm(da + db)
    n = np.array([-t[1], t[0]])

    def hit(p, d):          # where the line q + s*n crosses edge line p + u*d
        A = np.array([n, -d]).T
        return np.linalg.solve(A, p - q)[0]
    sa, sb = hit(a1, da), hit(b1, db)
    angle = float(np.degrees(np.arccos(np.clip(abs(da @ db), 0, 1))))
    return abs(sb - sa), n, [min(sa, sb), max(sa, sb)], angle


def reference_scale(img_rgb, prompt, query_xy, exclude=None):
    """px/cm at query_xy from the prompted reference object.
    Returns a dict ready for the JSON scale block (found=False with a reason
    when nothing usable was found - those are kept and reported)."""
    obj = prompt["object"]
    spec = REFERENCE_SIZES[obj]
    size_cm = prompt.get("size_cm") or spec["size_cm"]
    out = {"object": obj, "found": False, "reference_size_cm": size_cm,
           "reference_size_source": spec["source"], "same_surface": prompt.get("same_surface")}
    if size_cm is None:
        out["reason"] = "no standard size for this object - the job must record it"
        return out
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in prompt["box"]]
    box = [max(0, x0), max(0, y0), min(w, x1), min(h, y1)]
    gray = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.2)
    n, coherence = _dominant_normal(gray, box, exclude)
    out["pattern_direction_deg"] = round(float(np.degrees(np.arctan2(-n[0], n[1]))), 1)
    out["direction_coherence"] = round(coherence, 2)
    q = np.asarray(query_xy, float)
    s, prof, shifted = _profile(gray, box, q, n, exclude)
    out["profile_band_shifted"] = shifted      # band had to move to stay inside the box

    if spec["kind"] == "repeating":
        f = _fit_repeats(s, prof)
        if f is None or "period_px_local" not in f:
            out["reason"] = (f or {}).get("reason", "pattern too short in the box")
            return out
        px = f["period_px_local"]
        out.update(found=True, n_repeats=f["n_repeats"], periodicity=round(f["strength"], 2),
                   spacing_px=round(px, 2), spacing_px_mean=round(f["period_px_mean"], 1),
                   fit_resid_px=round(f["fit_resid_px"], 2), query_inside_pattern=f["query_inside"])
        fit_pct = 100 * f["fit_resid_px"] / px / np.sqrt(max(f["n_repeats"], 1))
        weak = f["n_repeats"] < 4 or f["strength"] < 0.3 or not f["query_inside"]
    elif prompt.get("edges"):
        px, n_e, marks, ang = _width_from_edges(gray, prompt["edges"], q)
        out.update(found=True, n_repeats=1, width_px=round(px, 1), edges_angle_deg=round(ang, 1),
                   method="manual edge clicks, snapped to the strongest nearby edge")
        out["profile_line"] = {"q": q.tolist(), "n": n_e.tolist(), "marks_s": marks}
        fit_pct = 100 * 2.0 / px
        weak = ang > 8           # edges far from parallel = strong perspective across the object
        f = {"edges_s": marks}
    else:
        f = _fit_single(s, prof)
        if f is None or f["width_px"] < 15:
            out["reason"] = "could not find both edges of the object"
            return out
        px = f["width_px"]
        out.update(found=True, n_repeats=1, width_px=round(px, 1), edge_strength=round(f["edge_strength"], 1))
        fit_pct = 100 * 2.0 / px           # ~1 px per edge
        weak = f["edge_strength"] < 3

    out["px_per_cm"] = px / size_cm
    out["scale_direction"] = "across the repeats only; assumed the same along them" \
        if spec["kind"] == "repeating" else "across the object only; assumed the same along it"
    out["uncertainty_pct"] = round(float(np.sqrt(spec["tolerance_pct"] ** 2 + fit_pct ** 2 + ONE_DIRECTION_PCT ** 2)), 1)
    # never high: medium at most, low when the reference may not be on the defect's surface
    notes = []
    if prompt.get("same_surface") is False:
        notes.append("reference is on a different surface from the defect - scale does not transfer")
    elif prompt.get("same_surface") is None:
        notes.append("not confirmed that the reference is on the defect's surface")
    if weak:
        notes.append("weak pattern fit (few repeats / weak periodicity / outside the pattern)")
    if coherence < 0.3:
        notes.append(f"pattern direction unclear (coherence {coherence:.2f})")
    out["confidence"] = "low" if notes else "medium"
    out["notes"] = notes
    if "profile_line" not in out:
        out["profile_line"] = {"q": q.tolist(), "n": n.tolist(),
                               "marks_s": f.get("peaks_s") or f.get("edges_s")}
    return out


def draw(o, ref):
    """Overlay: magenta ticks where each repeat / edge was found."""
    if not ref or not ref.get("found"):
        return o
    q, n = np.array(ref["profile_line"]["q"]), np.array(ref["profile_line"]["n"])
    t = np.array([-n[1], n[0]])
    marks = ref["profile_line"]["marks_s"]
    a, b = q + min(marks) * n, q + max(marks) * n
    cv2.line(o, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 255), 2)
    for m in marks:
        p = q + m * n
        cv2.line(o, tuple(map(int, p - 18 * t)), tuple(map(int, p + 18 * t)), (255, 0, 255), 3)
    return o
