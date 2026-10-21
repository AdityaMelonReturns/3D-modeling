"""The three edge types from the one-pager. Each returns masks on the
ORIGINAL photo (warped to the straight-on view later in metrics.py).

  crisp   broken drywall, missing shingle, hole opening, rot cavity
          -> SAM2 mask as is (minus the ruler, holes filled, specks dropped)
  thin    cracks, gaps
          -> inside the box: Frangi ridge filter + threshold + skeleton for
             length; width from the dark band around that skeleton
  diffuse stains, moss, mold, paint loss
          -> colour difference from the clean surface around the box,
             reported as a range: tight (clearly different) / loose
"""
import numpy as np
import cv2
from scipy import ndimage
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize


def _box_slices(box, shape, pad=0):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = shape[:2]
    return slice(max(0, y0 - pad), min(h, y1 + pad)), slice(max(0, x0 - pad), min(w, x1 + pad))


def _keep_major_components(m, frac=0.10):
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m.astype(np.int32), lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= frac * sizes.max()) + 1
    return np.isin(lab, keep)


def crisp(sam_mask, card_m=None, box=None):
    m = sam_mask.copy()
    if card_m is not None:
        m &= ~card_m
    if box is not None:      # SAM2 occasionally leaks far outside the prompt box
        inside = np.zeros_like(m)
        inside[_box_slices(box, m.shape, pad=10)] = True
        m &= inside
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    m = ndimage.binary_fill_holes(_keep_major_components(m))
    return {"mask": m}


def _line_kernel(length, theta):
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2
    dx, dy = np.cos(theta) * c, np.sin(theta) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def linear_ridges(ridge, length, n_angles=36):
    """Keep only ridge response that continues in a straight line for
    `length` px: max over angles of a morphological opening with a line.
    Shingle granules / wood grain are short and wiggly; a crack or gap is not."""
    r = ridge.astype(np.float32)
    out = np.zeros_like(r)
    for th in np.linspace(0, np.pi, n_angles, endpoint=False):
        out = np.maximum(out, cv2.morphologyEx(r, cv2.MORPH_OPEN, _line_kernel(length, th)))
    return out


def thin(img_rgb, box, card_m=None, pos_points=(), sigmas=(2, 3, 4, 5, 6), line_frac=0.12):
    """pos_points: optional clicks ON the crack - pick which line is the
    defect when the box also holds tab joints / board edges."""
    ys, xs = _box_slices(box, img_rgb.shape)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)[ys, xs].astype(float) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)           # suppress shingle-granule noise
    ridge = frangi(gray, sigmas=sigmas, black_ridges=True)
    ridge /= np.percentile(ridge, 99.5) + 1e-9
    if card_m is not None:
        ridge[card_m[ys, xs]] = 0
    diag = np.hypot(*ridge.shape)
    lin = linear_ridges(ridge, max(21, int(line_frac * diag)) | 1)
    lin[:3, :], lin[-3:, :], lin[:, :3], lin[:, -3:] = 0, 0, 0, 0
    vals = lin[lin > 0.02]
    if vals.size == 0:
        return None
    band = lin > max(0.15, np.percentile(vals, 85))
    band = remove_small_objects(band, max(20, int(0.0005 * band.size)))
    skel = skeletonize(band)
    lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
    if n == 0:
        return None
    if len(pos_points):
        # keep pieces passing within 20 px of a click
        near = np.zeros_like(skel)
        for x, y in pos_points:
            cv2.circle(near.view(np.uint8), (int(x - xs.start), int(y - ys.start)), 20, 1, -1)
        keep = np.unique(lab[near & skel])
        keep = keep[keep > 0]
    else:
        lens = ndimage.sum(skel.astype(np.int32), lab, range(1, n + 1))
        keep = np.flatnonzero(lens >= 0.35 * lens.max()) + 1
    skel = np.isin(lab, keep)
    if not skel.any():
        return None

    # width: pixels clearly darker than the surface right next to the line
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)[ys, xs].astype(float)
    r = int(max(sigmas) * 2)
    near = cv2.dilate(skel.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)).astype(bool)
    ring = cv2.dilate(near.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~near
    bg = np.median(g[ring]) if ring.any() else np.median(g)
    core = np.percentile(g[skel], 20)
    dark = (g < (bg + core) / 2) & near
    dark = ndimage.binary_closing(dark, iterations=1)
    lab_d, _ = ndimage.label(dark)
    touching = np.unique(lab_d[skel & dark])
    dark = np.isin(lab_d, touching[touching > 0])

    full = lambda a: _paste(a, img_rgb.shape, ys, xs)
    return {"mask": full(dark | skel), "skeleton": full(skel), "ridge_band": full(band)}


def diffuse(img_rgb, box, card_m=None, tight=3.0, loose=2.0):
    ys, xs = _box_slices(box, img_rgb.shape)
    bw, bh = xs.stop - xs.start, ys.stop - ys.start
    pad = max(20, int(0.25 * max(bw, bh)))
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(float)
    outer = np.zeros(img_rgb.shape[:2], bool)
    outer[_box_slices(box, img_rgb.shape, pad=pad)] = True
    inner = np.zeros_like(outer)
    inner[ys, xs] = True
    ring = outer & ~inner
    if card_m is not None:
        ring &= ~card_m
    # the ring holds the clean surface but also grass, deck gaps, shadow...
    # -> cluster its colours and take the biggest cluster as "clean surface"
    ref = lab_img[ring]
    rng = np.random.default_rng(0)
    sample = ref[rng.choice(len(ref), min(20000, len(ref)), replace=False)].astype(np.float32)
    _, lbl, ctr = cv2.kmeans(sample, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                             3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    clean = sample[lbl == np.bincount(lbl).argmax()]
    med = np.median(clean, 0)
    std = 1.4826 * np.median(np.abs(clean - med), 0) + 2.0        # per-channel robust spread
    d = np.sqrt((((lab_img - med) / std) ** 2).sum(2) / 3)
    out = {}
    for name, k in (("tight", tight), ("loose", loose)):
        m = (d > k) & inner
        if card_m is not None:
            m &= ~card_m
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
        out[name] = remove_small_objects(m, max(50, int(0.002 * inner.sum())))
    out["mask"] = out["tight"]
    out["reference_ring_px"] = int(ring.sum())
    return out


def _paste(crop, shape, ys, xs):
    m = np.zeros(shape[:2], bool)
    m[ys, xs] = crop
    return m
