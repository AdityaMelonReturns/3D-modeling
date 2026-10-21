"""Measure: warp the mask into the straight-on, fixed cm/px view, then
count / trace it there.

  area       mask pixels x (cm/px)^2
  length     along the mask's skeleton (longest path), so a curved crack
             is measured along its curve, not end-to-end
  width      thickness = 2 x distance-to-edge, sampled on the skeleton
             (median = typical, p95 = widest)
  extent     min-area rectangle of the mask (overall L x W - "how big
             across" for a hole or tear)
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

MAX_CANVAS = 5000


def warp_mask(mask, H):
    """Warp a boolean mask with H onto a canvas just big enough to hold it."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((1, 1), bool), np.eye(3)
    corners = np.float32([[[xs.min(), ys.min()], [xs.max() + 1, ys.min()],
                           [xs.max() + 1, ys.max() + 1], [xs.min(), ys.max() + 1]]])
    wc = cv2.perspectiveTransform(corners, H)[0]
    x0, y0 = np.floor(wc.min(0)) - 2
    x1, y1 = np.ceil(wc.max(0)) + 2
    W, Hh = int(x1 - x0), int(y1 - y0)
    if W <= 0 or Hh <= 0 or W > MAX_CANVAS or Hh > MAX_CANVAS:
        raise ValueError(f"rectified canvas {W}x{Hh} implausible - rectification unstable")
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    out = cv2.warpPerspective(mask.astype(np.uint8), T @ H, (W, Hh), flags=cv2.INTER_NEAREST)
    return out.astype(bool), T @ H


def longest_skeleton_path(skel):
    """Length (px) of the longest geodesic path through a skeleton - the
    'along the crack' length. Two Dijkstra sweeps (farthest-from-farthest)."""
    ys, xs = np.nonzero(skel)
    n = len(xs)
    if n < 2:
        return float(n), None
    idx = -np.ones(skel.shape, int)
    idx[ys, xs] = np.arange(n)
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < skel.shape[0]) & (x2 >= 0) & (x2 < skel.shape[1])
        j = np.full(n, -1)
        j[ok] = idx[y2[ok], x2[ok]]
        sel = j >= 0
        rows += list(np.arange(n)[sel]); cols += list(j[sel])
        w += [np.hypot(dy, dx)] * int(sel.sum())
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G + G.T
    d0 = dijkstra(G, indices=0)
    d0[np.isinf(d0)] = -1
    a = int(np.argmax(d0))
    da, pred = dijkstra(G, indices=a, return_predecessors=True)
    da[np.isinf(da)] = -1
    b = int(np.argmax(da))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    pts = np.stack([xs[path], ys[path]], 1)
    return float(da[b]), pts


def _largest_skeleton_component(skel):
    lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
    if n <= 1:
        return skel
    sizes = ndimage.sum(skel.astype(np.int32), lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def measure_mask(mask_rect, px_per_cm, skeleton_rect=None, want=("area", "length_width")):
    """All numbers in cm / cm^2 from a mask already in the rectified view."""
    out = {}
    if mask_rect.sum() == 0:
        return out
    if "area" in want:
        out["area_cm2"] = float(mask_rect.sum() / px_per_cm ** 2)
    if "length_width" in want:
        cnts, _ = cv2.findContours(mask_rect.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        allpts = np.vstack(cnts)
        (_, _), (w, h), _ = cv2.minAreaRect(allpts)
        out["extent_cm"] = [float(max(w, h) / px_per_cm), float(min(w, h) / px_per_cm)]

        skel = skeleton_rect if skeleton_rect is not None else skeletonize(mask_rect)
        # pieces of a crack can be split by a twig etc.; length = sum of the
        # longest path through each substantial piece
        lab, n = ndimage.label(skel, structure=np.ones((3, 3)))
        total, pieces = 0.0, []
        sizes = ndimage.sum(skel.astype(np.int32), lab, range(1, n + 1)) if n else []
        for i in range(1, n + 1):
            if sizes[i - 1] < 0.1 * max(sizes):
                continue
            L, _ = longest_skeleton_path(lab == i)
            total += L
            pieces.append(L / px_per_cm)
        out["length_cm"] = float(total / px_per_cm)
        out["length_pieces_cm"] = [round(p, 2) for p in sorted(pieces, reverse=True)]

        dt = cv2.distanceTransform(mask_rect.astype(np.uint8), cv2.DIST_L2, 5)
        on = skel & mask_rect
        thick = 2 * dt[on] if on.any() else 2 * dt[mask_rect]
        out["width_cm"] = {"median": float(np.median(thick) / px_per_cm),
                           "max_p95": float(np.percentile(thick, 95) / px_per_cm)}
    return out


def measure_pixels(mask, skeleton=None, want=("area", "length_width")):
    """Same measurements in raw photo pixels (used when there is no scale)."""
    return measure_mask(mask, 1.0, skeleton, want)
