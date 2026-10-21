"""Outline: SAM2 mask from the hand-drawn box (+ optional positive/negative
clicks). Segmentation always runs on the ORIGINAL photo - the mask is warped
to the straight-on view afterwards, never the other way round.

Masks are cached per (photo, prompt) so re-running the measurement step is
instant; SAM2-large on this laptop's CPU takes ~30 s per photo.
"""
import hashlib
import json
import os

import numpy as np

from config import SAM2_CHECKPOINT, SAM2_CONFIG

_PREDICTOR = None


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _PREDICTOR = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device))
    return _PREDICTOR


def load_prompts(path):
    """prompts/<defect>.json: {"05": {"box": [x0,y0,x1,y1],
    "neg_points": [[x,y],...], "pos_points": [[x,y],...]}} - points optional.
    Same box format YOLO/GroundingDINO will produce later."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {k: {"box": v["box"], "neg_points": v.get("neg_points") or [],
                "pos_points": v.get("pos_points") or []} for k, v in raw.items()}


def segment(img_rgb, box, pos_points=(), neg_points=(), cache_path=None):
    """Returns (mask HxW bool, SAM2 score). Uses cache_path if the prompt matches."""
    prompt = {"box": [int(v) for v in box],
              "pos": [[round(float(a), 1) for a in p] for p in pos_points],
              "neg": [[round(float(a), 1) for a in p] for p in neg_points]}
    key = hashlib.md5(json.dumps(prompt, sort_keys=True).encode()).hexdigest()[:12]
    if cache_path and os.path.exists(cache_path + ".npz"):
        z = np.load(cache_path + ".npz")
        if str(z["key"]) == key:
            return z["mask"].astype(bool), float(z["score"])

    pred = _predictor()
    pred.set_image(img_rgb)
    pts = list(pos_points) + list(neg_points)
    labels = [1] * len(pos_points) + [0] * len(neg_points)
    masks, scores, _ = pred.predict(
        box=np.array(box, float),
        point_coords=np.array(pts, float) if pts else None,
        point_labels=np.array(labels) if pts else None,
        multimask_output=False,
    )
    mask, score = masks[0].astype(bool), float(scores[0])
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path + ".npz", mask=mask, score=score, key=key)
    return mask, score
