"""Local defect-crop depth cross-check (Depth Pro) - NOT the primary geometry
source anymore (see Geometry_third/PLAN.md). DA3 (geometry_da3.py) now owns
multi-view depth + camera pose. This module is kept only for stage [C'] - an
optional second opinion on the depth right around the defect, on whichever
single photo has the best SAM2 score, cropped tight. If it disagrees with
DA3 by a lot, that's a signal to lower confidence, not something to silently
average away.

Confirmed on CPU (Geometry_Second): ~113s one-time model load, ~283s per image
inference. Load the model ONCE per process and reuse across images.
"""
import time

import numpy as np
import torch
import depth_pro


class DepthEstimator:
    def __init__(self, device=None, verbose=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device, precision=torch.float32
        )
        self.model.eval()
        if verbose:
            print(f"[depth] Depth Pro loaded on {self.device} in {time.time()-t0:.1f}s")

    def infer(self, shot, verbose=True):
        """shot: a common.ShotData. Returns dict(depth_m: HxW np.float32,
        focaltxt_used_px: float).

        IMPORTANT: auto_rotate=False. Depth Pro's loader applies EXIF
        rotation by default, which would return a depth map in a DIFFERENT
        orientation (e.g. 1920x996 rotated to portrait) than every other
        stage in this pipeline (K matrix, SAM2 masks, LightGlue) - all of
        which operate on the raw, unrotated buffer (same convention COLMAP
        uses in the Geometry pipeline). Caught this exact mismatch as an
        IndexError during the first local dry-run on `ceiling` - keeping
        auto_rotate off is what makes depth_m align pixel-for-pixel with
        shot.K, the SAM2 mask, etc.
        """
        t0 = time.time()
        image, _, f_px = depth_pro.load_rgb(shot.image_path, auto_rotate=False)
        # Prefer our own EXIF-derived focal length for consistency with the
        # rest of the pipeline's intrinsics, unless Depth Pro's own EXIF
        # read disagrees wildly (then trust Depth Pro's, it parses real EXIF).
        f_px_final = f_px if f_px is not None else shot.focal_px
        image_t = self.transform(image)
        with torch.no_grad():
            prediction = self.model.infer(image_t, f_px=f_px_final)
        depth = prediction["depth"].detach().cpu().numpy().astype(np.float32)
        assert depth.shape == (shot.height, shot.width), (
            f"depth_m shape {depth.shape} != expected (H={shot.height}, W={shot.width}) "
            f"for {shot.name} - orientation mismatch between Depth Pro's output and the "
            f"raw-buffer convention (K matrix, SAM2 masks) used everywhere else in this "
            f"pipeline. Don't silently proceed - fix the mismatch instead of resizing/transposing."
        )
        if verbose:
            print(f"[depth] {shot.name}: inferred in {time.time()-t0:.1f}s, "
                  f"range {depth.min():.2f}-{depth.max():.2f} m, f_px={float(f_px_final):.1f}")
        return {"depth_m": depth, "focal_px_used": float(f_px_final)}

    def infer_crop(self, crop_image_path, f_px, verbose=True):
        """Same as infer(), but for a saved crop file (no shot object, no
        full-image shape assertion - the crop's dims are whatever they are).
        f_px must be passed explicitly: a crop we save ourselves has no real
        EXIF, and Depth Pro's own EXIF read would silently be wrong/absent -
        pass the ORIGINAL (uncropped) image's focal length in pixels, since
        cropping changes field of view, not focal length."""
        t0 = time.time()
        image, _, _ = depth_pro.load_rgb(crop_image_path, auto_rotate=False)
        image_t = self.transform(image)
        # depth_pro.infer() calls f_px.squeeze() internally - it expects a
        # tensor (that's what depth_pro.load_rgb's own f_px normally is), not
        # a plain python float. Hit this because we pass a hand-computed
        # float (DA3's own K[0,0]) instead of reading it back off the file.
        f_px_t = torch.tensor(float(f_px), device=self.device)
        with torch.no_grad():
            prediction = self.model.infer(image_t, f_px=f_px_t)
        depth = prediction["depth"].detach().cpu().numpy().astype(np.float32)
        if verbose:
            print(f"[depth] crop inferred in {time.time()-t0:.1f}s, "
                  f"range {depth.min():.2f}-{depth.max():.2f} m, f_px={f_px:.1f}")
        return depth
