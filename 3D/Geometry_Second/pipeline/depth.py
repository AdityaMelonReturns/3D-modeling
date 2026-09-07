"""Stage [C]: Monocular Metric Depth (Depth Pro) - the primary geometry source.

Confirmed on this machine (CPU): ~113s one-time model load, ~283s per image
inference. That's ~4.7 min/image - fine to validate correctness on one or two
images locally, impractical to run a full 8-defect batch without a GPU. Load
the model ONCE per process and reuse across images (don't reload per-image).
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
