"""Stage [B+C] combined: multi-view geometry via Depth Anything 3 (DA3).

Replaces Geometry_Second's pose.py (LightGlue matching + sensor-heuristic
translation) and depth.py-as-primary-source. DA3 takes every photo of one
defect TOGETHER and solves depth + camera position jointly - this is the
actual fix for the thing that was broken before: Geometry_Second's own
pose_confidence came back "low" on every single defect set we ran (LightGlue
found almost no strong matches on plain painted surfaces).

Checkpoint: DA3-BASE - Apache 2.0 (commercial-safe), same as the larger
non-commercial DA3 variants for camera pose + depth. NOT DA3METRIC-LARGE:
tested on the pod (2026-08-27) and confirmed DA3METRIC-LARGE returns
extrinsics=None, intrinsics=None no matter what - it's a depth-only
specialized checkpoint with no camera-pose head at all, useless for the one
thing we actually needed DA3 for. DA3-BASE gives real (N,3,4) extrinsics,
(N,3,3) intrinsics, and depth+conf, confirmed on `hole`'s 6 images.
This also means we get scale-ambiguous (not "metric") depth from DA3 now -
that's fine, we were always going to ruler-correct it ourselves regardless
(Depth Pro made the same "metric" claim and was 60-99% off every time we
checked it against the ruler, so DA3METRIC-LARGE's claim was never going to
be trusted blind either).

Coordinate convention (confirmed from DA3's own docs, not guessed):
`prediction.extrinsics` is world-to-camera, OpenCV/COLMAP format:
    X_camera = R @ X_world + t
We convert that to the camera-to-world form fusion.py already expects
(R_c2w, camera_center_world) via the standard inverse of a rigid transform.
STILL WORTH SANITY-CHECKING ON THE FIRST REAL RUN - see check_camera_layout().
"""
import cv2
import numpy as np
import torch


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor, possibly on GPU
        return x.detach().cpu().numpy()
    return np.asarray(x)


class DA3Geometry:
    def __init__(self, device=None, checkpoint="depth-anything/DA3-BASE"):
        from depth_anything_3.api import DepthAnything3
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(checkpoint)
        self.model = self.model.to(device=self.device)
        self.checkpoint = checkpoint

    def infer_multiview(self, image_paths, target_sizes=None):
        """image_paths: list[str], all photos of ONE defect, sent together so
        DA3 solves them jointly (that's the whole point - not one-at-a-time).

        target_sizes: optional list of (height, width), one per image - DA3
        processes images at its own internal resolution (504px on the long
        side by default), NOT the original photo resolution. Found this live
        on the pod: SAM2 masks and box prompts are all in original-resolution
        pixel coordinates, so a (266, 504) depth map indexed with original
        (e.g. 996, 1920) mask coordinates throws straight out of bounds.
        When target_sizes is given, depth/conf are upsampled and K is rescaled
        to match each shot's real resolution, so everything downstream (masks,
        backprojection) stays in one consistent coordinate system - the same
        convention Geometry_Second used throughout.

        Returns a list of per-shot dicts, same order as image_paths:
          depth_m: HxW float32              (DA3's own scale-ambiguous depth)
          conf: HxW float32                 (DA3's own per-pixel confidence)
          K: 3x3 intrinsics
          R_c2w: 3x3 camera-to-world rotation
          cam_center_world: (3,) camera position in world frame
        """
        prediction = self.model.inference(image_paths)

        depth = _to_numpy(prediction.depth).astype(np.float32)
        conf = _to_numpy(prediction.conf).astype(np.float32)
        intrinsics = _to_numpy(prediction.intrinsics).astype(np.float64)
        extrinsics = _to_numpy(prediction.extrinsics).astype(np.float64)

        results = []
        for i in range(len(image_paths)):
            R_w2c = extrinsics[i][:, :3]
            t_w2c = extrinsics[i][:, 3]
            R_c2w = R_w2c.T
            cam_center_world = -R_w2c.T @ t_w2c

            depth_i, conf_i, K_i = depth[i], conf[i], intrinsics[i].copy()
            if target_sizes is not None:
                th, tw = target_sizes[i]
                dh, dw = depth_i.shape
                if (th, tw) != (dh, dw):
                    depth_i = cv2.resize(depth_i, (tw, th), interpolation=cv2.INTER_LINEAR)
                    conf_i = cv2.resize(conf_i, (tw, th), interpolation=cv2.INTER_LINEAR)
                    sx, sy = tw / dw, th / dh
                    K_i[0, 0] *= sx; K_i[0, 2] *= sx  # fx, cx
                    K_i[1, 1] *= sy; K_i[1, 2] *= sy  # fy, cy

            results.append({
                "depth_m": depth_i,
                "conf": conf_i,
                "K": K_i,
                "R_c2w": R_c2w,
                "cam_center_world": cam_center_world,
            })
        return results


def check_camera_layout(results, shot_names):
    """Cheap sanity check to run on the FIRST real dataset: prints pairwise
    distances between camera centers. If DA3's convention were actually
    camera-to-world (i.e. we inverted something that didn't need inverting),
    these distances would typically come out wildly too large, all-zero, or
    NaN - obviously wrong at a glance, not a subtle bug. A real capture
    walking around a defect should give distances in the tens of centimetres
    to a few metres (DA3's own unscaled units, pre ruler-correction), not
    micrometres or kilometres."""
    centers = [r["cam_center_world"] for r in results]
    print("[geometry_da3] camera center pairwise distances (sanity check, pre-scale-correction):")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            print(f"  {shot_names[i]} <-> {shot_names[j]}: {d:.3f}")
