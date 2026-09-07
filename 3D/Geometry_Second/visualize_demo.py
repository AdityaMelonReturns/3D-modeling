"""One-off visualization: shows exactly how the pipeline's internal data maps
back onto the original photo, pixel-for-pixel. Produces a 4-panel figure:
  1. Original photo
  2. SAM2 segmentation mask, overlaid
  3. Depth Pro's depth map, overlaid (as a color map - blue=near, red=far)
  4. Signed-distance-to-plane heatmap, overlaid (red=bulges out, blue=recedes/hole)

Run once per shot for inspection - not part of the main run_defect.py flow.
Usage: python visualize_demo.py <defect_id> <shot_name> <box_x0> <box_y0> <box_x1> <box_y1> [device]
device defaults to cuda if available, else cpu (CPU is the ~5min/image path).
"""
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.common import load_defect_set, load_image_rgb
from pipeline.depth import DepthEstimator
from pipeline.segment import DefectSegmenter
from pipeline.plane import fit_reference_plane, signed_distance
from pipeline.fusion import backproject
import open3d as o3d


def main():
    defect_id = sys.argv[1]
    shot_name = sys.argv[2]
    box = list(map(int, sys.argv[3:7]))
    device = sys.argv[7] if len(sys.argv) > 7 else ("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.join(os.path.dirname(__file__), "..", "Geometry", "workspace", defect_id)
    shots = load_defect_set(base_dir, defect_id)
    shot = next(s for s in shots if s.name == shot_name)
    rgb = load_image_rgb(shot)

    print(f"Running Depth Pro on device={device} ...")
    depth_est = DepthEstimator(device=device)
    depth_m = depth_est.infer(shot)["depth_m"]

    print("Running SAM2...")
    segmenter = DefectSegmenter(device=device)
    seg = segmenter.segment(rgb, box)
    mask, mask_dilated = seg["mask"], seg["mask_dilated"]
    surround_mask = mask_dilated & ~mask

    print("Fitting local reference plane (single-image demo, not multi-view fused)...")
    surround_pcd = backproject(depth_m, shot.K, surround_mask, rgb)
    normal, point_on_plane, inlier_ratio = fit_reference_plane(surround_pcd, ransac_thresh=0.01)
    print(f"plane inlier_ratio={inlier_ratio:.2f}")

    # signed distance for every pixel in the (dilated) region, for a full heatmap layer
    ys, xs = np.nonzero(mask_dilated)
    z = depth_m[ys, xs]
    X = (xs - shot.K[0, 2]) * z / shot.K[0, 0]
    Y = (ys - shot.K[1, 2]) * z / shot.K[1, 1]
    pts3d = np.stack([X, Y, z], axis=1)
    d = signed_distance(pts3d, normal, point_on_plane)

    heat = np.zeros(rgb.shape[:2], dtype=np.float32)
    heat[ys, xs] = d
    d_scale = np.percentile(np.abs(d), 98) + 1e-9

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))

    axes[0].imshow(rgb)
    axes[0].set_title(f"1. Original photo\n{shot_name}")
    x0, y0, x1, y1 = box
    axes[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=2))

    axes[1].imshow(rgb)
    mask_overlay = np.zeros((*mask.shape, 4))
    mask_overlay[mask] = [1, 0, 0, 0.5]
    mask_overlay[surround_mask] = [1, 1, 0, 0.25]
    axes[1].imshow(mask_overlay)
    axes[1].set_title("2. SAM2 segmentation\nred=defect core, yellow=surrounding margin")

    axes[2].imshow(rgb)
    depth_masked = np.ma.masked_where(~mask_dilated, depth_m)
    im2 = axes[2].imshow(depth_masked, cmap="turbo", alpha=0.75)
    plt.colorbar(im2, ax=axes[2], label="metres (Depth Pro raw, uncorrected)")
    axes[2].set_title("3. Depth Pro depth map\n(blue=near camera, red=far)")

    axes[3].imshow(rgb)
    heat_masked = np.ma.masked_where(~mask_dilated, heat)
    im3 = axes[3].imshow(heat_masked, cmap="coolwarm", vmin=-d_scale, vmax=d_scale, alpha=0.8)
    plt.colorbar(im3, ax=axes[3], label="metres from fitted reference plane")
    axes[3].set_title("4. Depth-deviation heatmap\nred=bulges toward camera, blue=recedes (hole)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "output", f"{defect_id}_{shot_name}_demo.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
