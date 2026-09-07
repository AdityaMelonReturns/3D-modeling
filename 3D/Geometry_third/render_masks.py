"""Render saved SAM2 masks (debug/masks/*.npy) as overlay images on the
original photos, for a completed run_defect.py output - no need to rerun
DA3/SAM2, just visualize what was already produced. Same as
Geometry_Second/render_masks.py, pointed at this pipeline's output/ folder.

Usage: python render_masks.py <defect_id>
Writes output/<defect_id>/debug/mask_overlays/<shot>_overlay.png per shot.
"""
import sys, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(__file__)


def main():
    defect_id = sys.argv[1]
    out_dir = os.path.join(HERE, "output", defect_id)
    masks_dir = os.path.join(out_dir, "debug", "masks")
    images_dir = os.path.join(HERE, "..", "Geometry", "workspace", defect_id, "images")
    overlay_dir = os.path.join(out_dir, "debug", "mask_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith("_mask.npy"))
    panels = []
    for mf in mask_files:
        shot_name = mf.replace("_mask.npy", "")
        mask = np.load(os.path.join(masks_dir, mf))
        img_path = os.path.join(images_dir, shot_name + ".jpg")
        rgb = np.array(Image.open(img_path).convert("RGB"))

        overlay = rgb.copy()
        red = np.zeros_like(rgb)
        red[..., 0] = 255
        alpha = 0.45
        overlay[mask] = (overlay[mask] * (1 - alpha) + red[mask] * alpha).astype(np.uint8)

        out_path = os.path.join(overlay_dir, shot_name + "_overlay.png")
        Image.fromarray(overlay).save(out_path)
        panels.append((shot_name, out_path, mask.sum()))
        print(f"{shot_name}: {mask.sum()}px masked -> {out_path}")

    return panels


if __name__ == "__main__":
    main()
