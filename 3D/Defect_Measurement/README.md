# Defect_Measurement

Implements `../Defect_Measurement_Simplified.docx`: defects are **measured from one photo**, not
reconstructed. No COLMAP, DA3, fusion, plane fitting on point clouds, meshing or Qwen in the defect path.
Runs on a laptop CPU (SAM2 ~30 s per photo, masks cached; re-runs take seconds).

```
C:\g2venv\Scripts\python.exe run.py                 # all 8 sets, every prompted photo
C:\g2venv\Scripts\python.exe run.py roof ceiling    # some sets
C:\g2venv\Scripts\python.exe run.py --primary-only  # one photo per defect
```

Outputs: `output/<defect>/<shot>.json`, `<shot>_overlay.jpg` (green box, red outline, blue card,
straight-on inset bottom-right), and `output/SUMMARY.md` / `summary.json` across all sets.

## Pipeline

| Step | What | Where |
|---|---|---|
| Box | Hand-drawn, `prompts/<defect>.json` (`box`, optional `pos_points` / `neg_points`). YOLO/GDINO later, same format. | `prompts/` |
| Outline | **crisp**: SAM2 mask from the box. **thin**: Frangi ridge filter, then a directional line-opening (keeps only long straight ridges), threshold, skeleton; positive clicks pick which line is the crack. **diffuse**: colour difference from the clean surface around the box (largest colour cluster in a ring), reported as a tight-loose range. Segmentation is always on the original photo. | `measure/edges.py`, `measure/segment.py` |
| Scale + rectify | Find the ruler card, reject false ones, map its outline to a straight-on rectangle; warp the **mask** (never the photo) and measure there. | `measure/card.py` |
| Measure | Area = pixels x (cm/px)^2. Length = longest path along the skeleton. Width = 2 x distance-to-edge on the skeleton (median and p95). Extent = min-area rectangle (L x across). | `measure/metrics.py` |
| Angle | Phone accelerometer rotated into this photo's camera frame; every edge is compared with true vertical at its own pixel (vertical vanishing point). | `measure/gravity.py` |
| Depth | Worker probe band only (`config.py -> probe`). Never from the image. | `config.py` |

Every number in the JSON carries `scale_source`, `confidence` (high / medium / low / none) and
`uncertainty_pct`. The JSON also includes the rejected card candidates with reasons, and flags when the
outline runs off the photo (size then reported as a **lower bound**).

## Things found while building this

1. **The ruler's blue body is 17.0 x 2.1 cm, not 15 cm.** Measured by reading its printed cm ticks: the 9
   photos where the ticks are clearly periodic give 17.0 cm (IQR 16.9-17.3). The old code treated the
   whole blue blob as 15 cm, so it read every size about 13% too large. **Confirm with a tape** (`config.py
   CARD_LENGTH_CM`); every scaled number moves 1:1 with it.
2. **Gravity in the camera frame: `g_cam = (-a_y, -a_x, -a_z)`** for these iPhone photos (pixels stored
   landscape, EXIF orientation 6), focal 14 mm equiv = 747 px. Checked on all 82 photos: 10.9% of 1,842
   long straight lines lie within 2 deg of the predicted vertical, versus at most 4.2% for each of the other 7
   axis mappings. This replaces the mixed sensor/DA3 frames that made walls read ~90 deg.
3. **False rulers are now rejected**: blue shadows on the roof, frosted window glass (hole), a blue wagon
   and a ruler-shaped strip of blue paint (railing). Checks: aspect, solidity, implied camera distance,
   minimum 200 px length, and end-width ratio (catches an outline cut short by glare, e.g.
   peeling_paint_08, where the blue stops at the 9 cm tick).
4. **Several old SAM2 masks were the ruler, not the defect** (roof 02/05/06, gaps 05/06/07). The detected
   ruler is now fed to SAM2 as "not this" clicks automatically.
5. **A ruler is a weak card for perspective.** It is 8x longer than wide, so its 4 corners fix
   perspective along it but barely across it. Both the homography and a plain affine mapping are computed;
   their disagreement goes into `uncertainty_pct`. A squarer card (credit card, A5 sheet) would fix this.

## Acceptance check (one-pager: "reproduce the pilot within ~15%")

| Defect | Ours (primary photo) | Pilot | Verdict |
|---|---|---|---|
| ceiling hole | 7.5 cm across (27.6 cm long tear), scale high | 5-7 cm | **pass** |
| roof crack | 9.9 cm (affine mapping: ~15% more), scale medium | 25-28 cm | **fail**: only part of the line is traced (pine needles break it), and which dark line is the crack vs a shingle joint needs confirming |
| wood1 split | 19.1 cm long, 18.1 x 2.5 cm extent, scale medium | 10-15 cm ("rough, ruler at angle") | **fail** by 11%; the outline visibly follows the split, so the pilot's rough guess may be low |
| railing | posts 4.6 deg from vertical (all 7 photos 2.1-6.8 deg) | 29-36 deg (pilot flagged it unreliable) | disagrees: the posts are near plumb. The pilot's plane fit mixed deck, siding and bushes |
| hole | not measured: no ruler in any photo | 2-3 cm visual guess | correct per one-pager (size band + photo) |

## Needs a human

- **Tape-measure the ruler body** (item 1).
- **roof**: confirm which line is the crack; redraw the box/clicks in `prompts/roof.json`.
- **rotten_wood**: the boxes outline the red house siding (09, 11) or a fingertip (04). Redraw around
  the actual rot. The ruler is hand-held in every photo, so confidence stays low anyway.
- **peeling_paint**: diffuse edges are the weakest type. The loose mask follows the peeled wood; the tight
  one only catches the darkest cracks (and the hand in 11), so the range is wide (07: 1.2-11.6 cm2).
- **Severity bands** use thresholds carried over from the old code; they need recalibrating.

## Capture advice for next time

Lay a **square-ish card flat on the defect's surface** in the same photo, fill the frame with the defect,
and record a probe-depth band for holes/rot. LiDAR distance would enable the one-pager's flatness rule.
