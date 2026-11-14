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

## v2: reference-object scale + darkness core (Intern Brief, Oct 26 2026)

Two gaps from v1: 30 of 55 photos had no usable ruler (pixel sizes only), and depth was never
reported. v2 adds, without changing the ruler path (a found ruler always wins):

```
photo + box -> outline -> scale:  1. ruler card (unchanged)
                                  2. reference object (NEW, measure/reference.py)
                                  3. none -> pixel sizes only
                       -> measure -> depth: probe band (unchanged)
                                            + dark core of deep damage (NEW, measure/depth_band.py)
```

**Task A - scale from building parts** (`measure/reference.py`, sizes + sources in
`docs/REFERENCE_SIZES.md`). Hand-drawn box per photo in `prompts_reference/<defect>.json`:
- repeating patterns (shingle courses, deck boards, brick): direction from long straight lines, a
  brightness profile across the repeats *through the defect*, spacing from every repeat with a smooth
  fit so perspective along the profile is handled; `n_repeats` reported.
- single objects (a 2x6 rail cap): two clicks on each edge, snapped to the strongest nearby edge.
- scale is known in one direction only - stated in the output. Confidence is medium at most, low when
  the reference is on another surface (`same_surface: false`) or the pattern fit is weak.
- JSON: `scale.source = "reference:<object>"`, `reference_size_source`, `n_repeats`, `inferred: true`
  on every number, size tolerance in `uncertainty_pct`.

Acceptance test (`python reference_acceptance.py` -> `output/REFERENCE_ACCEPTANCE.md`), ruler ignored,
compared at the ruler's position, on the 25 ruler photos:
- reference found on 8, none visible on 17 (ceiling drywall, gaps fascia, fence boards: no standard part)
- within target on 1 of 8 (wood1_06 rail cap +9.4%). roof_03 shingles -15%, wood1_05 -19%.
- peeling_paint 06-11: the white boards repeat every ~4.5 cm by the ruler, not the 5-1/2 in of a deck
  board, so they fail by 60-85%. Either they are not deck boards (2x2 slats + gap would fit) or the
  hand-held ruler is off the surface. Needs someone who has seen it.
- on the 30 no-ruler photos: scale inferred on 6. **roof_01 crack = 26.7 cm +-11% from shingle courses**
  (pilot: 25-28 cm; the tiny ruler in that photo agrees within 10%). wood1_02/03/04 from the rail cap.
  peeling_paint_02/04 are inferred from floor deck boards on a different surface -> low, not usable.

**Task B - depth** (`measure/depth_band.py`, `train_depth_band.py`, `docs/CAPTURE_PROTOCOL.md`):
- *dark core* ("darkness classifier"): pixels inside the outline darker than 60% of the clean surface
  in a thin ring around it -> `depth.dark_core` with core area (cm2 when scaled) and fraction. An
  area, never a depth. Shaded rims are flagged (`rim_uneven`).
- `depth.features`: brightness ratio, colour difference, texture, shape - the classifier's inputs.
- side-photo depth reader (ruler pushed in, hidden length = depth). Simulated on 6 real rulers by
  hiding 0.5-6 cm: ~1 cm median error from the ruler width - too coarse for 1 / 3 cm band edges, so
  the protocol asks for a close-up (readable ticks) and the worker's reading at the surface.
- classifier: refuses to train below 200 labels (`labels/depth_labels.csv` is empty). Split by
  property, confusion matrix, deep-called-shallow rate, majority baseline - all ready; `--smoke`
  checks the code on random data.

Not done yet: Grounding DINO in place of hand boxes; adding the inferred numbers to the viewer.
Found while doing this: wood1_05's SAM2 outline is the shadowed side of the rail, not the rot
(v1's 42.6 cm there is wrong) - its box needs redrawing.
