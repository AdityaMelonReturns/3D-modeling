# Geometry_third — Plan (simple version)

## What we are doing

Same goal as before: take a few photos of one house defect (hole, crack, peeling
paint, etc.) and turn them into a real 3D measurement (depth, width, length,
area, volume, tilt).

`Geometry` (COLMAP) and `Geometry_Second` (Depth Pro + SAM2) both had one
weak link in common: figuring out **where each camera was standing** relative
to the others. COLMAP couldn't do it with only 4-9 low-texture photos.
Geometry_Second worked around it with a guess (phone's tilt sensor + "assume
every camera is looking at the same spot") — and it showed: every single
defect we ran came back `pose_confidence: low`.

## What's changing

Swap that one weak stage for a model built specifically for this situation:
**Depth Anything 3 (DA3)**. Give it all the photos of one defect at once, and
it works out the depth *and* the camera positions *together*, instead of us
guessing the camera positions separately. This is the single biggest fix.

Everything downstream (turn depth+mask into 3D points, fit a flat reference
surface, measure the defect against it, build a mesh) barely has to change —
those stages never cared which model produced the points.

## The new pipeline

```
photos of one defect
        |
        +------------------+------------------+
        |                                      |
        v                                      v
   DA3 (all photos together)              SAM2 (per photo)
   -> depth + confidence                  -> defect mask
   -> camera position for each photo         (box you point at, same as before)
        |                                      |
        +------------------+-------------------+
                           |
                           v
              project masked pixels into 3D
              (using DA3's own camera positions -
               no more guessing)
                           |
                           v
              merge all photos into one 3D point cloud
                           |
                           v
              fit the "undamaged surface" plane
                           |
                           v
           measure: depth / width / length / area /
                    volume / tilt, each with a
                    confidence flag
                           |
                           v
              3D model you can view + JSON numbers
```

Depth Pro doesn't disappear — it moves to a smaller, second job: zooming into
just the defect on the sharpest photo and cross-checking the depth there.
It's no longer the thing that decides scale or camera position.

The ruler-in-frame scale check we already built stays exactly as it was.
DA3 claims to give real-world (metric) depth on its own, but Depth Pro made
that same claim and was 60-99% off every time we checked it against the
ruler — so we don't trust DA3's metric claim blindly either. The ruler stays
as the one honest scale check, not a bonus.

## Exactly where Depth Pro is used (and where it is deliberately NOT)

This is worth spelling out precisely, since "use it correctly" was the whole
point of moving away from Geometry_Second's design.

**Where it runs — one place only:** stage `[C']` in `run_defect.py`, function
`local_depth_pro_crosscheck()`, which calls `DepthEstimator.infer_crop()` in
`pipeline/depth_local.py`. What it actually does:

1. Look at SAM2's scores for every photo of this defect, pick the ONE photo
   with the best (most trustworthy) mask.
2. Crop that photo tightly around the defect box (with a margin), save just
   that crop.
3. Run Depth Pro on the crop only — not the full photo, not any other photo.
4. Pass it the ORIGINAL photo's focal length by hand (a crop we save
   ourselves has no real camera EXIF data to read, and cropping doesn't
   change the focal length, only the field of view).
5. Compare Depth Pro's median depth in the masked region against DA3's own
   median depth in that same region, on that same photo.
6. Write both numbers side by side into `measurements.json` under
   `local_depth_crosscheck`, plus a plain disagreement %.

**Where it explicitly does NOT run / does NOT decide anything:**
- It does not touch camera position — DA3's own camera positions are used
  for every photo, unchanged.
- It does not touch the scale correction — the ruler is still cross-checked
  against DA3's depth only (`pipeline/scale.py`), never Depth Pro's.
- It never runs on more than one photo per defect, and never on the whole
  photo — only that one tight crop.
- Its number is never blended/averaged into `depth_mm`, `width_mm`, `area_mm2`,
  etc. Those come from DA3 (after ruler correction) alone. Depth Pro's crop
  reading sits next to them as a second opinion — if it disagrees a lot,
  that's a flag to trust the measurement less, not a number we quietly mix in.

This matches the one job Depth Pro is actually good at and was built for:
a single close-up photo, not multi-photo alignment or absolute scale — which
is exactly what the recommendation doc argued for, and exactly the opposite
of how Geometry_Second used it (as the primary source for both).

## What we reuse untouched from Geometry_Second

These stages don't care where the 3D points came from, so they carry over as-is:
- SAM2 defect segmentation (`segment.py`)
- ruler detection + scale cross-check (`scale.py`)
- turning depth+mask into 3D points, and merging multiple photos (`fusion.py`)
- fitting the reference plane (`plane.py`)
- computing the actual measurements (`measure.py`)
- building the 3D mesh file (`mesh_export.py`)
- severity rules + optional Claude description (`interpret.py`)

## What's new

- `geometry_da3.py` — runs DA3 on all photos of one defect, returns per-photo
  depth, confidence, and camera position.
- `common.py` — simpler photo loader. Sensor-metadata JSON (tilt/heading) is
  now *optional*, used only as a secondary "which way is down" check for the
  tilt measurement — not required to run the pipeline at all. This also means
  a plain zip with no metadata (like the water-bottle test case) works with
  this pipeline without any special-casing.
- `depth_local.py` — Depth Pro, kept only for the local defect-crop cross-check
  described above.

## Known risks (being upfront about these before we run anything)

1. **DA3 needs `xformers`.** Our GPU (RTX PRO 6000 Blackwell) is very new;
   we already hit one "prebuilt package doesn't support this GPU" wall with
   plain PyTorch. DA3's own repo has a note for "older GPUs without xformers
   support" (their Issue #11) — so there's a known fallback path, but we
   won't know if we need it until we actually try installing on the pod.
   The install is capped at 5 minutes so a source-build fallback can't quietly
   eat the whole 1-hour pod window - it fails fast and we proceed without it.
2. **DA3's "metric" claim is unverified for our photos.** Every monocular/
   few-photo depth model we've tried so far overstates its own accuracy.
   Treat DA3's numbers the same way — cross-check with the ruler, don't
   assume "metric" means correct.
3. **Camera-position convention** — confirmed from DA3's own docs: their
   camera-position output is in the same format COLMAP uses (world-to-camera).
   We coded against that directly, but we'll still sanity-check it on the
   first real run (do the camera positions end up in physically sensible
   places relative to each other) before trusting it fully.
4. **Setup now installs into the existing `geo2` env, not a fresh one** -
   changed after you flagged the 1-hour pod limit: building a brand-new env
   would re-download torch/opencv/open3d/SAM2/Depth Pro that already work in
   `geo2`, costing several minutes we don't have. The trade-off is that
   `xformers`/DA3 could in theory disturb the already-working torch install
   in `geo2` (that's exactly the kind of thing that broke things before, on
   the cu126-vs-cu130 issue). The setup script guards against this directly:
   it installs both with `--no-deps` so pip can't pull in a different torch
   on its own, and it runs a real CUDA test before AND after, printing an
   explicit warning if anything changed - so if this bites us, we'll see it
   immediately in the output rather than finding out later.

## Licensing note

We're using `DA3METRIC-LARGE` — the metric-depth variant, and it's
Apache 2.0 (free for commercial use, no approval process). The bigger/more
accurate DA3 variants are non-commercial only, so we're deliberately not
using those.

## Rollout plan

1. Build all the code below (done in this same session, before any pod is running).
2. You start the pod once.
3. We run **one** zip first — `hole`, since it's our hardest defect category
   (a true cavity, not just a surface mark) and we already have a
   Geometry_Second result for it to compare against directly.
4. If that looks right, run a second zip to confirm it's not a fluke.
5. Compare both against the existing Geometry_Second numbers/mesh for the
   same zips, then decide together whether to redo the rest of the 8 zips
   through this pipeline.

`bottle` is understood to be for your other use case, not part of this
defect-measurement pipeline — not touching it here.
