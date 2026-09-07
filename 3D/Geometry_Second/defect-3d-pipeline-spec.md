# Defect 3D Reconstruction Pipeline — Implementation Spec

## 1. The Core Problem (and why we're not doing classical SfM)

Classical photogrammetry (COLMAP feature-match → sparse → dense) needs 60-80% visual overlap between consecutive frames and real texture to match on. With only 4-5 images shot from spread-out angles around a defect on a wall/ceiling, two things kill it:

- **Low overlap, wide baselines.** 4-5 "walk-around" shots are the worst case for feature matching — not enough redundancy to recover from a single failed pair.
- **Textureless surfaces.** Drywall, ceilings, painted walls give SIFT/ORB almost nothing to lock onto in the defect interior — the one place we actually need geometry.

**Decision: skip full SfM as the geometry source.** Use it (or the sensor data) only to get rough relative camera poses. The actual depth/geometry comes from **per-image monocular metric depth estimation**, fused across views. This is the standard move in industry pipelines that have to work with sparse, uncontrolled captures (real estate scanning apps, insurance-inspection apps do exactly this).

Pipeline stages are collapsed from 3 (match/sparse/dense) into essentially 2: **pose estimation (lightweight)** and **depth fusion (does the real work)**.

---

## 2. Pipeline Overview

```
Input: 4-5 JPEGs + sidecar JSONs per defect
   │
   ├─► [A] Preprocess & Parse
   │       - EXIF-correct orientation, undistort using intrinsics
   │       - Parse sidecar JSON → gravity vector, heading, GPS, focal length
   │
   ├─► [B] Camera Pose Estimation (hybrid, lightweight)
   │       - Try sparse feature matching (LightGlue/SuperPoint, not COLMAP SIFT)
   │       - Fallback / regularizer: sensor-derived pose priors
   │       - Output: best-effort relative poses + confidence flag
   │
   ├─► [C] Monocular Metric Depth (primary geometry source)
   │       - Depth Pro (Apple) per image → metric depth map
   │       - This is the workhorse; works even with zero texture
   │
   ├─► [D] Scale Calibration
   │       - Detect scale card (ArUco / known-size card) if present
   │       - Correct/verify Depth Pro's metric scale against it
   │       - If absent: flag measurements as "depth-model-scale" not "card-verified"
   │
   ├─► [E] Defect Segmentation
   │       - SAM2, prompted per image (box/point) → per-image defect mask
   │
   ├─► [F] Back-projection + Multi-view Fusion
   │       - Each masked depth map → point cloud (using intrinsics + pose)
   │       - Align via poses, refine with colored ICP (Open3D)
   │       - Merge into single fused point cloud (defect + surrounding margin)
   │
   ├─► [G] Reference Plane Fitting
   │       - RANSAC plane fit on the *surrounding* (non-defect) points
   │       - This plane = "what the surface should look like without the defect"
   │
   ├─► [H] Measurement Extraction
   │       - Depth/relief, width/length, area, volume, inclination angle
   │
   ├─► [I] Meshing + Export
   │       - Poisson/ball-pivot mesh, depth-deviation heatmap texture
   │       - Export .glb/.ply/.obj
   │
   └─► [J] Interpretation Layer
           - Rule-based + VLM (Claude vision) severity/diagnosis
           - Final JSON report + confidence scores
```

---

## 3. Stage Details

### [A] Preprocess & Parse
- Rotate image per `exif_orientation`.
- Build camera intrinsics matrix `K` from `focal_length_mm`, sensor size (derive from `focal_length_35mm_equiv` ratio if sensor size unknown for iPhone 15 Pro — hardcode known sensor width for that model, it's fixed hardware).
- From `raw.accelerometer_g` → **gravity vector in camera frame** (this is your ground-truth "down" per shot, independent of any reconstruction — use it later for inclination even if everything else fails).
- From `orientation.pitch/roll/yaw` + `heading.true_heading_deg` → rough world-frame camera rotation estimate.
- GPS/altitude/barometer: log only, not precise enough for pose (±1400m horizontal accuracy in your sample — basically useless for metric pose, don't rely on it beyond "same location" sanity check).

### [B] Camera Pose Estimation
Two-track approach, take whichever is better per set:

1. **Feature-based (try first):** Use LightGlue + SuperPoint (far better than SIFT on low-texture) for pairwise matching across all 4-5 images → 2-view/N-view relative pose via essential matrix. With only 4-5 images this is cheap to brute-force (all pairs).
2. **Sensor-based (fallback/regularizer):** Convert pitch/roll/yaw + heading into absolute rotation matrices per shot. Translation is the weak point (no reliable metric position) — approximate by assuming roughly constant distance-to-wall (estimate from depth map's own median value once [C] runs — yes, use depth output to bootstrap pose translation, it's fine to loop back).
3. **Fuse:** If feature matching produced any usable pairs, use it for rotation refinement; always use sensor gravity vector as a hard constraint (it's more reliable than anything vision-derived). Tag each set with a `pose_confidence: high/medium/low` based on how many image pairs matched.

Do not spend more than a day trying to make feature matching robust here — it's a supporting signal, not the geometry source. If it fails entirely, sensor-only pose is an acceptable degraded mode; flag it as such.

### [C] Monocular Metric Depth — the actual geometry engine
- Use **Depth Pro** (Apple, metric-scale native — doesn't need a reference object to know real-world scale, unlike ZoeDepth/MiDaS which are often relative-depth only). Run per image independently.
- This is what saves you from the "4-5 images is not enough for MVS" problem — you get a full dense depth map from **one image**, no matching required.
- Store depth map + confidence map per image.

### [D] Scale Calibration
- If scale card visible: detect via ArUco (if it's a printed marker) or simple contour/aspect-ratio detection (if it's a known-size card). Compute known-size-in-pixels → metric ratio at that image region, cross-check against Depth Pro's own metric output at that same region.
  - If they disagree by more than ~5-10%: trust the scale card, apply a correction scalar to that image's depth map.
- If no card: trust Depth Pro's native metric scale but tag `scale_source: "model_estimate"` (lower confidence) vs `"card_verified"`.
- This directly answers your "does the scale card matter" question — log both numbers when a card is present, don't just overwrite.

### [E] Defect Segmentation
- SAM2 per image, prompt with a box (Claude can draw/pick this) around the visible defect. Store binary mask per image aligned to that image's pixel grid.
- Keep a margin (mask dilated ~15-20%) so the fused point cloud includes surrounding surface for plane-fitting in [G].

### [F] Back-projection + Fusion
- For each image: `depth_map × mask(dilated) × K⁻¹` → per-image point cloud in camera frame → transform to world frame using [B]'s pose.
- Merge all point clouds. Run **colored ICP** (Open3D) pairwise to clean up residual misalignment from imperfect poses — this compensates for [B] being approximate.
- Output: one fused point cloud covering the defect + immediate surrounding surface.

### [G] Reference Plane
- Take only the *non-masked* (surrounding, non-defect) points from the fused cloud.
- RANSAC plane fit (Open3D `segment_plane`) → plane normal `n`, plane point `p0`.
- This plane is your "healthy surface" baseline — every measurement below is relative to it.

### [H] Measurements (this is your "different outputs" deliverable)
For each defect point `x`, signed distance to plane: `d = n · (x - p0)`.

- **Depth/relief** (holes, rot, gaps): `max(|d|)`, `mean(|d|)` over defect points, plus a depth histogram.
- **Width / Length**: project defect points onto the plane (drop the normal component), take 2D convex hull, run PCA on hull points → major axis = length, minor axis = width.
- **Area**: hull area (2D, on-plane).
- **Volume** (holes/rot only): area × mean depth (or integrate `d` over a fine grid — more accurate for irregular pits).
- **Inclination angle**: angle between plane normal `n` and gravity vector (from accelerometer, [A]) → this is the surface tilt in degrees, independent of camera pose accuracy. Cross-check against device `pitch_deg`/`roll_deg` for a second opinion — report both, flag disagreement.
- **Confidence score** per measurement: combine `pose_confidence`, `scale_source`, and point-cloud density/consistency in the defect region (few points = low confidence, don't just report a number silently).

### [I] Meshing + Export
- Poisson surface reconstruction (Open3D) on fused cloud → mesh.
- Texture/vertex-color by `d` (signed distance) as a heatmap — this alone is often the most useful visual output for a human reviewing the defect.
- Export `.glb` (viewable) + `.ply` (raw) + the original masked images for reference.

### [J] Interpretation
- Rule-based first pass: thresholds per defect type (e.g., depth > X mm + area > Y cm² → "moderate", etc. — you'll calibrate these against your real measurements once you have a few sets processed).
- Then send: cropped defect image + computed metrics (depth, width, length, angle, area) to Claude (vision) with a prompt asking for defect type confirmation + severity reasoning + plain-language description. Structured JSON output.
- Final report combines: measurements, confidence scores, mesh/heatmap file paths, and the interpretation text.

---

## 4. Known Failure Modes — Don't Fight These, Just Detect & Flag

| Case | Behavior |
|---|---|
| Defect smaller than ~5-10mm extent | Depth Pro resolution floor — report "below reliable measurement threshold," don't fabricate a number |
| No scale card anywhere in set | Use model-estimate scale, flag confidence as low |
| Feature matching finds 0 good pairs | Fall back fully to sensor-based pose, flag `pose_confidence: low`, propagate that to all downstream measurement confidences |
| Inclination < 2° | Report as "no measurable lean, within sensor noise" rather than a false-precision number |
| Extreme angle spread between the 4-5 shots (e.g. one from directly above, rest from the side) | Fusion may produce a sparse/gappy region — flag low point density areas in the heatmap rather than interpolating silently |

---

## 5. Libraries / Tools

- **Depth**: Depth Pro (Apple, open weights) — primary. ZoeDepth as secondary/cross-check if useful.
- **Matching**: LightGlue + SuperPoint (lightweight, GPU, much better than SIFT on low-texture indoor scenes)
- **Segmentation**: SAM2
- **Point cloud / geometry**: Open3D (ICP, RANSAC plane, Poisson meshing, PCA)
- **Scale card detection**: OpenCV (ArUco module if you can print markers going forward — much more reliable than a plain card)
- **Mesh export**: trimesh or Open3D's own exporter
- **Interpretation**: Claude (vision) via API, structured JSON output mode

Skip COLMAP entirely as a dependency unless you specifically want it as an optional cross-check for pose on the sets where texture happens to be good (inclined surfaces with visible trim, etc.) — don't make it load-bearing.

---

## 6. Output Package Per Defect Set

```
/output/{defect_id}/
  mesh.glb
  pointcloud.ply
  heatmap_depth.png
  measurements.json   # depth, width, length, area, volume, angle, confidences
  interpretation.json # defect type, severity, description (from Claude)
  debug/
    poses.json         # per-image pose + confidence source (feature/sensor)
    masks/              # per-image defect masks used
```

`measurements.json` shape:
```json
{
  "defect_id": "ceiling",
  "depth_mm": {"max": 0.0, "mean": 0.0, "confidence": "medium"},
  "width_mm": 0.0,
  "length_mm": 0.0,
  "area_mm2": 0.0,
  "volume_mm3": 0.0,
  "inclination_deg": {"value": 0.0, "source_agreement": true, "confidence": "high"},
  "scale_source": "card_verified",
  "pose_confidence": "medium",
  "shots_used": 4
}
```
