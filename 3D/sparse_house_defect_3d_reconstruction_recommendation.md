# Recommended 3D Reconstruction Strategy for Sparse House-Defect Images

## Executive recommendation

### Start with: Depth Anything 3 (DA3)

For this project, the **first system to implement and benchmark should be Depth Anything 3 in multi-view mode**, using the 2–3 available images jointly.

Then add:

1. **A defect segmentation/detection model** for holes, peeling paint, cracks, water damage, roof damage, etc.
2. **Depth Pro or Metric3D v2 as a local depth-refinement/support model**, not as the main multi-view reconstruction engine.
3. **VGGT as the main alternative benchmark**.
4. **MASt3R as the second alternative benchmark**, especially for large viewpoint changes.
5. **COLMAP as a classical baseline**, not the primary approach.

The core idea is:

```text
2–3 inspection images
        |
        +-----------------------+
        |                       |
        v                       v
Multi-view geometry       Defect detection/
DA3                       segmentation
        |                       |
        v                       v
camera/3D geometry        defect masks
        |                       |
        +-----------+-----------+
                    |
                    v
          Local defect geometry
                    |
                    v
       size / depth / area / volume /
       position / confidence
```

This is a better fit than trying to make traditional photogrammetry reconstruct an entire house from only 2–3 partial photographs.

---

# 1. Your actual problem

The target is not a normal photogrammetry dataset.

You have images such as:

- ceiling holes
- roof holes
- peeling paint
- cracks
- water damage
- damaged plaster
- missing material
- surface depressions/protrusions
- other structural/maintenance defects

Typical input:

```text
Image 1: front/oblique view
Image 2: side/oblique view
Image 3: another angle
```

The images may:

- cover only part of the house
- not cover 360 degrees
- have large viewpoint changes
- have weak texture
- contain shadows
- have repetitive paint/plaster texture
- show defects only partially
- contain areas that are completely unobserved

That changes the algorithm choice.

You do **not** need to solve "perfect full-house reconstruction from arbitrary three photos."

The more useful objective is:

> Build a consistent 3D representation of the **observed surface and defect**, and estimate defect geometry as accurately as possible.

---

# 2. Important limitation: 2–3 images cannot recover unseen geometry

No model can truly reconstruct geometry that the cameras never observe.

For example:

```text
camera A                 camera B
    \                       /
     \                     /
      \       defect      /
       \        X        /
        +---------------+
          visible area

             |
             |
             v

       completely unseen
             area
```

A neural model may produce a plausible surface in the unseen area, but this should be treated as **inference/hypothesis**, not measured geometry.

For an inspection system, keep these concepts separate:

- **Observed geometry**
- **Model-inferred geometry**
- **Defect confidence**
- **Measurement confidence**

This becomes very important if the system is eventually used for engineering, insurance, maintenance, or safety decisions.

---

# 3. First choice: Depth Anything 3

## Why DA3 is the first model to test

Depth Anything 3 was introduced in November 2025 and is explicitly designed to predict **spatially consistent geometry from an arbitrary number of visual inputs**, with or without known camera poses.

Its benchmark reports that DA3 outperformed VGGT on its visual-geometry benchmark by:

- 44.3% average improvement in camera-pose accuracy
- 25.1% average improvement in geometric accuracy

according to the authors' benchmark. Treat these as paper-reported benchmark numbers rather than as a guarantee for house-defect images.

Source:
https://arxiv.org/abs/2511.10647

Official implementation:
https://github.com/ByteDance-Seed/Depth-Anything-3

### Why that matters for your project

Your input count is tiny:

```text
N = 2
N = 3
```

DA3 is specifically designed around arbitrary-view geometry rather than requiring a long photogrammetry sequence.

Official example usage accepts an image list and produces multi-view predictions, including depth.

---

# 4. Why not use Depth Pro as the main method?

Depth Pro is excellent, but its role is different.

Depth Pro is a **monocular metric-depth model**.

Input:

```text
one image
     |
     v
Depth Pro
     |
     v
metric depth
```

It produces high-resolution metric depth with absolute scale and does not require camera intrinsics metadata.

That makes it very attractive for:

- single-image defects
- local depth
- depth boundaries
- surface geometry around a defect

But it does not solve the same problem as multi-view reconstruction.

The main issue is consistency:

```text
Image A -> Depth Pro -> 3D_A
Image B -> Depth Pro -> 3D_B
Image C -> Depth Pro -> 3D_C

The three independently estimated scenes
may not align perfectly in one global coordinate system.
```

Therefore:

### Use Depth Pro for:

- monocular fallback
- local depth refinement
- depth boundary quality
- cases where only one usable image exists

### Do not make Depth Pro the primary:

- multi-view pose estimator
- cross-image alignment engine
- global sparse-view reconstruction engine

Source:
https://arxiv.org/abs/2410.02073

---

# 5. Why COLMAP should be a baseline, not the first choice

COLMAP remains an excellent and very important system.

It performs:

```text
features
   ->
matching
   ->
camera pose / SfM
   ->
MVS
   ->
dense point cloud
   ->
mesh
```

Official documentation:
https://colmap.github.io/tutorial.html

However, COLMAP's own guidance emphasizes:

- good texture
- high visual overlap
- different viewpoints
- preferably seeing each object in at least 3 images
- enough shared content to establish feature correspondences

This is precisely where your data can be difficult.

A damaged white ceiling could be:

```text
mostly textureless
+
lighting change
+
peeling paint
+
large crack/hole
+
only 2 views
```

Traditional feature matching may not find enough stable correspondences.

So COLMAP should be retained as:

> **Classical photogrammetry baseline / fallback / validation system**

not the first neural reconstruction system.

---

# 6. Second choice: VGGT

VGGT is an excellent second system to benchmark.

It was introduced at CVPR 2025 and directly predicts:

- camera parameters
- point maps
- depth maps
- 3D point tracks

from one, a few, or many views.

Paper:
https://arxiv.org/abs/2503.11651

Official repository:
https://github.com/facebookresearch/vggt

This is very close to your problem.

A useful conceptual pipeline is:

```text
Image 1
Image 2  ---> VGGT ---> cameras + depth + point maps + tracks
Image 3
```

### Why I want VGGT in your benchmark

It is a strong, well-established few-view geometry model and is also useful as a bridge to a downstream COLMAP/3D Gaussian workflow.

The official repository also provides code to save predictions in COLMAP format.

---

# 7. Important 2026 update: VGGT-Omega

The official VGGT repository states that **VGGT-Omega** was released in May 2026.

It also reports an implementation update that allows roughly 2–3x more input frames under the same GPU memory budget.

That is less important for your current 2–3 image case, but it matters if your data collection later grows from:

```text
3 images
    ->
10 images
    ->
20+
```

So VGGT is a good technology to keep in the project even if DA3 wins the first benchmark.

---

# 8. Third choice: MASt3R

MASt3R is particularly interesting when image viewpoints differ strongly.

Paper:
https://arxiv.org/abs/2406.09756

Official repository:
https://github.com/naver/mast3r

Its main strength is strong 3D-grounded image matching.

This is useful when you have:

```text
view A             view B

   ______            ______
  /      \          /      \
 / defect \        / defect \
```

but the visual appearance changes significantly.

MASt3R was designed to improve matching robustness over DUSt3R while preserving strong 3D geometry capabilities.

### When MASt3R may beat DA3

Do not assume DA3 automatically wins every defect category.

I would specifically test MASt3R on:

- large viewpoint changes
- textured roof defects
- irregular geometry
- objects/details visible from very different angles
- cases where correspondence is hard

---

# 9. DUSt3R: keep as a research baseline

DUSt3R is an earlier foundation for this family of approaches.

Official repository:
https://github.com/naver/dust3r

Paper:
https://arxiv.org/abs/2312.14132

It is still useful for:

- understanding point-map based reconstruction
- baseline experiments
- research comparisons
- debugging a multi-view geometry pipeline

But for a new system, I would prioritize:

```text
DA3
VGGT
MASt3R
```

before spending much engineering time on DUSt3R.

---

# 10. The biggest architecture decision: reconstruction vs defect detection

Do not combine these into one problem.

You really have two separate tasks:

## Task A — What is the 3D geometry?

Possible tools:

- DA3
- VGGT
- MASt3R
- COLMAP
- Depth Pro
- Metric3D
- other geometry models

## Task B — Where is the defect?

Possible tools:

- object detection
- semantic segmentation
- instance segmentation
- open-vocabulary segmentation
- custom defect segmentation model

The system should eventually look like:

```text
                   input images
                        |
           +------------+------------+
           |                         |
           v                         v
    multi-view geometry       defect segmentation
       DA3 / VGGT /            detector/segmenter
        MASt3R                      |
           |                         |
           +------------+------------+
                        |
                        v
                 mask-to-3D fusion
                        |
                        v
                local 3D defect
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
        area          depth        volume
```

This separation is extremely important.

---

# 11. Recommended production architecture

I recommend this architecture as the first serious implementation.

## Stage 0 — Image quality check

Before running any reconstruction:

Check:

- blur
- exposure
- severe shadows
- lens distortion
- too little overlap
- extreme viewpoint change
- whether the defect is actually visible
- whether the surface is mostly textureless

Produce a per-image quality score.

Example:

```text
image_001.jpg
  sharpness      = 0.91
  exposure       = 0.87
  defect_visible = yes
  overlap_score  = 0.78

image_002.jpg
  sharpness      = 0.62
  exposure       = 0.71
  defect_visible = yes
  overlap_score  = 0.74
```

---

# 12. Stage 1 — Defect segmentation

The defect should be segmented before detailed geometric analysis.

Example:

```text
RGB image
    |
    v
defect segmentation
    |
    v
binary/instance mask

     _____________
    |             |
    |    ####     |
    |   ######    |
    |    ####     |
    |_____________|
```

Possible classes:

```text
hole
crack
peeling_paint
water_damage
spalling
missing_tile
roof_damage
surface_depression
surface_bulge
exposed_material
other
```

### Important

You should build the data model so that a single image can contain multiple defects.

Do not assume:

```text
one image = one defect
```

Instead:

```text
image
 |
 +-- defect 1
 +-- defect 2
 +-- defect 3
```

---

# 13. Stage 2 — Multi-view geometry using DA3

Send all available views of the same defect/scene together:

```text
[
  image_01,
  image_02,
  image_03
]
```

DA3 should produce the common geometry representation.

Store:

```text
camera poses
depth
point representation
confidence
image-to-image correspondence information
```

Do not immediately turn the result into a final watertight mesh.

For your use case, a point representation is often safer.

---

# 14. Stage 3 — Local crop refinement

This is an important optimization.

You do not necessarily need the highest possible geometry resolution across the whole house.

Instead:

### First pass

Run geometry on the complete images.

### Second pass

Crop around the defect:

```text
full image
+------------------------------+
|                              |
|       +-------------+        |
|       |    defect   |        |
|       |     XXX     |        |
|       +-------------+        |
|                              |
+------------------------------+
```

Then run local geometry/depth estimation.

Use:

- Depth Pro
- Metric3D v2
- potentially DA3 again on the cropped multi-view set

This gives you more computational budget exactly where the inspection measurement is needed.

---

# 15. Stage 4 — Fuse the segmentation with 3D

You have:

```text
2D defect mask
+
depth / point cloud
```

Take every masked pixel:

```text
pixel (u,v)
depth z
camera intrinsics K
camera pose T
```

and back-project:

```text
X_camera = z * K^-1 [u,v,1]
X_world  = T * X_camera
```

Now the defect becomes a 3D subset of the scene.

Conceptually:

```text
2D mask
  |
  v
back-projection
  |
  v
3D defect points
```

This is the core bridge between computer vision and 3D inspection.

---

# 16. What measurements you can derive

Depending on geometry quality, you can estimate:

### Surface-level defects

- defect area
- perimeter
- length
- width
- orientation
- location

### 3D defects

- estimated depth
- cavity volume
- maximum depression depth
- protrusion height
- surface roughness
- local surface-normal change

For example:

```text
Defect ID: D-018
Type: hole

2D area:       0.043 m²
3D surface area: 0.047 m²
max observed depth: 0.071 m
estimated volume: 0.0012 m³
confidence: 0.82
```

But these numbers should only be treated as measurement-grade after validation against real measurements.

---

# 17. Very important: holes are harder than peeling paint

Different defect types have different geometry difficulty.

## Category A — mostly 2D

Examples:

- peeling paint
- stains
- discoloration

RGB segmentation may be more important than 3D.

## Category B — surface geometry

Examples:

- cracks
- bulges
- shallow depressions
- spalling

Depth and surface normals become important.

## Category C — true cavities

Examples:

- roof holes
- ceiling holes
- deep missing material

These are the hardest.

The camera must actually see enough of the cavity.

If the camera sees only:

```text
████████████████
██████    ██████
██████    ██████
████████████████
```

but cannot see inside:

```text
        ↓ invisible interior
        ?
        ?
        ?
```

no vision-only model can guarantee the true depth.

This should be reflected in your confidence score.

---

# 18. Capture strategy

If you have any control over how photographs are taken, this is one of the highest-impact improvements.

Do not simply take:

```text
Image 1: random
Image 2: random
Image 3: random
```

Instead, aim for:

```text
          camera 2
              \
               \
                X  defect
               /
              /
         camera 1

             +

          camera 3
            /
```

The three images should:

- all see the defect
- have substantial overlap
- have different camera positions, not only rotation
- preserve similar lighting where practical

COLMAP's official documentation also recommends high visual overlap and different viewpoints for successful reconstruction.

---

# 19. What if the images do not overlap?

This is a major branch in the design.

Suppose:

```text
Image A = roof
Image B = different wall
Image C = ceiling
```

and there is no shared geometry.

Then do **not** expect a single globally accurate reconstruction.

Instead create:

```text
scene
 |
 +-- local reconstruction A
 |
 +-- local reconstruction B
 |
 +-- local reconstruction C
```

and relate them only if there is some external localization information.

This is much more honest and technically robust.

---

# 20. What if you only have one image?

The fallback should be:

```text
single image
    |
    +--> defect segmentation
    |
    +--> Depth Pro / Metric3D
    |
    +--> local surface normals
```

This gives you a useful 2.5D estimate.

Do not pretend it is the same as multi-view reconstruction.

---

# 21. What if you later get 5–10 images?

This is where the system gets substantially stronger.

Then use:

```text
5–10 overlapping images
        |
        v
DA3 / VGGT / MASt3R
        |
        v
camera poses + dense geometry
        |
        v
optional COLMAP refinement
        |
        v
dense point cloud / mesh
```

COLMAP becomes more valuable here because the increased overlap gives classical SfM/MVS more information.

---

# 22. Where COLMAP still fits

Do not remove COLMAP from the project.

Use it for:

### A. Baseline

Compare:

```text
COLMAP
vs
DA3
vs
VGGT
vs
MASt3R
```

### B. Pose refinement

If a neural model gives good initial camera poses, a classical optimization stage can refine them.

### C. More-image scenes

If the project later gets:

```text
10+
20+
50+
```

images, COLMAP becomes much more useful.

### D. Validation

If COLMAP and a learned model agree on geometry, that can be a useful consistency signal.

---

# 23. Recommended experiment matrix

Do not start with a huge production system.

Build an evaluation dataset first.

I recommend:

```text
100–300 real defects
```

with several categories.

For each defect, collect:

```text
1 view
2 views
3 views
5 views
10 views (where possible)
```

Then test:

```text
Experiment A
Depth Pro

Experiment B
COLMAP

Experiment C
DUSt3R

Experiment D
MASt3R

Experiment E
VGGT

Experiment F
DA3
```

---

# 24. Metrics to measure

Do not evaluate only "looks good."

Measure:

## Geometry

- absolute depth error
- relative depth error
- point-to-point error
- point-to-plane error
- camera pose error

## Defect

- precision
- recall
- IoU / mask quality
- defect localization error

## Measurement

- area error %
- width error %
- length error %
- depth error %
- volume error %

Example:

```text
Ground truth hole depth = 80 mm

DA3 estimate            = 74 mm
Depth Pro estimate      = 91 mm
COLMAP estimate         = 62 mm

absolute errors:

DA3        = 6 mm
Depth Pro  = 11 mm
COLMAP     = 18 mm
```

This is far more useful than saying:

> "DA3 point cloud looks better."

---

# 25. Recommended final ranking for your project

## Overall starting order

### 1. Depth Anything 3

**PRIMARY**

Use for:

- 2–3 image joint geometry
- sparse-view reconstruction
- camera pose
- depth
- point representation

Why:

- specifically designed for arbitrary-view geometry
- excellent current benchmark results
- directly matches your sparse-view setup
- released recently enough to be a serious first choice

---

### 2. VGGT

**PRIMARY ALTERNATIVE**

Use for:

- multi-view geometry
- camera pose
- depth
- point maps
- comparison against DA3

Why:

- designed explicitly for one/few/many views
- mature implementation
- strong 2025 results
- active development continued into 2026

---

### 3. MASt3R

**SPECIALIST ALTERNATIVE**

Use when:

- viewpoint changes are large
- matching is difficult
- correspondence quality matters

---

### 4. Depth Pro

**LOCAL / SINGLE-VIEW SUPPORT**

Use when:

- only one image is usable
- a local crop needs depth
- you care about sharp depth boundaries
- you need a monocular metric-depth estimate

---

### 5. COLMAP

**BASELINE / REFINEMENT / MORE-VIEWS PATH**

Use when:

- you have enough overlap
- you later collect more images
- you want classical SfM/MVS
- you want a sanity-check baseline

---

### 6. DUSt3R

**RESEARCH BASELINE**

Useful, but I would not make it the first implementation now.

---

# 26. Recommended first implementation

Build exactly this first:

```text
                  INPUT
             2–3 images
                  |
                  v
        +------------------+
        |   image quality  |
        |      check       |
        +--------+---------+
                 |
                 v
        +------------------+
        | defect segmenter |
        +--------+---------+
                 |
        +--------+---------+
        |                  |
        v                  v
       DA3             Depth Pro
   multi-view         local/monocular
        |                  |
        +--------+---------+
                 |
                 v
        3D defect extraction
                 |
                 v
       measurements + confidence
```

Do **not** start with:

```text
COLMAP
    ->
full house mesh
    ->
defect detection
```

That is the wrong order for your data regime.

---

# 27. First coding milestone

The first prototype should have only these inputs:

```text
/images/
    img1.jpg
    img2.jpg
    img3.jpg
```

and produce:

```text
/output/
    depth_1.npy
    depth_2.npy
    depth_3.npy

    points_3d.ply

    cameras.json

    confidence_1.npy
    confidence_2.npy
    confidence_3.npy
```

Then visualize the result in:

- Open3D
- MeshLab
- 3D viewer

Do not start with a full dashboard or web application.

First prove the geometry.

---

# 28. Second milestone

Add segmentation.

Input:

```text
images
```

Output:

```text
defect_masks
+
3D point cloud
```

Then visualize:

```text
              3D scene

                  X
               XXXXX
              XXXXXXX
               XXXXX
```

where X marks the defect points.

---

# 29. Third milestone

Add measurement.

For each defect:

```text
bbox_2d
mask_2d
3d_points
surface normals
confidence
```

calculate:

```text
area
length
width
depth
volume
```

where the geometry supports it.

---

# 30. Fourth milestone: benchmark the alternatives

Run exactly the same dataset through:

```text
DA3
VGGT
MASt3R
COLMAP
Depth Pro
```

Then make a table:

| Model | 2 views | 3 views | Large viewpoint change | Textureless surfaces | Defect depth | Runtime | GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA3 | | | | | | | |
| VGGT | | | | | | | |
| MASt3R | | | | | | | |
| COLMAP | | | | | | | |
| Depth Pro | | | | | | | |

Use your real defect dataset to fill it.

---

# 31. Commercial-use / licensing check

This matters if this becomes a product.

### VGGT

The official repository states that a commercial-use-friendly license is available, but **only the newer VGGT-1B-Commercial checkpoint is licensed for commercial usage; the original checkpoint remains non-commercial**.

Repository:
https://github.com/facebookresearch/vggt

### MASt3R / DUSt3R

Their repositories currently state:

```text
CC BY-NC-SA 4.0
```

which is non-commercial.

MASt3R repository:
https://github.com/naver/mast3r

DUSt3R repository:
https://github.com/naver/dust3r

### Important

Before putting any model into a commercial product, check the exact checkpoint and current license that you use. Model-code license and model-weight license may differ.

---

# 32. Recommended technology stack

For a practical Python prototype:

```text
Python
PyTorch
OpenCV
Open3D
NumPy
```

Geometry:

```text
Depth Anything 3
VGGT
MASt3R
COLMAP
```

Depth support:

```text
Depth Pro
Metric3D v2
```

Segmentation:

```text
your chosen defect segmentation model
```

Visualization:

```text
Open3D
MeshLab
```

Storage:

```text
PNG/JPEG
NPY/NPZ
PLY
JSON
```

---

# 33. Suggested data structure

Use a scene/defect structure instead of a flat image list.

Example:

```text
dataset/
|
+-- scene_0001/
|   |
|   +-- images/
|   |   +-- img_01.jpg
|   |   +-- img_02.jpg
|   |   +-- img_03.jpg
|   |
|   +-- defects/
|       +-- defect_01.json
|       +-- defect_02.json
|
+-- scene_0002/
|
+-- scene_0003/
```

A defect annotation could contain:

```json
{
  "defect_id": "scene_0001_defect_01",
  "type": "ceiling_hole",
  "images": [
    "img_01.jpg",
    "img_02.jpg",
    "img_03.jpg"
  ],
  "mask_paths": [
    "mask_01.png",
    "mask_02.png",
    "mask_03.png"
  ],
  "ground_truth": {
    "width_mm": 240,
    "height_mm": 180,
    "depth_mm": 75
  }
}
```

---

# 34. One change I strongly recommend to the requirements

Do not define the system as:

> "3D model of the house from 2–3 images."

Define it as:

> "Sparse multi-view 3D reconstruction and metric defect measurement from 1–3 inspection images."

That wording is technically much more realistic.

It also makes your engineering target measurable.

---

# 35. Final architecture I recommend

```text
                    ┌──────────────────────┐
                    │ 1–3 inspection imgs  │
                    └──────────┬───────────┘
                               |
                               v
                    ┌──────────────────────┐
                    │ Image quality /      │
                    │ overlap assessment   │
                    └──────────┬───────────┘
                               |
                    +----------+----------+
                    |                     |
                    v                     v
          ┌───────────────────┐   ┌──────────────────┐
          │ DA3 multi-view    │   │ Defect detector  │
          │ geometry          │   │ / segmenter      │
          └─────────┬─────────┘   └────────┬─────────┘
                    |                      |
                    v                      v
              point/depth             2D defect mask
                    |                      |
                    +----------+-----------+
                               |
                               v
                    ┌──────────────────────┐
                    │ Mask-to-3D fusion    │
                    └──────────┬───────────┘
                               |
                               v
                    ┌──────────────────────┐
                    │ Local defect geometry│
                    └──────────┬───────────┘
                               |
                 +-------------+--------------+
                 |             |              |
                 v             v              v
              area          depth          volume
                 |             |              |
                 +-------------+--------------+
                               |
                               v
                     measurement confidence
```

And then, behind this primary pipeline:

```text
                 ┌──────────────┐
                 │ DA3          │  PRIMARY
                 ├──────────────┤
                 │ VGGT         │  benchmark
                 ├──────────────┤
                 │ MASt3R       │  benchmark
                 ├──────────────┤
                 │ COLMAP       │  classical baseline
                 ├──────────────┤
                 │ Depth Pro    │  monocular/local support
                 └──────────────┘
```

# 36. My decision in one sentence

**Start with DA3 + defect segmentation, use Depth Pro only as a local/single-view depth supplement, benchmark VGGT and MASt3R, and keep COLMAP as the classical baseline and future refinement path.**

This recommendation is based on the current 2026 landscape: DA3 is explicitly aimed at arbitrary-view geometry and reports stronger benchmark results than VGGT; VGGT is designed for one/few/many views and remains actively developed; COLMAP's own documentation emphasizes overlap and multiple observations of the same object, which makes it less suited to your sparse, irregular inspection images. [DA3 paper](https://arxiv.org/abs/2511.10647), [VGGT paper](https://arxiv.org/abs/2503.11651), [COLMAP docs](https://colmap.github.io/tutorial.html)

---

## Sources

- Depth Anything 3: https://arxiv.org/abs/2511.10647
- Depth Anything 3 GitHub: https://github.com/ByteDance-Seed/Depth-Anything-3
- VGGT paper: https://arxiv.org/abs/2503.11651
- VGGT GitHub: https://github.com/facebookresearch/vggt
- MASt3R paper: https://arxiv.org/abs/2406.09756
- MASt3R GitHub: https://github.com/naver/mast3r
- DUSt3R GitHub: https://github.com/naver/dust3r
- Depth Pro paper: https://arxiv.org/abs/2410.02073
- COLMAP documentation: https://colmap.github.io/tutorial.html

## Practical first choice

**Implement DA3 first.**

Then compare against VGGT on a small real dataset before building the rest of the production system.

The winner should be decided by **your defect-level metrics**, especially:

```text
3D defect localization
depth error
area error
volume error
multi-view consistency
failure rate on 2-view / 3-view cases
```

rather than by generic 3D benchmark scores alone.
