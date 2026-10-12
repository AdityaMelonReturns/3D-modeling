# Qwen review — all defects

## Summary

| Defect | Photos | Passed V1 (went to SAM2 + DA3) | Masks Qwen trusted (V2) | Mean IoU vs hand-drawn boxes | Rule-based severity | Qwen severity | Qwen: numbers plausible? | Max depth (mm) |
|---|---|---|---|---|---|---|---|---|
| hole | 6 | 6 | 3 of 6 | n/a (manual boxes used) | severe | low | False | 826 |
| ceiling | 9 | 9 | 8 of 9 | 0.48 | severe | severe | False | 305 |
| roof | 6 | 3 | 0 of 3 | 0.05 | severe | moderate | False | 532 |
| gaps | 8 | 4 | 3 of 4 | 0.00 | low | low | False | - |
| invlined_railing | 7 | 7 | 2 of 7 | 0.47 | severe | moderate | False | 923 |
| peeling_paint | 11 | 11 | 5 of 11 | 0.33 | severe | moderate | True | 618 |
| rotten_wood | 20 | 16 | 1 of 16 | 0.12 | severe | severe | False | 852 |
| wood1 | 6 | 5 | 2 of 5 | 0.10 | severe | moderate | True | 543 |

## hole

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| hole__shot_01 | yes | manual | - | NO | The red overlay does not accurately delineate a hole or missing material; it appears to mark a paint smear or stain. |
| hole__shot_02 | yes | manual | - | NO | The red region appears to mark a non-defect object or texture rather than a structural defect. |
| hole__shot_03 | yes | manual | - | yes | The red overlay accurately highlights the hole in the wall. |
| hole__shot_04 | yes | manual | - | yes | The red overlay precisely outlines the hole in the wall, indicating a missing material defect. |
| hole__shot_05 | yes | manual | - | yes | The red overlay accurately highlights a hole in the wall, consistent with a missing material defect. |
| hole__shot_06 | yes | manual | - | NO | The red overlay does not accurately delineate a hole or missing material; it appears to mark a small, irregular paint smudge or imperfection. |

**Qwen's final verdict:** severity **low** (rule-based math said **severe**), measurement plausible: **False**

> A small, irregularly shaped hole in a textured wall with minor damage around the edges.

> Notes: The measurements are implausible: the depth (max 825.99 mm) and volume (34,832,426 mm³) are far too large for a typical wall hole; the length (1151.79 mm) exceeds the visible wall area, suggesting the model misinterpreted the scene or the data is corrupted.

## ceiling

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| ceiling__shot_01 | yes | qwen | 0.58 | NO | The red overlay does not tightly match a defect; it appears to mark a non-defective area or an unrelated object. |
| ceiling__shot_02 | yes | qwen | 0.50 | yes | The red overlay precisely outlines the missing wall material, indicating a defect consistent with the description. |
| ceiling__shot_03 | yes | qwen | 0.63 | yes | The red overlay accurately outlines the missing plaster around the defect. |
| ceiling__shot_04 | yes | qwen | 0.64 | yes | The red overlay accurately outlines the missing plaster area. |
| ceiling__shot_05 | yes | qwen | 0.89 | yes | The red overlay accurately outlines the missing plaster area. |
| ceiling__shot_06 | yes | qwen | 0.54 | yes | The red overlay accurately outlines the damaged area where plaster is missing. |
| ceiling__shot_07 | yes | qwen | 0.18 | yes | The red overlay accurately outlines the missing plaster area. |
| ceiling__shot_08 | yes | qwen | 0.19 | yes | The red overlay accurately outlines the damaged area where material is missing from the wall. |
| ceiling__shot_09 | yes | qwen | 0.21 | yes | The red overlay precisely outlines the missing plaster area, matching the defect. |

**Qwen's final verdict:** severity **severe** (rule-based math said **severe**), measurement plausible: **False**

> A large, jagged hole in the ceiling with exposed interior structure and damaged drywall.

> Notes: The measured area and volume are implausibly large for a ceiling defect of this nature; the width and length measurements appear inconsistent with the visual scale of the hole, suggesting potential errors in depth or geometry estimation.

## roof

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| roof__shot_01 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| roof__shot_02 | NO | qwen | - | - | The photo shows a ruler and shadow, not a crack or gap. |
| roof__shot_03 | yes | qwen | 0.00 | NO | The red overlay does not tightly follow the crack's edge. |
| roof__shot_04 | yes | qwen | 0.06 | NO | The red line does not tightly follow a crack or gap but appears to be a scribble or marker line. |
| roof__shot_05 | yes | qwen | 0.08 | NO | The red region covers a shadow and background, not a crack or gap. |
| roof__shot_06 | NO | qwen | - | - | The defect is not a crack or gap but a peeling paint edge. |

**Qwen's final verdict:** severity **moderate** (rule-based math said **severe**), measurement plausible: **False**

> The roof has a noticeable irregularity with a measured depth of 531mm and an inclination of 69.4 degrees, indicating a significant structural issue.

> Notes: The measured depth of 531mm is implausible given the visible scale of the ruler (approx. 200mm) and the shadow cast; the volume and area estimates are likely inflated due to 3D reconstruction artifacts.

## gaps

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| gaps__shot_01 | yes | qwen | 0.00 | NO | The red dot marks a small, non-defective surface feature rather than a crack or gap. |
| gaps__shot_02 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| gaps__shot_03 | NO | qwen | - | - | The image shows a ruler and roof structure, not a crack or gap. |
| gaps__shot_04 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| gaps__shot_05 | yes | qwen | 0.01 | yes | The red overlay precisely highlights a visible crack in the wooden surface. |
| gaps__shot_06 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| gaps__shot_07 | yes | qwen | 0.00 | yes | The red overlay accurately highlights a visible crack on the surface. |
| gaps__shot_08 | yes | qwen | 0.00 | yes | The red overlay precisely marks a small, visible crack on the siding surface. |

**Qwen's final verdict:** severity **low** (rule-based math said **low**), measurement plausible: **False**

> Small gaps are visible between the siding panels of the house's exterior wall.

> Notes: Measurement data is unreliable due to low confidence and insufficient point count; visual inspection shows minor gaps.

## invlined_railing

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| invlined_railing__shot_01 | yes | qwen | 0.49 | yes | The red overlay accurately highlights the tilted railing structure. |
| invlined_railing__shot_02 | yes | qwen | 0.54 | NO | The red region covers a structure that is not leaning or tilted; it appears upright and properly aligned. |
| invlined_railing__shot_03 | yes | qwen | 0.62 | NO | The red overlay covers a structural beam that is not visibly leaning or tilted. |
| invlined_railing__shot_04 | yes | qwen | 0.51 | NO | The red region covers a railing that is not visibly leaning or tilted. |
| invlined_railing__shot_05 | yes | qwen | 0.31 | yes | The red overlay accurately highlights the tilted railing structure. |
| invlined_railing__shot_06 | yes | qwen | 0.25 | NO | The red region covers a structural element that is not visibly leaning or tilted. |
| invlined_railing__shot_07 | yes | qwen | 0.53 | NO | The red region highlights a railing, but it is not visibly leaning or tilted; it appears structurally upright. |

**Qwen's final verdict:** severity **moderate** (rule-based math said **severe**), measurement plausible: **False**

> The railing is visibly tilted and misaligned, indicating a structural defect.

> Notes: The measured inclination of 66.5 degrees is implausible for a railing; it appears to be more of a horizontal or near-horizontal tilt, suggesting potential measurement error or misalignment in the 3D reconstruction.

## peeling_paint

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| peeling_paint__shot_01 | yes | qwen | 0.31 | NO | The red overlay covers a painted surface with chipped paint, not a structural defect like a hole or decay. |
| peeling_paint__shot_02 | yes | qwen | 0.27 | NO | The red overlay does not tightly match a visible defect like a hole or decay; it appears to mark a painted surface with chipping or wear. |
| peeling_paint__shot_03 | yes | qwen | 0.37 | NO | The red overlay covers a painted surface with peeling paint, not a structural defect like a hole or decay. |
| peeling_paint__shot_04 | yes | qwen | 0.67 | NO | The red overlay covers a painted surface with peeling paint, not a structural defect like a hole or decay. |
| peeling_paint__shot_05 | yes | qwen | - | NO | The red region appears to mark a painted surface rather than a structural defect. |
| peeling_paint__shot_06 | yes | qwen | 0.08 | yes | The red overlay precisely outlines the damaged area where wood is missing or decayed. |
| peeling_paint__shot_07 | yes | qwen | 0.15 | NO | The red overlay does not tightly match a hole or decayed material; it appears to cover a small, irregular mark that may be paint or a scratch. |
| peeling_paint__shot_08 | yes | qwen | 0.20 | yes | The red overlay accurately highlights the area of missing material and peeling paint, indicating decay or damage. |
| peeling_paint__shot_09 | yes | qwen | 0.42 | yes | The red overlay accurately highlights the damaged, missing material area. |
| peeling_paint__shot_10 | yes | qwen | 0.64 | yes | The red overlay accurately highlights a visible area of missing or decayed material on the siding. |
| peeling_paint__shot_11 | yes | qwen | 0.13 | yes | The red overlay accurately highlights a missing section of material consistent with decay or damage. |

**Qwen's final verdict:** severity **moderate** (rule-based math said **severe**), measurement plausible: **True**

> The paint is peeling off the wooden surface in a large area, revealing the underlying material.

> Notes: 

## rotten_wood

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| rotten_wood__shot_01 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| rotten_wood__shot_02 | yes | qwen | - | NO | The red overlay does not tightly match a visible defect like a hole or decayed material; it appears to cover a painted surface. |
| rotten_wood__shot_03 | yes | qwen | - | yes | The red overlay accurately highlights a section of decayed wood with visible rot and missing material. |
| rotten_wood__shot_04 | yes | qwen | 0.04 | NO | The red overlay does not tightly match a visible defect like a hole or decay; it appears to cover a non-defective area of the railing. |
| rotten_wood__shot_05 | yes | qwen | - | NO | The red overlay covers a non-defective area of the structure. |
| rotten_wood__shot_06 | yes | qwen | 0.26 | NO | The red overlay does not accurately delineate a defect; it appears to mark a painted surface rather than a structural issue. |
| rotten_wood__shot_07 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| rotten_wood__shot_08 | yes | qwen | 0.09 | NO | The red region covers a painted surface without indicating a visible defect like a hole or decay. |
| rotten_wood__shot_09 | yes | qwen | 0.16 | NO | The red overlay does not tightly match a visible defect like a hole or decay; it covers painted wood without clear evidence of damage. |
| rotten_wood__shot_10 | yes | qwen | 0.12 | NO | The red overlay includes non-defective areas and does not tightly match the actual defect. |
| rotten_wood__shot_11 | yes | qwen | 0.11 | NO | The red overlay covers a large area of painted wood without indicating a specific defect like a hole or decay. |
| rotten_wood__shot_12 | yes | qwen | - | NO | The red region marks a painted area, not a structural defect like a hole or decay. |
| rotten_wood__shot_13 | yes | qwen | - | NO | The red overlay covers a large area of painted wood without indicating a specific defect like a hole or decay. |
| rotten_wood__shot_14 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| rotten_wood__shot_15 | yes | qwen | 0.06 | NO | The red overlay covers a large area of painted wood without indicating a specific defect like a hole or decay. |
| rotten_wood__shot_16 | yes | qwen | - | NO | The red overlay covers a large area of painted wood without indicating any actual defect like a hole or decay. |
| rotten_wood__shot_17 | yes | qwen | - | NO | The red overlay covers a painted surface without indicating any actual defect. |
| rotten_wood__shot_18 | yes | qwen | - | NO | The red overlay covers a painted surface, not a structural defect like rot or missing material. |
| rotten_wood__shot_19 | NO | qwen | - | - | degenerate box from Qwen: [0, 0, 0, 0] |
| rotten_wood__shot_20 | yes | qwen | - | NO | The red overlay covers a painted surface without clearly indicating a structural defect. |

**Qwen's final verdict:** severity **severe** (rule-based math said **severe**), measurement plausible: **False**

> The wooden fence section is significantly damaged with large, rotting areas that appear structurally compromised.

> Notes: The depth measurement of 851.7 mm is implausible for a fence defect in this context; it likely exceeds the actual physical depth of the rot. The scale from the ruler appears to be used for calibration, but the depth value is inconsistent with typical fence wood thickness and defect scale.

## wood1

| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |
|---|---|---|---|---|---|
| wood1__shot_01 | NO | qwen | - | - | No clear defect such as hole, crack, peeling paint, or rot is visible in the image. |
| wood1__shot_02 | yes | qwen | 0.14 | NO | The red region covers a painted surface with minor wear, not a structural defect like a hole or decay. |
| wood1__shot_03 | yes | qwen | 0.03 | NO | The red region covers a non-defective area of the wood surface. |
| wood1__shot_04 | yes | qwen | 0.00 | yes | The red region precisely outlines a visible area of decayed wood with missing material. |
| wood1__shot_05 | yes | qwen | 0.02 | yes | The red overlay precisely highlights a large, dark, deteriorated area consistent with rotten wood. |
| wood1__shot_06 | yes | qwen | 0.29 | NO | The red region appears to mark a paint stain or discoloration, not a structural defect like rot or missing material. |

**Qwen's final verdict:** severity **moderate** (rule-based math said **severe**), measurement plausible: **True**

> A long, cracked red wooden surface with peeling paint and visible wood grain damage.

> Notes: 
