# Defect 3D Prototype — Results Report

Plain-English summary of what we tried, what worked, what didn't, and what it means. Covers all 8 defect zips from the capture app.

---

## 1. The short version

- We turned every defect zip into a 3D model (sparse → dense point cloud → mesh). All 8 worked to some degree.
- 5 of the 8 sets included the little "Teddy Bear" ruler as a scale card. Where it was laid flat next to the defect, we could get a real cm-based size estimate. Where it was held up in the air at an angle, it wasn't usable for measurement.
- We got a lean-angle estimate for the inclined railing/pergola, but it's not trustworthy yet — explained below.
- The RunPod GPU box got stopped partway through, so one extra piece of work (see peeling_paint) is on pause until it's back up.

---

## 2. Per-defect results

| Set | Photos used | 3D model | Scale card in shot? | Size estimate | Confidence |
|---|---|---|---|---|---|
| roof | 6/6 | Good — clean mesh | Yes, flat on shingles | Shingle crack ≈ 25–28 cm long | Medium (ruler clean, crack edges estimated by eye) |
| rotten_wood | 15/20 | Good — best mesh of the 8 | Yes, but held up in hand at an angle | Not usable for measurement | Low (wrong angle for scaling) |
| ceiling | 9/9 | Good | Yes, right at the hole | Hole ≈ 5–7 cm across | Medium |
| gaps | 7/8 | Good | Yes, but in a different photo than the gap itself | Not directly measurable | Low (ruler and defect not in the same frame) |
| hole | 6/6 | Good | No ruler in any shot | Rough visual guess only, ≈ 2–3 cm | Very low |
| invlined_railing | 7/7 | Good | No ruler | Lean angle: 29–36° depending on method (see §4) | Low — needs better isolation of the structure |
| peeling_paint | 4/11 registered (+7 more sitting unused) | Partial | Yes, in the unused 7 | Not yet measured | **On hold** — GPU pod stopped before we could process the better half of this set |
| wood1 | 6/6 | Good | Yes, but at an angle | Rough only, ≈ 10–15 cm | Low-medium |

"Confidence" here is about the **measurement**, not the 3D model — all 8 models themselves came out fine (see the earlier mesh numbers already shared).

---

## 3. What we found about the scale card (ruler)

This was worth checking properly, and the answer is nuanced:

- **When laid flat, directly against the same surface as the defect, in the same photo** (roof, ceiling): the ruler gave a clean, trustworthy size reference. We could isolate it from the photo automatically (by its color) and get a solid pixel-to-cm conversion.
- **When held up in the air at an angle to the camera** (wood1, rotten_wood): the ruler appears "foreshortened" in the photo — its 2D length in the image no longer matches its real length, because it's tilted relative to the camera. This makes it unreliable for measurement, even though it's clearly visible.
- **When it's in the set but not in the same photo as the defect** (gaps, peeling_paint's usable shots): you can't directly compare pixel sizes between two different photos taken from different distances, so it doesn't help unless we also use the 3D camera positions to cross-check distances (possible in theory, not done here).
- **3 of the 8 sets had no ruler at all** (rotten_wood's defect shots, hole, invlined_railing) — for these, any size we quote is a rough visual guess, not a measurement.

**Bottom line: the scale card genuinely works, but only if it's laid flat, next to the defect, in the same shot.** That's a simple, fixable instruction for whoever is capturing these.

---

## 4. The railing lean angle — what we found, honestly

We did what the brief asked: fit a flat plane to the 3D points, and compare it to "true vertical" using the phone's accelerometer data.

Two real findings came out of this:

1. **Figuring out which way the phone's sensor axes line up with the 3D model's axes actually worked well.** We tested several possible ways the phone's motion sensor could be wired up relative to the camera, and one of them (a 90° rotation) matched consistently across all 7 photos in the set. That's a solid, reusable result.

2. **The plane fit itself is not reliable yet**, because the 3D point cloud for this set mixes together everything in view — the deck flooring, the house siding, bushes in the background, and the pergola/railing structure itself. Fitting "one flat surface" to all of that gives different answers depending on exactly which points get included: we got 29° one way and 36° another way, just by changing which points count. That 7° swing tells us the bottleneck isn't the math — it's that we haven't isolated *just* the railing/pergola points from the rest of the scene.

**What this means:** the lean-angle method works in principle, but needs the point cloud manually cropped (in a viewer like MeshLab or CloudCompare) to just the structure being measured before the angle number means anything. That's a quick manual step, not a redesign.

---

## 5. Capture recommendations (things to change going forward)

- **Lay the scale card flat, directly on or against the same surface as the defect, in the same photo.** Don't hold it up in the air at an angle — it stops being usable for measurement the moment it's tilted relative to the camera.
- **Take one scale-card photo per defect, framed so the card and the defect are both clearly visible together.** Several sets had the card in a "reference" shot that didn't include the actual defect — that breaks the measurement.
- **More photos only helps if they're spread out.** rotten_wood (20 photos, spread around the subject) reconstructed better than peeling_paint (11 photos, all very similar close-ups) — which only fully used 4 of them. A spread-out set of 6–8 photos beats a bunched-up set of 11.
- **6 well-spread photos is a workable minimum.** roof, hole, and wood1 all used every single photo successfully with just 6 shots each.
- For anything where "is it tilted/level" matters (like the railing), it would help a lot if the capture also included **one or two reference shots of something known to be flat/vertical nearby** (like the deck's edge board) — that would make isolating "what plane are we even measuring" much easier afterward.

---

## 6. What's incomplete right now

- **peeling_paint**: 11 photos went in, but only 4 connected into one 3D model — the other 7 (which include the scale-card photo) landed in a second, disconnected cluster that we started processing on the GPU, but the RunPod pod was stopped mid-way. This is the one piece of unfinished work — restarting the pod would let us finish it in a few minutes.
- **Railing lean angle**: needs the point cloud manually cropped to isolate the structure before the number is trustworthy (see §4).
- **Real ground-truth check**: we don't have independently-measured real-world dimensions for any of these defects to check our estimates against — everything above is checked against the in-photo ruler only, not a tape measure on the actual house.

---

## 7. Where everything lives

All under `3D/Geometry/`:
- `tools/` — COLMAP (local, CPU) + RunPod setup scripts
- `workspace/<defect-name>/` — per-defect: `images/`, `metadata/` (sensor JSON), `sparse/0/` (point cloud + poses), `dense/fused.ply` (dense point cloud), `mesh_dense/mesh_poisson.ply` (final mesh)
- `scripts/run_sparse_sfm.sh`, `run_dense_on_runpod.sh` — rerun the pipeline on any zip
