# Geometry Prototype — Defect 3D from Phone Captures

**Goal:** Given multi-angle phone images + sensor sidecars, how accurately can we recover defect geometry (depth, width, extent) and surface inclination — without LiDAR? Find what works, what fails, and the thresholds.

**What you're getting:** Zip from our capture prototype app. ~8–9 defect sets (rotten wood, peeling paint, hole, broken drywall, inclined surface, gap). Each set has 4–9 JPEGs from different angles + a sidecar JSON per image. Many shots include a scale card in frame. You'll get access to a RunPod pod with GPU.

**Sensor / metadata fields in each sidecar JSON** (verify against the actual files — trust the data over this list):

- Orientation: pitch / roll / yaw from DeviceMotion (fused accel + gyro). Note: yaw is relative, not north-referenced — use heading for world azimuth
- Raw accelerometer vector (gravity direction — this is your gravity reference for inclination)
- Compass heading from Location: true heading + magnetic heading + accuracy estimate
- Raw magnetometer vector
- GPS: latitude, longitude, altitude (coarse, ±5–15 m — a prior, not survey-grade)
- Barometric relative altitude (useful for floor-to-floor deltas)
- Camera: focal length (from EXIF), device model (for intrinsics lookup)
- Timestamp per capture

## Approach (suggested, not prescriptive)

1. **Reconstruct:** COLMAP per defect set (feature match → sparse → dense). Use the sensor priors for initialization if helpful. If COLMAP struggles on a set, try alternatives (Meshroom, or monocular depth like MiDaS / Depth Pro as a fallback for sets that won't reconstruct).
2. **Scale:** Detect the scale card → pin metric units. Compare: how much worse are measurements on sets/shots without the card?
3. **Measure:** Per defect — depth (hole, rot), width/extent (crack, gap, peeling area), whatever's meaningful for that defect type. Spot-check a few against real measurements where possible.
4. **Inclination:** For the inclined-surface sets — fit a plane to the reconstruction, compare its normal to the gravity vector from sidecar pitch/roll. Report the estimated lean angle and your confidence.
5. **Diagnose:** Light pass — given the geometry + image, what would you call the defect and its severity? Doesn't need to be a model; reasoning is fine.

## What I expect to fail (find the threshold, don't fight it)

- The small gap — likely below reconstruction resolution. Tell me the size floor.
- Textureless surfaces (clean drywall) — COLMAP may not find matchable features.
- Inclination under ~2° — sensor noise floor.

## Things worth learning along the way

- Does 4 shots vs. 8 shots change reconstruction quality? What's the practical minimum?
- Does the scale card actually matter, and where should it sit in frame?
- Capture recommendations to feed back into the field protocol (angle spread, distance, lighting).

## Deliverable (light, startup mode — not a report)

- Working pipeline on the pod (repo or folder, runnable)
- Per defect set: worked / partial / failed, measured values vs. actual, one-line why
- Quick takeaways: minimum shots, scale card value, capture recommendations
- We'll sync in 3–4 days on progress and decide next steps from there

Use Claude Code freely. If something's ambiguous, make a call and note it — don't block on me.