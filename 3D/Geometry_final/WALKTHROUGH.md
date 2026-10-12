# Walkthrough — what happens, stage by stage

Written for: whoever picks this project up next (including future us) and
wants to know what a run actually does, in order, without reading the code
first. Matches the stage letters printed by `run_defect.py` while it runs.

## [A] Load the defect set

Reads every photo in `../Geometry/workspace/<defect_id>/images/`. If a
matching `metadata/<shot>.json` sidecar exists (the capture app's sensor
data - tilt, heading, accelerometer), reads that too, but it's optional -
photos with no sidecar still load fine, they just skip the tilt measurement
later (see `gravity.py`).

## [V1] Qwen quality gate + auto-box (NEW)

For each photo: send it to Qwen3-VL with a prompt describing the defect
type and ask for three things back, as JSON:
- `usable`: is the defect actually visible / photo not blurry or badly lit?
- `box`: `[x0, y0, x1, y1]` pixel box around the defect
- `defect_type_guess`: what kind of defect this looks like

If `prompts/<defect_id>.json` has a manual entry for this shot, that box is
used instead and Qwen's box is only logged for comparison, not used - see
DESIGN.md's rollout safety net. Photos Qwen marks unusable are skipped
before they ever reach SAM2 or DA3, the same way a shot with no box prompt
was already skipped in Geometry_third.

## [B+C] DA3 - depth + camera position

All usable photos of this one defect go in together, and DA3 works out a
depth map and a real camera position for every one of them jointly - not
one photo at a time, and not a phone-sensor guess. This is the fix that
Geometry_third proved out; unchanged here.

## [D] Scale calibration

Same ruler-detection logic as the last two folders: find the physical
scale-card in frame (if it's there, and not held at an angle), compare its
known 15cm length against what DA3's own depth says it measures, and use
that ratio to correct DA3's depth for this photo. DA3's raw depth is no more
trustworthy at face value than Depth Pro's was - this step is still
mandatory, not optional.

## [E] SAM2 segmentation

Turns each photo's box (from Qwen, or from the manual prompts file) into a
precise pixel mask of the defect. Unchanged code from Geometry_third.

## [V2] Qwen mask verification (NEW)

For each photo, render the SAM2 mask as a red overlay (same visual as
`render_masks.py` produced by hand in earlier folders) and ask Qwen: does
this actually match the defect described? Saves a yes/no + short reason per
photo to `debug/`. This is the automatic version of the manual overlay
inspection that took real time in the last two folders - it doesn't block
the pipeline (a "no" is logged as a confidence flag, not a hard stop), so a
bad mask still gets measured, just clearly marked as suspect.

## [F] Back-projection + fusion

Each photo's masked, scale-corrected depth becomes a set of 3D points
(using DA3's own camera position for that photo), and all photos' points
get merged into one point cloud. Colored-ICP refinement is attempted first;
if it fails (it has, on every defect run so far, in every folder), falls
back to the pose-based alignment directly. The merged point cloud is
written to disk right here, before anything riskier happens next.

## [G] Reference plane fit

Fits a flat plane to the non-defect "surrounding" points (the margin SAM2's
dilated mask captured around the core defect) - this plane is the stand-in
for "what this surface should look like without the damage."

## [H] Measurements

Depth, width, length, area, volume, and tilt, all measured against that
plane. Written to `measurements.json` immediately - before meshing, which
is the one stage in this whole pipeline that's ever actually failed/hung
outright (Open3D's Poisson reconstruction, in earlier folders). Losing the
mesh is disclosed and fine; losing the measurements would not be.

## [I] Mesh export (best-effort)

Alpha-shape reconstruction (not Poisson - see Geometry_third's
`mesh_export.py` docstring for why that switch happened) on the merged
point cloud. If this fails for a particular defect's point geometry, the
run still completes successfully; only the mesh file is missing.

## [J] Interpretation

Rule-based severity (unchanged thresholds from earlier folders) plus, for
the first time in this project, an actual VLM description: Qwen looks at
one representative photo plus the computed measurements and writes a
plain-English severity summary. Earlier folders had this exact code path
(`claude_vision_interpret`) sitting unused the whole project because no
API key was ever set - Qwen replaces it and actually runs, since it's the
same model already loaded locally for stages V1/V2, not a separate paid
API call.
