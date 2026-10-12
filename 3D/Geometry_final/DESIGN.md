# Geometry_final — Design (draft, will be updated once we've run this on the pod)

## What changed from Geometry_third

Geometry_third proved DA3 + SAM2 works (real camera positions instead of guessed
ones), but two things were still manual and one thing was arguably redundant:

1. **Box prompts were hand-drawn.** Every defect's `prompts/<id>.json` was made
   by us looking at each photo and typing pixel coordinates. Slow, and a real
   source of past errors (e.g. `roof__shot_03`'s bad 0.213 SAM2 score traced
   back to a loosely-drawn box).
2. **Nobody checked SAM2's actual output.** We only found bad masks by manually
   opening overlay images after the fact.
3. **Two depth models (DA3 + Depth Pro) were doing overlapping jobs.** Useful
   as a cross-check, but it's two models, two checkpoints, two load times, for
   one signal. Decision this round: **keep one model per job, not two for
   redundancy.** DA3 is the geometry engine. Depth Pro is dropped entirely.

This version adds a VLM (**Qwen3-VL-4B-Instruct**, running locally on the
pod - not a paid API, unlike the Claude-vision option we never actually turned
on in the last two folders) to do the two things a human was doing by hand:
draw the box, and check the mask. It also finally activates the
"interpretation" stage that's existed in every folder so far but never ran,
since it's now the same VLM already loaded for the other two jobs.

**Checkpoint size update (2026-10-04, found live on the pod):** originally
planned as Qwen3-VL-30B-A3B-Instruct (the practical "best" pick on paper -
see the chat history for the 235B-flagship-vs-30B-MoE reasoning). Two real
download attempts both failed partway through with `Disk quota exceeded`:
this pod's `/workspace` has a genuine, hard ~40GB storage quota, separate
from the "195TB free" `df` reports (that's the whole shared cluster, not
this pod's actual allowance). Freeing ~28GB of old conda envs and leftover
per-defect folders from earlier sessions wasn't enough headroom for a fresh
60GB download on top of our ~26GB baseline (conda env + checkpoints).
Dropped to **Qwen3-VL-4B-Instruct** (~8GB on disk) instead, which downloaded
and loaded in under 10 seconds with the same baseline untouched. Still a
real grounding-capable VLM (the whole Qwen3-VL family is trained on 2D
grounding at every size, not just the largest) - if its auto-box/mask-check
quality proves too weak in practice, the honest next lever is asking for
more storage on this pod/volume, not silently living with a worse model.

## The three models, one job each

| Model | Job | Why this one |
|---|---|---|
| **Qwen3-VL-4B-Instruct** | Read each photo, draw the defect's box, sanity-check SAM2's mask, describe severity at the end | Apache 2.0 (commercial-safe), runs locally on the pod - no API key dependency, which is what killed the Claude-vision stage in both earlier folders. Downsized from the originally-planned 30B-A3B MoE variant after hitting this pod's real ~40GB storage quota live - see the update note above |
| **SAM2** | Turn the VLM's box into a precise pixel mask | Unchanged from Geometry_Second/third - already proven, box-prompt code doesn't care where the box came from |
| **DA3 (DA3-BASE)** | All photos of one defect, together: depth + real camera position | Unchanged reasoning from Geometry_third - the only one of DA3/VGGT/COLMAP/Depth Pro with no hard photo-overlap minimum, and it actually computes camera position instead of guessing or needing a second model to cross-check it |

No Claude, no Depth Pro, no COLMAP in the default path. (COLMAP stays available
as a completely separate, optional thing you could run by hand on a
well-covered defect set - not part of this pipeline.)

## Full flow

```
photos of one defect
        |
        v
[NEW] Qwen3-VL: quality gate + auto-box, per photo
        |  -> usable? / box [x0,y0,x1,y1] / defect type guess
        v
SAM2: turn each box into a precise mask         (unchanged)
        |
        v
[NEW] Qwen3-VL: does this mask actually match the described defect?
        |  -> flags bad masks automatically, same job we did by hand before
        v
DA3: all photos together -> depth + real camera position for each   (unchanged)
        |
        v
ruler-in-frame scale correction                  (unchanged - still mandatory,
        |                                          DA3's own "metric" claim is
        |                                          just as unverified as Depth
        |                                          Pro's was)
        v
back-project masked pixels -> merge into one point cloud   (unchanged, saved
        |                                                    to disk immediately)
        v
fit reference plane -> measure (depth/width/length/area/volume/tilt)  (unchanged)
        |
        v
mesh (best-effort, alpha-shape)                  (unchanged, disposable)
        |
        v
[NOW ACTUALLY RUNS] Qwen3-VL: plain-English severity description
```

## Rollout safety net

Auto-boxing from a VLM is new and unproven on our actual photos. So:

- If `prompts/<defect_id>.json` exists (same format as before), **use it** -
  manual boxes still work exactly like Geometry_third, no behavior change.
- If it doesn't exist, fall back to Qwen's auto-box.
- Either way, Qwen's own quality-gate and mask-verification stages still run,
  so we get the new confidence signals regardless of where the box came from.

This means we can re-run the 8 defects we already have manual boxes for,
compare Qwen's boxes against our hand-drawn ones on the same photos, and only
trust auto-boxing generally once that comparison looks right - rather than
assuming a brand-new model call is correct on the first try.

## Known risks / trade-offs (being upfront before we run anything)

1. **Losing the two-model cross-check.** Geometry_third's most honest finding
   (DA3 vs Depth Pro disagreeing by 16%-529% depending on the photo) came from
   having two independent depth guesses. With only DA3, we lose that specific
   signal. We still have DA3's own per-pixel confidence map (already saved as
   `mean_conf_per_shot`), just not a second, independent opinion.
2. **Qwen3-VL needs transformers built from source** (`pip install
   git+https://github.com/huggingface/transformers` - not yet in a stable
   release as of this writing). That's the same class of risk as xformers was
   for DA3: a bleeding-edge install that could disturb the already-working
   torch/SAM2/DA3 install in the `geo2` env. Setup script protects this the
   same way we protected torch before - snapshot versions, install with
   `--no-deps` where possible, verify everything still imports afterward.
3. ~~**~60GB checkpoint download**~~ - moot now, see the checkpoint-size
   update above; 4B-Instruct is ~8GB and downloads in seconds. Kept here as a
   record of why the checkpoint choice changed, not as an open risk anymore.
   on the first setup run.
4. **Auto-box quality is unverified until we actually run it** - see rollout
   safety net above. Don't trust it blindly just because a bigger model is
   involved.

## What's reused unchanged from Geometry_third

`common.py`, `gravity.py`, `geometry_da3.py`, `segment.py`, `scale.py`,
`fusion.py`, `plane.py`, `measure.py`, `mesh_export.py` (alpha-shape version),
`interpret.py`'s rule-based severity thresholds. None of these cared which
model produced the boxes or the points; that's exactly why they carry over.

## What's new

`pipeline/vlm_qwen.py` - loads Qwen3-VL once, exposes three calls: quality
gate + auto-box, mask verification, and severity description. `run_defect.py`
rewritten to wire these in at the right points and drop the Depth Pro stage
entirely.
