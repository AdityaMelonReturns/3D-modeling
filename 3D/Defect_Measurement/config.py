"""Everything you'd tune per job lives here: the scale card's physical size,
and one entry per defect set saying how to measure it.

Follows Defect_Measurement_Simplified.docx (the lead's one-pager): one photo
per defect is enough, 2D measurement on a rectified view, no COLMAP / DA3 /
fusion / mesh in the defect path.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
WORKSPACE = os.path.join(ROOT, "Geometry", "workspace")         # <defect>/images, <defect>/metadata
PROMPTS_DIR = os.path.join(HERE, "prompts")                       # hand-drawn boxes (YOLO later)
OUTPUT_DIR = os.path.join(HERE, "output")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = os.path.join(ROOT, "Geometry_Second", "checkpoints", "sam2", "sam2.1_hiera_large.pt")

# The blue "TEDDY BEAR" ruler. The blue body runs from the bear's overalls to
# just past the 15 cm tick, so it is NOT 15 cm long (the old pipeline assumed
# it was). Measured by reading the printed cm ticks (autocorrelation of the
# tick strip) on every ruler photo; the 9 photos where the ticks are clearly
# periodic give body length 17.0 cm (IQR 16.9-17.3) and width 2.1 cm.
# TODO: confirm with a tape measure on the physical ruler; every scaled number
# moves 1:1 with CARD_LENGTH_CM.
CARD_LENGTH_CM = 17.0
CARD_WIDTH_CM = 2.1
CARD_DIMS_SOURCE = "read from printed cm ticks in 9 clear photos (17.0 cm, IQR 16.9-17.3); confirm with tape"

# Flatness rule from the one-pager (needs LiDAR; we have none yet, so every
# result is "flat by surface type").
FLAT_OK, FLAT_APPROX = 0.10, 0.25

# Probe depth bands from the one-pager. Depth is never read from the image.
PROBE_BANDS = ["<1 cm", "1-3 cm", ">3 cm", "through"]

# Kept from Geometry_final/pipeline/interpret.py - need recalibrating against
# real inspector calls; they are here so every report shows a band, not as truth.
SEVERITY_THRESHOLDS = {
    "area":  {"area_cm2": {"moderate": 20, "severe": 80}},
    "crack": {"width_mm": {"moderate": 3, "severe": 10}, "length_cm": {"moderate": 10, "severe": 30}},
    "angle": {"lean_deg": {"moderate": 5, "severe": 15}},
}

# Per defect set:
#   edge      crisp | thin | diffuse   (one-pager "Edges" section)
#   measures  which numbers to report
#   severity  key into SEVERITY_THRESHOLDS
#   card_on_surface  True = card lies on / against the defect's surface,
#                    False = hand-held in the air (scale still reported, flagged)
#   primary   the one photo whose numbers are THE answer; other prompted
#             photos are measured too and only used as a consistency check
#   probe     worker's probe band (None = not probed yet)
#   pilot     the pilot's hand estimate, for the acceptance check
DEFECTS = {
    "ceiling": dict(edge="crisp", measures=["area", "length_width"], severity="area",
                    card_on_surface=True, primary="05", probe=None,
                    pilot=dict(metric="width_cm", lo=5, hi=7, text="hole ~5-7 cm across")),
    "roof": dict(edge="thin", measures=["length_width"], severity="crack",
                 card_on_surface=True, primary="03", probe=None,
                 pilot=dict(metric="length_cm", lo=25, hi=28, text="shingle crack ~25-28 cm long")),
    "hole": dict(edge="crisp", measures=["area", "length_width"], severity="area",
                 card_on_surface=None, primary="03", probe=None,
                 pilot=dict(metric="length_cm", lo=2, hi=3, text="~2-3 cm (visual guess, no ruler)")),
    "gaps": dict(edge="thin", measures=["length_width"], severity="crack",
                 card_on_surface=False, primary="02", probe=None, pilot=None),
    "invlined_railing": dict(edge="crisp", measures=["angle"], severity="angle",
                             card_on_surface=None, primary="02", probe=None,
                             pilot=dict(metric="lean_deg", lo=29, hi=36, text="lean 29-36 deg (3D plane fit, flagged unreliable)")),
    "peeling_paint": dict(edge="diffuse", measures=["area"], severity="area",
                          card_on_surface=True, primary="07", probe=None, pilot=None),
    "rotten_wood": dict(edge="crisp", measures=["area", "length_width"], severity="area",
                        card_on_surface=False, primary="09", probe=None, pilot=None),
    "wood1": dict(edge="crisp", measures=["area", "length_width"], severity="crack",
                  card_on_surface=True, primary="06", probe=None,
                  pilot=dict(metric="length_cm", lo=10, hi=15, text="~10-15 cm (ruler at an angle)")),
}
