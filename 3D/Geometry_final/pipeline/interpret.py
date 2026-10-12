"""Stage [J], rule-based half only.

The vision half of stage [J] (defect-type confirmation + plain-language
severity) now runs through pipeline/vlm_qwen.py's describe_severity()
instead of a Claude API call - see DESIGN.md's "why one model per job".
This module just holds the pure-math severity thresholds, no VLM code, no
API key dependency at all.
"""

# Calibrate these against real measurements as you process more sets (spec
# Sec.J: "you'll calibrate these against your real measurements once you
# have a few sets processed" - these starting points are rough placeholders,
# not derived from data yet).
SEVERITY_THRESHOLDS = {
    "hole_or_rot": {"depth_mm": {"moderate": 5, "severe": 15}, "area_mm2": {"moderate": 2000, "severe": 8000}},
    "crack_or_gap": {"width_mm": {"moderate": 3, "severe": 10}, "length_mm": {"moderate": 100, "severe": 300}},
    "inclination": {"deg": {"moderate": 5, "severe": 15}},
}


def rule_based_severity(defect_type, measurements):
    """defect_type: one of SEVERITY_THRESHOLDS keys (caller picks based on
    which defect set this is - not auto-classified here)."""
    thresholds = SEVERITY_THRESHOLDS.get(defect_type, {})
    flags = []
    for metric, levels in thresholds.items():
        if metric == "deg":
            value = measurements.get("inclination_deg", {}).get("value")
        else:
            value = measurements.get(metric)
            if isinstance(value, dict):
                value = value.get("max")
        if value is None:
            continue
        if value >= levels.get("severe", float("inf")):
            flags.append(("severe", metric, value))
        elif value >= levels.get("moderate", float("inf")):
            flags.append(("moderate", metric, value))
    if not flags:
        return "low", flags
    order = {"low": 0, "moderate": 1, "severe": 2}
    worst = max(flags, key=lambda f: order[f[0]])
    return worst[0], flags
