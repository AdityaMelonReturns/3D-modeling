"""Stage [J]: Interpretation Layer.

Rule-based first pass (thresholds you calibrate once a few real sets are
processed) + optional Claude-vision pass for defect-type confirmation and
plain-language severity. The Claude call needs ANTHROPIC_API_KEY set in the
environment - if it isn't, this stage still runs (rule-based only) and
clearly flags that the vision confirmation was skipped, rather than failing
the whole pipeline over one optional stage.
"""
import base64
import json
import os

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


def claude_vision_interpret(image_path, measurements, defect_id, model="claude-sonnet-5"):
    """Sends one representative defect photo + the computed measurements to
    Claude for defect-type confirmation + plain-language description.
    Returns None (not a failure) if ANTHROPIC_API_KEY isn't set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    import anthropic
    client = anthropic.Anthropic()

    with open(image_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    media_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"

    prompt = f"""This is a photo of a house defect ("{defect_id}"). Measured geometry from a 3D reconstruction:
{json.dumps(measurements, indent=2)}

Given the photo and these measurements, respond with JSON only:
{{"defect_type": "...", "severity": "low|moderate|severe", "description": "one plain-English sentence",
"measurement_plausible": true/false, "notes": "anything the numbers get wrong vs what you see, if anything"}}"""

    resp = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = resp.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": text, "parse_error": True}
