"""Summarize Qwen's verdicts across all defects in output/ into one table:
V1 (which photos went on to SAM2/DA3), V2 (which masks it trusted), stage J
(severity + plausibility), and how Qwen's auto-boxes compare to the boxes we
drew by hand in Geometry_third/prompts/ for the same photos (IoU).

Usage: python summarize_qwen.py   -> writes QWEN_REVIEW.md next to this file
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANUAL_PROMPTS = os.path.join(HERE, "..", "Geometry_third", "prompts")
DEFECTS = ["hole", "ceiling", "roof", "gaps", "invlined_railing", "peeling_paint", "rotten_wood", "wood1"]


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def manual_box(defect, shot):
    path = os.path.join(MANUAL_PROMPTS, f"{defect}.json")
    if not os.path.exists(path):
        return None
    v = json.load(open(path)).get(shot)
    if v is None:
        return None
    return v["box"] if isinstance(v, dict) else v


def main():
    lines = ["# Qwen review — all defects", ""]
    summary_rows = []
    for d in DEFECTS:
        dbg_path = os.path.join(HERE, "output", d, "debug", "geometry_debug.json")
        if not os.path.exists(dbg_path):
            continue
        dbg = json.load(open(dbg_path))
        interp = json.load(open(os.path.join(HERE, "output", d, "interpretation.json")))
        meas = json.load(open(os.path.join(HERE, "output", d, "measurements.json")))
        v1, v2 = dbg.get("v1_quality_gate", {}), dbg.get("v2_mask_verification", {})
        qv = interp.get("qwen_vision") or {}

        lines += [f"## {d}", "",
                  "| Photo | V1: sent to SAM2/DA3? | Box source | IoU vs hand-drawn box | V2: mask trusted? | Qwen's reason (V2, else V1) |",
                  "|---|---|---|---|---|---|"]
        ious = []
        for shot in sorted(v1):
            a = v1[shot]
            used = bool(a.get("usable") and a.get("box"))
            src = a.get("source", "?")
            mb = manual_box(d, shot)
            iou_txt = "-"
            if used and src == "qwen" and mb:
                val = iou(a["box"], mb)
                ious.append(val)
                iou_txt = f"{val:.2f}"
            b = v2.get(shot)
            v2_txt = "-" if b is None else ("yes" if b.get("matches") else "NO")
            reason = (b or a).get("reason", "")
            lines.append(f"| {shot} | {'yes' if used else 'NO'} | {src} | {iou_txt} | {v2_txt} | {reason} |")

        n_total, n_used = len(v1), sum(1 for a in v1.values() if a.get("usable") and a.get("box"))
        n_ok = sum(1 for b in v2.values() if b.get("matches"))
        mean_iou = sum(ious) / len(ious) if ious else None
        summary_rows.append((d, n_total, n_used, n_ok, mean_iou, interp.get("rule_based_severity"),
                             qv.get("severity"), qv.get("measurement_plausible"),
                             (meas.get("depth_mm") or {}).get("max")))
        lines += ["", f"**Qwen's final verdict:** severity **{qv.get('severity')}** "
                      f"(rule-based math said **{interp.get('rule_based_severity')}**), "
                      f"measurement plausible: **{qv.get('measurement_plausible')}**", "",
                  f"> {qv.get('description', '')}", "",
                  f"> Notes: {qv.get('notes', '')}", ""]

    head = ["## Summary", "",
            "| Defect | Photos | Passed V1 (went to SAM2 + DA3) | Masks Qwen trusted (V2) | Mean IoU vs hand-drawn boxes | Rule-based severity | Qwen severity | Qwen: numbers plausible? | Max depth (mm) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        iou_txt = f"{r[4]:.2f}" if r[4] is not None else "n/a (manual boxes used)"
        depth = f"{r[8]:.0f}" if isinstance(r[8], (int, float)) else "-"
        head.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} of {r[2]} | {iou_txt} | {r[5]} | {r[6]} | {r[7]} | {depth} |")
    out = lines[:2] + head + [""] + lines[2:]
    with open(os.path.join(HERE, "QWEN_REVIEW.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(head))


if __name__ == "__main__":
    main()
