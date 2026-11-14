"""Task A acceptance test: on every photo that HAS a ruler, ignore the ruler,
infer scale from the reference object, and compare with the ruler's scale.

Both scales are taken at the same spot (the ruler's centre), so perspective
doesn't make them differ. The ruler is masked out so it can't help the
reference fit.

    python reference_acceptance.py      -> output/REFERENCE_ACCEPTANCE.md + .json
                                           output/reference_check/<defect>_<shot>.jpg

Targets from the brief: within 10% on repeating patterns, 15% on single
objects. Photos with no reference are listed and counted, not dropped.
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CARD_LENGTH_CM, OUTPUT_DIR
from measure import card as cardmod
from measure import reference
from measure.reference_sizes import REFERENCE_SIZES
from run import load_shot

HERE = os.path.dirname(os.path.abspath(__file__))


def load_ref_prompts(defect):
    p = os.path.join(HERE, "prompts_reference", f"{defect}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    rows = []
    os.makedirs(os.path.join(OUTPUT_DIR, "reference_check"), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "[0-9][0-9].json"))):
        rec = json.load(open(f))
        if rec.get("scale", {}).get("source") != "card":
            continue
        d, s = rec["defect"], rec["shot"]
        _, img, _, focal, _ = load_shot(d, s)
        card, _ = cardmod.detect_card(img, focal)
        ruler_ppc = card["length_px"] / CARD_LENGTH_CM
        row = {"photo": f"{d}_{s}", "ruler_px_per_cm": round(ruler_ppc, 2)}
        prompt = load_ref_prompts(d).get(s)
        if prompt is None:
            row.update(found=False, reason="no standard-size object visible (none prompted)")
            rows.append(row)
            continue
        ref = reference.reference_scale(img, prompt, card["center"], exclude=cardmod.card_mask(img.shape, card, 10))
        row.update(object=prompt["object"], kind=REFERENCE_SIZES[prompt["object"]]["kind"], found=ref["found"])
        if ref["found"]:
            diff = 100 * (ref["px_per_cm"] / ruler_ppc - 1)
            target = 10 if row["kind"] == "repeating" else 15
            row.update(ref_px_per_cm=round(ref["px_per_cm"], 2), diff_pct=round(diff, 1),
                       within_target=bool(abs(diff) <= target), target_pct=target,
                       n_repeats=ref["n_repeats"], confidence=ref["confidence"], notes=ref["notes"],
                       implied_object_cm=round(ref.get("spacing_px", ref.get("width_px", 0)) / ruler_ppc, 2))
        else:
            row["reason"] = ref.get("reason")
        o = reference.draw(img.copy(), ref)
        x0, y0, x1, y1 = map(int, prompt["box"])
        import cv2
        cv2.rectangle(o, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.polylines(o, [np.int32(card["rect_quad"])], True, (0, 120, 255), 3)
        Image.fromarray(o).resize((960, 498)).save(os.path.join(OUTPUT_DIR, "reference_check", f"{d}_{s}.jpg"), quality=85)
        rows.append(row)

    found = [r for r in rows if r.get("found")]
    ok = [r for r in found if r.get("within_target")]
    L = ["# Reference-object scale - acceptance test on the ruler photos", "",
         "Ruler ignored; scale inferred from a standard-size building part at the ruler's position, then",
         "compared with the ruler's own scale (ruler = 17.0 cm body, still awaiting a tape check).", "",
         f"- Ruler photos: **{len(rows)}**",
         f"- Reference found: **{len(found)}**; no reference found / visible: **{len(rows) - len(found)}**",
         f"- Within target (10% repeating, 15% single object): **{len(ok)} of {len(found)}**", "",
         "| Photo | Reference | Found | Inferred px/cm | Ruler px/cm | Difference | Target met | Repeats | Implied size (by ruler) | Notes |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("found"):
            L.append(f"| {r['photo']} | {r['object']} | yes | {r['ref_px_per_cm']} | {r['ruler_px_per_cm']} | "
                     f"{r['diff_pct']:+.1f}% | {'yes' if r['within_target'] else 'no'} | {r['n_repeats']} | "
                     f"{r['implied_object_cm']} cm | {'; '.join(r['notes']) or '-'} |")
        else:
            L.append(f"| {r['photo']} | {r.get('object', '-')} | no | - | {r['ruler_px_per_cm']} | - | - | - | - | {r.get('reason', '')} |")
    # ---- the photos with NO ruler: what the fallback gave them (run.py output) ----
    L += ["", "## Photos with no usable ruler - reference-object fallback (numbers marked inferred)", "",
          "| Photo | Reference | Result | px/cm | Confidence | Key number | Notes / reason |", "|---|---|---|---|---|---|---|"]
    n_inf = n_none = 0
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "[0-9][0-9].json"))):
        rec = json.load(open(f))
        sc = rec.get("scale", {})
        if sc.get("source") == "card":
            continue
        ref = sc.get("reference")
        M = rec.get("measurements", {})
        key = ""
        if "length" in M:
            key = f"length {M['length']['value']} cm +-{M['length'].get('uncertainty_pct')}%"
        elif "area" in M:
            key = f"area {M['area']['value']} cm2 +-{M['area'].get('uncertainty_pct')}%"
        name = f"{rec['defect']}_{rec['shot']}"
        if rec.get("error"):
            L.append(f"| {name} | - | outline failed | - | - | - | {rec['error']} |")
            n_none += 1
        elif ref and ref.get("found"):
            n_inf += 1
            L.append(f"| {name} | {ref['object']} | **inferred** | {ref['px_per_cm']:.2f} | {ref['confidence']} | {key} | "
                     f"{'; '.join(ref.get('notes') or []) or '-'} |")
        elif ref:
            n_none += 1
            L.append(f"| {name} | {ref['object']} | not found | - | - | - | {ref.get('reason')} |")
        else:
            n_none += 1
            why = "angle only - no scale needed" if "angle" in M else "no standard-size object visible"
            L.append(f"| {name} | - | none | - | - | - | {why} |")
    L.insert(8, f"- No-ruler photos: **{n_inf + n_none}**; scale inferred from a reference: **{n_inf}**; none: **{n_none}**")
    open(os.path.join(OUTPUT_DIR, "REFERENCE_ACCEPTANCE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(rows, open(os.path.join(OUTPUT_DIR, "reference_acceptance.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
