"""Measure defects from single photos (Defect_Measurement_Simplified.docx).

    python run.py                       # all 8 sets, every prompted photo
    python run.py roof ceiling          # just these sets
    python run.py --primary-only        # only each set's chosen photo

Per photo:  box (prompts/) -> outline (SAM2 / ridge filter / colour diff)
            -> card -> straight-on view -> numbers, each tagged with its
            scale source and confidence.
Writes output/<defect>/<shot>.json + <shot>_overlay.jpg, and
output/SUMMARY.md + summary.json across all sets.
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CARD_DIMS_SOURCE, CARD_LENGTH_CM, CARD_WIDTH_CM, DEFECTS, OUTPUT_DIR,
                    PROMPTS_DIR, SEVERITY_THRESHOLDS, WORKSPACE)
from measure import card as cardmod
from measure import edges, gravity, metrics, segment

CARD_DIMS_UNCERT_PCT = 3.0     # +-0.5 cm on 17 cm until the ruler is tape-measured


def load_shot(defect, shot):
    stem = f"{defect}__shot_{shot}"
    img = np.array(Image.open(os.path.join(WORKSPACE, defect, "images", stem + ".jpg")).convert("RGB"))
    md_path = os.path.join(WORKSPACE, defect, "metadata", stem + ".json")
    md = json.load(open(md_path)) if os.path.exists(md_path) else {}
    w = img.shape[1]
    f35 = md.get("intrinsics", {}).get("focal_length_35mm_equiv")
    focal = f35 / 36.0 * w if f35 else None
    a = md.get("raw", {}).get("accelerometer_g")
    accel = [a["x"], a["y"], a["z"]] if a else None
    return stem, img, md, focal, accel


def scale_confidence(card, rect, on_surface):
    notes = []
    if card is None:
        return "none", ["no plausible scale card in this photo"]
    if on_surface is False:
        notes.append("card held in the air, not on the defect's surface - size is approximate")
    dis = rect.get("scale_disagreement_pct")
    if dis is not None and dis > 25:
        notes.append(f"perspective correction uncertain at the defect ({dis:.0f}% homography vs affine)")
    if card["length_px"] < 350:
        notes.append(f"card only {card['length_px']:.0f} px long - coarse scale")
    if on_surface is False or (dis is not None and dis > 25):
        level = "low"
    elif notes or (dis is not None and dis > 10):
        level = "medium"
    else:
        level = "high"
    return level, notes


def severity(kind, nums):
    flags = []
    for metric, lv in SEVERITY_THRESHOLDS.get(kind, {}).items():
        v = nums.get(metric)
        if v is None:
            continue
        flags.append("severe" if v >= lv["severe"] else "moderate" if v >= lv["moderate"] else "low")
    if not flags:
        return None
    return max(flags, key=["low", "moderate", "severe"].index)


def draw_overlay(img, box, mask, card, cands, skel=None, loose=None, segments=None, text=(), points=((), ())):
    o = img.copy()
    if loose is not None:
        o[loose & ~mask] = (0.6 * o[loose & ~mask] + [255 * 0.4, 160 * 0.4, 0]).astype(np.uint8)
    o[mask] = (0.5 * o[mask] + [127, 0, 0]).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cnts, -1, (255, 40, 40), 2)
    if skel is not None:
        o[cv2.dilate(skel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = (255, 255, 0)
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(o, (x0, y0), (x1, y1), (0, 255, 0), 3)
    for c in cands:
        if c is not card and c["length_px"] > 100:
            cv2.drawContours(o, [np.intp(cv2.boxPoints(cv2.minAreaRect(c["contour"])))], -1, (128, 128, 128), 1)
    if card is not None:
        q = np.array(card["quad"] or card["rect_quad"], np.int32)
        cv2.polylines(o, [q], True, (0, 120, 255), 3)
    for s in segments or []:
        x, y, X, Y = map(int, s["xyxy"])
        col = (255, 0, 255) if s["deg_from_vertical"] < 45 else (0, 255, 255)
        cv2.line(o, (x, y), (X, Y), col, 3)
    for pts, col in zip(points, ((0, 255, 0), (255, 0, 0))):
        for x, y in pts:
            cv2.circle(o, (int(x), int(y)), 9, col, -1)
            cv2.circle(o, (int(x), int(y)), 9, (0, 0, 0), 2)
    for i, t in enumerate(text):
        cv2.putText(o, t, (12, 36 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(o, t, (12, 36 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return o


def rect_view(mask, H, extra=None):
    """Small straight-on rendering of the warped mask for the overlay."""
    try:
        m, _ = metrics.warp_mask(mask, H)
    except ValueError:
        return None
    v = np.full(m.shape + (3,), 255, np.uint8)
    v[m] = (200, 40, 40)
    return v


def measure_shot(defect, shot, prompt, cfg, outdir):
    stem, img, md, focal, accel = load_shot(defect, shot)
    box = prompt["box"]
    card, cands = cardmod.detect_card(img, focal)
    card_m = cardmod.card_mask(img.shape, card) if card is not None else None

    neg = list(prompt["neg_points"])
    if card is not None:
        x0, y0, x1, y1 = box
        neg += [p for p in cardmod.card_neg_points(card)
                if x0 - 50 <= p[0] <= x1 + 50 and y0 - 50 <= p[1] <= y1 + 50]

    rec = {"defect": defect, "shot": shot, "photo": stem + ".jpg", "box_xyxy": box,
           "box_source": "hand-drawn (prompts/%s.json)" % defect, "edge_type": cfg["edge"],
           "image_size": [img.shape[1], img.shape[0]], "focal_px": focal}

    skel, loose, segs, sam_score = None, None, None, None
    if cfg["edge"] == "thin":
        e = edges.thin(img, box, card_m, pos_points=prompt["pos_points"])
        if e is None:
            rec["error"] = "ridge filter found no line in the box"
            ov = draw_overlay(img, box, np.zeros(img.shape[:2], bool), card, cands,
                              points=(prompt["pos_points"], neg), text=[f"{defect} {shot}: no line found"])
            Image.fromarray(ov).save(os.path.join(outdir, f"{shot}_overlay.jpg"), quality=88)
            return rec, None
        mask, skel = e["mask"], e["skeleton"]
    else:
        sam_mask, sam_score = segment.segment(img, box, prompt["pos_points"], neg,
                                              cache_path=os.path.join(outdir, "masks", shot))
        rec["sam2"] = {"score": sam_score, "neg_points": neg, "pos_points": prompt["pos_points"]}
        if cfg["edge"] == "diffuse":
            e = edges.diffuse(img, box, card_m)
            mask, loose = e["tight"], e["loose"]
            rec["sam2"]["note"] = "SAM2 mask kept for comparison only; diffuse area uses colour difference"
            rec["sam2_area_px"] = int(sam_mask.sum())
        else:
            mask = edges.crisp(sam_mask, card_m, box)["mask"]

    if mask.sum() == 0:
        rec["error"] = "empty mask"
        return rec, None

    # an outline that runs into the edge of the box or the photo means the
    # defect continues past it: lengths/areas are then lower bounds
    H_img, W_img = mask.shape
    x0, y0, x1, y1 = box
    m8 = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys_, xs_ = np.nonzero(m8)
    at_photo = bool((xs_.min() <= 3) or (ys_.min() <= 3) or (xs_.max() >= W_img - 4) or (ys_.max() >= H_img - 4))
    at_box = bool((xs_.min() <= x0 + 3) or (ys_.min() <= y0 + 3) or (xs_.max() >= x1 - 4) or (ys_.max() >= y1 - 4))
    rec["clipped"] = {"at_photo_edge": at_photo, "at_box_edge": at_box,
                      "meaning": ("defect continues outside the photo - sizes are lower bounds" if at_photo else
                                  "outline touches the box - widen the box if the defect continues" if at_box else None)}

    # ---- scale + rectification -------------------------------------------
    ys, xs = np.nonzero(mask)
    centroid = [float(xs.mean()), float(ys.mean())]
    rect, nums, nums_alt = None, {}, {}
    want = [m for m in cfg["measures"] if m != "angle"]
    level, notes = scale_confidence(card, {}, cfg["card_on_surface"])
    if card is not None and want:
        rect = cardmod.build_rectifier(card, centroid)
        level, notes = scale_confidence(card, rect, cfg["card_on_surface"])
        try:
            mr, _ = metrics.warp_mask(mask, rect["H"])
            nums = metrics.measure_mask(mr, rect["px_per_cm"], want=want)
            if loose is not None:
                lr, _ = metrics.warp_mask(loose, rect["H"])
                nums["area_cm2_loose"] = float(lr.sum() / rect["px_per_cm"] ** 2)
        except ValueError as ex:
            notes.append(str(ex))
            level, nums = "none", {}

        # same numbers through the other mapping (homography <-> affine): their
        # difference is reported as part of each number's uncertainty
        alt_H = rect["H_affine"] if "homography" in rect["method"] else rect.get("H_homography")
        if nums and alt_H is not None:
            try:
                ar, _ = metrics.warp_mask(mask, alt_H)
                nums_alt = metrics.measure_mask(ar, rect["px_per_cm"], want=want)
            except ValueError:
                nums_alt = {}

    px = metrics.measure_pixels(mask, want=want) if want else {}

    rec["scale"] = {
        "source": "card" if nums else "none",
        "confidence": level,
        "notes": notes,
        "card_on_surface": cfg["card_on_surface"],
        "card_dims_cm": [CARD_LENGTH_CM, CARD_WIDTH_CM],
        "card_dims_source": CARD_DIMS_SOURCE,
        "flatness": "assumed flat by surface type (no LiDAR in capture)",
    }
    if card is not None:
        rec["scale"]["card"] = {k: card[k] for k in ("length_px", "width_px", "aspect", "fill", "solidity",
                                                     "implied_distance_cm", "end_width_ratio", "quad", "threshold")}
        rec["scale"]["card"]["tick_check"] = cardmod.tick_check(img, card)
    if rect:
        rec["scale"].update(px_per_cm=rect["px_per_cm"], rectification=rect["method"],
                            homography_vs_affine_pct=rect["scale_disagreement_pct"])
    rec["rejected_card_candidates"] = [
        {"center": [round(v) for v in c["center"]], "length_px": round(c["length_px"]), "why": c["reject_reasons"]}
        for c in cands if not c["plausible"] and c["length_px"] > 150][:6]

    # ---- numbers, each with its source ------------------------------------
    def tag(v, unit, extra=None):
        d = {"value": round(v, 2) if isinstance(v, float) else v, "unit": unit,
             "scale_source": rec["scale"]["source"], "confidence": level}
        if extra:
            d.update(extra)
        return d

    def spread(key, sub=None):
        a, b = nums.get(key), nums_alt.get(key)
        if sub:
            a, b = (a or {}).get(sub), (b or {}).get(sub)
        if a is None or b is None:
            return None
        return round(abs(b - a) / max(a, 1e-9) * 100, 1)

    unc = lambda k, s=None: {"uncertainty_pct": round(float(np.hypot(CARD_DIMS_UNCERT_PCT, spread(k, s) or 0) * (2 if k.startswith("area") else 1)), 1)}
    M = {}
    if nums:
        if "area_cm2" in nums:
            extra = unc("area_cm2")
            if "area_cm2_loose" in nums:
                extra["range_tight_loose"] = [round(nums["area_cm2"], 1), round(nums["area_cm2_loose"], 1)]
            M["area"] = tag(nums["area_cm2"], "cm2", extra)
        if "length_cm" in nums:
            M["length"] = tag(nums["length_cm"], "cm", {**unc("length_cm"), "pieces_cm": nums["length_pieces_cm"],
                                                         "method": "along skeleton (longest path per piece)"})
            M["width"] = tag(nums["width_cm"]["median"], "cm", {"max_p95_cm": round(nums["width_cm"]["max_p95"], 2),
                                                                 "method": "2 x distance-to-edge on skeleton"})
            M["extent"] = tag(nums["extent_cm"][0], "cm", {"across_cm": round(nums["extent_cm"][1], 2),
                                                           "method": "min-area rectangle (overall L x W)"})
    elif px:
        M["not_measured"] = {"reason": "no usable scale in this photo - pixel sizes only",
                             "pixels": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in px.items()
                                        if k in ("area_cm2", "length_cm", "extent_cm")}}

    if "angle" in cfg["measures"]:
        if accel is None or focal is None:
            M["angle"] = {"not_measured": "no accelerometer / focal in metadata"}
        else:
            ang = gravity.member_angles(img, mask, accel, focal)
            if ang:
                segs = ang.pop("segments")
                M["angle"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in ang.items()}
                M["angle"].update(source="phone accelerometer, rotated into this photo's camera frame",
                                  segments_used=len(segs),
                                  caveat="angle measured in the image; equals the true 3D lean only for "
                                         "members roughly facing the camera")
    for k in ("length", "area", "extent"):
        if k in M and at_photo:
            M[k]["lower_bound"] = True
    rec["measurements"] = M
    rec["depth"] = {"band": cfg["probe"], "source": "worker probe" if cfg["probe"] else None,
                    "note": None if cfg["probe"] else "not probed - depth is never read from the image"}

    sev_in = {}
    if "area" in M: sev_in["area_cm2"] = M["area"]["value"]
    if "length" in M: sev_in["length_cm"] = M["length"]["value"]
    if "width" in M: sev_in["width_mm"] = M["width"]["max_p95_cm"] * 10
    if "angle" in M and "posts_lean_from_vertical_deg" in M["angle"]:
        sev_in["lean_deg"] = M["angle"]["posts_lean_from_vertical_deg"]
    rec["severity_band"] = severity(cfg["severity"], sev_in)
    rec["severity_note"] = "thresholds carried over from earlier code - need recalibration"

    # ---- overlay -------------------------------------------------------------
    lines = [f"{defect} {shot}  scale: {rec['scale']['source']} ({level})"]
    if "area" in M:
        r = M["area"].get("range_tight_loose")
        lines.append(f"area {M['area']['value']} cm2" + (f"  (range {r[0]}-{r[1]})" if r else ""))
    if "length" in M:
        lines.append(f"length {M['length']['value']} cm  width {M['width']['value']}-{M['width']['max_p95_cm']} cm")
        lines.append(f"extent {M['extent']['value']} x {M['extent']['across_cm']} cm")
    if "angle" in M and "posts_lean_from_vertical_deg" in M["angle"]:
        lines.append(f"posts lean {M['angle']['posts_lean_from_vertical_deg']} deg from vertical")
    if at_photo:
        lines.append("runs off the photo -> LOWER BOUND")
    if "not_measured" in M:
        lines.append("NOT MEASURED: no usable scale card")
    ov = draw_overlay(img, box, mask, card, cands, skel=skel, loose=loose, segments=segs, text=lines,
                      points=(prompt["pos_points"], neg))
    if rect:
        rv = rect_view(mask, rect["H"])
        if rv is not None:
            s = min(300 / rv.shape[0], 400 / rv.shape[1])
            rv = cv2.resize(rv, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            h, w = rv.shape[:2]
            ov[-h - 10:-10, -w - 10:-10] = rv
            cv2.rectangle(ov, (ov.shape[1] - w - 10, ov.shape[0] - h - 10), (ov.shape[1] - 10, ov.shape[0] - 10), (0, 0, 0), 2)
    Image.fromarray(ov).save(os.path.join(outdir, f"{shot}_overlay.jpg"), quality=88)
    return rec, ov


def pilot_check(cfg, rec):
    p = cfg.get("pilot")
    if not p or rec is None:
        return None
    M = rec.get("measurements", {})
    val = {"length_cm": (M.get("length") or {}).get("value"),
           "width_cm": (M.get("extent") or {}).get("across_cm"),
           "lean_deg": (M.get("angle") or {}).get("posts_lean_from_vertical_deg")}.get(p["metric"])
    if val is None:
        return {"pilot": p["text"], "ours": None, "pass": None}
    ok = p["lo"] * 0.85 <= val <= p["hi"] * 1.15
    return {"pilot": p["text"], "metric": p["metric"], "ours": val, "pass_within_15pct": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("defects", nargs="*", default=list(DEFECTS))
    ap.add_argument("--primary-only", action="store_true")
    args = ap.parse_args()

    summary = {}
    for d in args.defects:
        cfg = DEFECTS[d]
        prompts = segment.load_prompts(os.path.join(PROMPTS_DIR, f"{d}.json"))
        shots = [cfg["primary"]] if args.primary_only else sorted(prompts)
        outdir = os.path.join(OUTPUT_DIR, d)
        os.makedirs(outdir, exist_ok=True)
        recs = {}
        for s in shots:
            if s not in prompts:
                print(f"[{d} {s}] no box in prompts - skipped", flush=True)
                continue
            rec, _ = measure_shot(d, s, prompts[s], cfg, outdir)
            recs[s] = rec
            with open(os.path.join(outdir, f"{s}.json"), "w") as f:
                json.dump(rec, f, indent=1, default=float)
            M = rec.get("measurements", {})
            brief = {k: (v.get("value") if isinstance(v, dict) and "value" in v else
                         {kk: vv for kk, vv in v.items() if kk.endswith("deg")} if k == "angle" else "-")
                     for k, v in M.items()}
            print(f"[{d} {s}] scale={rec.get('scale', {}).get('source')}/{rec.get('scale', {}).get('confidence')} "
                  f"{brief} {rec.get('error', '')}", flush=True)
        prim = recs.get(cfg["primary"])
        summary[d] = {"primary": cfg["primary"], "primary_result": prim, "pilot_check": pilot_check(cfg, prim),
                      "all_shots": {s: r.get("measurements") for s, r in recs.items()}}
    # merge with sets measured in earlier runs, so a partial run keeps the rest
    sj = os.path.join(OUTPUT_DIR, "summary.json")
    if os.path.exists(sj):
        old = json.load(open(sj))
        old.update(summary)
        summary = {d: old[d] for d in DEFECTS if d in old}
    with open(sj, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    write_summary_md(summary)


def _key_numbers(M):
    if not M:
        return "-"
    parts = []
    if "area" in M:
        r = M["area"].get("range_tight_loose")
        parts.append(f"area {M['area']['value']:.1f} cm2" + (f" (tight-loose {r[0]}-{r[1]})" if r else "")
                     + f" +-{M['area'].get('uncertainty_pct', '?')}%")
    if "length" in M:
        lb = " (lower bound)" if M["length"].get("lower_bound") else ""
        parts.append(f"length {M['length']['value']:.1f} cm{lb} +-{M['length'].get('uncertainty_pct', '?')}%")
        parts.append(f"width {M['width']['value'] * 10:.1f}-{M['width']['max_p95_cm'] * 10:.1f} mm")
        parts.append(f"extent {M['extent']['value']:.1f} x {M['extent']['across_cm']:.1f} cm")
    if "angle" in M and isinstance(M["angle"], dict):
        a = M["angle"]
        if "posts_lean_from_vertical_deg" in a:
            parts.append(f"posts {a['posts_lean_from_vertical_deg']:.1f} deg from vertical")
        if "members_tilt_from_horizontal_deg" in a:
            parts.append(f"rails {a['members_tilt_from_horizontal_deg']:.1f} deg from level (in image)")
    if "not_measured" in M:
        parts.append("NOT MEASURED - " + M["not_measured"]["reason"])
    return "; ".join(parts)


def write_summary_md(summary):
    L = ["# Defect measurements - summary", "",
         "Generated by `run.py`. One primary photo per defect is THE answer; the other prompted photos",
         "of the same defect are measured the same way and shown only as a consistency check.", "",
         "| Defect | Photo | Scale source (confidence) | Numbers | Pilot hand estimate | Within 15%? |",
         "|---|---|---|---|---|---|"]
    for d, s in summary.items():
        r = s.get("primary_result") or {}
        sc = r.get("scale", {})
        pc = s.get("pilot_check") or {}
        ok = pc.get("pass_within_15pct")
        L.append(f"| {d} | {s['primary']} | {sc.get('source', '-')} ({sc.get('confidence', '-')}) | "
                 f"{_key_numbers(r.get('measurements'))} | {pc.get('pilot', '-')} | "
                 f"{'yes' if ok else 'no' if ok is False else '-'} |")
    L += ["", "## Every photo (consistency check)", ""]
    for d, s in summary.items():
        L += [f"### {d}", "", "| Photo | Numbers |", "|---|---|"]
        for shot, M in sorted((s.get("all_shots") or {}).items()):
            star = " (primary)" if shot == s["primary"] else ""
            L.append(f"| {shot}{star} | {_key_numbers(M)} |")
        L.append("")
    with open(os.path.join(OUTPUT_DIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
