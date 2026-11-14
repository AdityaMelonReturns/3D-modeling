"""Task B, phase 2: baseline depth-band classifier.

Labels come from labels/depth_labels.csv (one row per labelled defect photo):
    property_id,defect,shot,band,depth_cm,source
    1123-main-st,rotten_wood,09,1-3 cm,2.4,ruler_photo
band is one of config.PROBE_BANDS; source is "probe" or "ruler_photo".
Features come from output/<defect>/<shot>.json -> depth.features (run.py
writes them for rot / hole photos).

    python train_depth_band.py            # count labels; train + test once >= 200
    python train_depth_band.py --smoke    # check the code runs, on RANDOM fake data

Test, as the brief asks: split by PROPERTY (GroupKFold) so photos of one
defect are never in both train and test; confusion matrix; accuracy per
band; how often a deep defect (>3 cm / through) is called shallow (<1 / 1-3 cm),
the costly mistake; compared with always guessing the most common band.
The model's output, once used, is depth.band with depth.source="classifier",
confidence "low", and never a cm number.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, PROBE_BANDS
from measure.depth_band import MIN_LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "labels", "depth_labels.csv")
DEEP, SHALLOW = set(PROBE_BANDS[2:]), set(PROBE_BANDS[:2])


def load():
    rows = list(csv.DictReader(open(LABELS))) if os.path.exists(LABELS) else []
    X, y, g, names, missing = [], [], [], None, []
    for r in rows:
        f = os.path.join(OUTPUT_DIR, r["defect"], f"{r['shot']}.json")
        feats = json.load(open(f)).get("depth", {}).get("features") if os.path.exists(f) else None
        if not feats:
            missing.append(f"{r['defect']}_{r['shot']}")
            continue
        names = names or sorted(k for k, v in feats.items() if isinstance(v, (int, float)))
        X.append([float(feats.get(k) or 0.0) for k in names])
        y.append(r["band"])
        g.append(r["property_id"])
    return np.array(X), np.array(y), np.array(g), names, rows, missing


def evaluate(X, y, groups, names):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "majority (always the most common band)": None,
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gradient boosting": lambda: GradientBoostingClassifier(random_state=0),
    }
    k = min(5, len(set(groups)))
    out = {}
    for name, make in models.items():
        pred = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            if make is None:
                pred[te] = Counter(y[tr]).most_common(1)[0][0]
            else:
                m = make().fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
        cm = confusion_matrix(y, pred, labels=PROBE_BANDS)
        per_band = {b: (float(np.mean(pred[y == b] == b)) if (y == b).any() else None) for b in PROBE_BANDS}
        deep = np.isin(y, list(DEEP))
        out[name] = {"accuracy": float(np.mean(pred == y)), "per_band_accuracy": per_band,
                     "deep_called_shallow": float(np.mean(np.isin(pred[deep], list(SHALLOW)))) if deep.any() else None,
                     "confusion_matrix(rows=true, cols=pred)": cm.tolist()}
    return out, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on random fake data to check the code works")
    args = ap.parse_args()

    if args.smoke:
        rng = np.random.default_rng(0)
        n = 240
        X = rng.normal(size=(n, 8))
        y = np.array(PROBE_BANDS)[rng.integers(0, 4, n)]
        g = np.array([f"prop{i % 40}" for i in range(n)])
        res, k = evaluate(X, y, g, [f"f{i}" for i in range(8)])
        print("SMOKE TEST on random data (numbers mean nothing; ~25% is chance):")
        for m, r in res.items():
            print(f"  {m}: accuracy {r['accuracy']:.2f}, deep called shallow {r['deep_called_shallow']:.2f}")
        return

    X, y, g, names, rows, missing = load()
    counts = Counter(y.tolist())
    L = ["# Depth-band classifier - label status", "",
         f"Labelled defect photos with features: **{len(y)}** of the **{MIN_LABELS}** needed before training "
         f"({len(set(g.tolist()))} properties).", "",
         "| Band | Labels |", "|---|---|"] + [f"| {b} | {counts.get(b, 0)} |" for b in PROBE_BANDS]
    if missing:
        L += ["", f"Labels whose photo has no features yet (run run.py on them): {', '.join(missing)}"]
    if len(y) < MIN_LABELS:
        L += ["", f"**Not trained.** Collect labels with the capture protocol (docs/CAPTURE_PROTOCOL.md); "
                  f"'>3 cm' and 'through' will be rare - track them."]
        print("\n".join(L))
        open(os.path.join(OUTPUT_DIR, "DEPTH_BAND_STATUS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        return
    res, k = evaluate(X, y, g, names)
    L += ["", f"Held-out test: {k}-fold split by property.", ""]
    for m, r in res.items():
        L += [f"### {m}", f"- accuracy {r['accuracy']:.2f}; deep called shallow {r['deep_called_shallow']}",
              f"- per band: {r['per_band_accuracy']}", f"- confusion (rows true, cols predicted, order {PROBE_BANDS}): "
              f"{r['confusion_matrix(rows=true, cols=pred)']}", ""]
    print("\n".join(L))
    open(os.path.join(OUTPUT_DIR, "DEPTH_BAND_STATUS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
