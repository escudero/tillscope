"""Step 5: the numbers, on the test trays, with everything frozen on validation in step 4.

Writes data/output/results_<split>.json and prints the tables of the README.

  python scripts/05_evaluate.py [split]      # default: test
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from common import OUT, names, supercategory, write_json
from tray import (Matcher, bill, ground_truth, match, overlap, selection, top2,
                  touched_bill)

LEVELS = ["easy", "medium", "hard"]
TOUCH_RATES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
SPREAD_GRID = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.01]
CAMERA_SETS = [None, ("top",), ("side",), ("45deg",), ("30deg",)]
VIEWS = [1, 3, 6, 12, 40, 160]   # catalog photos per product (above 12: the whole catalog)
EXTRA_MODELS = ["dinov2", "clip", "dinov2-large", "dinov2-giant", "dinov2-448",
                "dinov2-large-448", "clip-large-336"]


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    cfg = json.loads((OUT / "config.json").read_text())
    preds = json.loads((OUT / "predictions.json").read_text())[split]
    sel = selection()
    gt = ground_truth(split)
    level = {o: lv for lv, offs in sel["trays"][split].items() for o in offs}
    matcher = Matcher(cfg["models"], full=cfg["full"], norm=cfg["norm"])

    # recognition alone, on the annotated boxes
    s, meta = matcher.meta("gt")
    mask = s == split
    gt_meta = meta[mask]
    gt_scores = matcher.scores("gt", mask)
    gt_prod, _, _ = top2(gt_scores)
    right = gt_prod == gt_meta[:, 1]
    oracle = defaultdict(list)
    for (o, c), p in zip(gt_meta, gt_prod):
        oracle[int(o)].append(int(p))

    # margins of the whole split, for the attendant curve
    all_margins = np.array([it["margin"] for items in preds.values() for it in items])

    rows, per_tray = {}, []
    for o, items in preds.items():
        o = int(o)
        g = gt[o]
        truth = Counter(g["category"])
        pairs = match(items, g["bbox"])
        named = sum(1 for k, gi in pairs.items() if items[k]["product"] == g["category"][gi])
        fixed, touches = touched_bill(items, pairs, g["category"], cfg["margin_min"])
        boxes = [it["box"] for it in items]
        per_tray.append({
            "offset": o, "level": level[o], "objects": len(g["category"]),
            "detections": len(items), "found": len(pairs), "named_right": named,
            "bill_right": bill(items) == truth,
            "oracle_bill_right": Counter(oracle[o]) == truth,
            "touched_bill_right": fixed == truth, "touches": touches,
            "overlap": overlap(boxes), "gt_overlap": overlap(g["bbox"]),
            "count_diff": abs(len(items) - len(g["category"])),
        })

    def summary(trays):
        n_obj = sum(t["objects"] for t in trays)
        n_det = sum(t["detections"] for t in trays)
        return {
            "trays": len(trays),
            "objects": n_obj,
            "found": sum(t["found"] for t in trays) / n_obj,
            "extra_boxes_per_tray": (n_det - sum(t["found"] for t in trays)) / len(trays),
            "named_right": sum(t["named_right"] for t in trays) / n_obj,
            "bill_accuracy": np.mean([t["bill_right"] for t in trays]),
            "oracle_bill_accuracy": np.mean([t["oracle_bill_right"] for t in trays]),
            "touched_bill_accuracy": np.mean([t["touched_bill_right"] for t in trays]),
            "touches_per_tray": np.mean([t["touches"] for t in trays]),
            "count_right": np.mean([t["count_diff"] == 0 for t in trays]),
            "mean_count_diff": np.mean([t["count_diff"] for t in trays]),
        }

    for lv in LEVELS:
        rows[lv] = summary([t for t in per_tray if t["level"] == lv])
    rows["all"] = summary(per_tray)
    # recognition alone per level
    lv_of_box = np.array([level[int(o)] for o, _ in gt_meta])
    for lv in LEVELS:
        rows[lv]["item_accuracy_gt_boxes"] = float(right[lv_of_box == lv].mean())
    rows["all"]["item_accuracy_gt_boxes"] = float(right.mean())

    # the attendant: check the least sure items, curve over how many get checked
    curves = defaultdict(list)
    for rate in TOUCH_RATES:
        thr = float(np.quantile(all_margins, rate)) if rate else -1.0
        ok, n_touch = defaultdict(list), defaultdict(list)
        for o, items in preds.items():
            g = gt[int(o)]
            fixed, t = touched_bill(items, match(items, g["bbox"]), g["category"], thr)
            for key in (level[int(o)], "all"):
                ok[key].append(fixed == Counter(g["category"]))
                n_touch[key].append(t)
        for key in ok:
            curves[key].append({"touch_rate": rate, "margin_min": thr,
                                "touches_per_tray": float(np.mean(n_touch[key])),
                                "bill_accuracy": float(np.mean(ok[key]))})
    touch_curve = curves["all"]

    # please spread the products: trays where a box sits under another are sent back
    spread_curve = []
    for thr in SPREAD_GRID:
        acc = [t for t in per_tray if t["overlap"] < thr]
        spread_curve.append({
            "overlap_max": thr, "accepted": len(acc) / len(per_tray),
            "bill_accuracy_accepted":
                float(np.mean([t["bill_right"] for t in acc])) if acc else None,
            "touched_bill_accuracy_accepted":
                float(np.mean([t["touched_bill_right"] for t in acc])) if acc else None,
        })
    acc = [t for t in per_tray if t["overlap"] < cfg["spread_overlap"]]
    spread = {lv: {"accepted": np.mean([t["overlap"] < cfg["spread_overlap"] for t in per_tray
                                        if t["level"] == lv]),
                   "touched_bill_accuracy_accepted": np.mean(
                       [t["touched_bill_right"] for t in acc if t["level"] == lv] or [np.nan])}
              for lv in LEVELS}
    spread["all"] = {"accepted": len(acc) / len(per_tray),
                     "touched_bill_accuracy_accepted":
                         float(np.mean([t["touched_bill_right"] for t in acc]))}

    def accuracy(models=cfg["models"], **kw):
        p, _, _ = top2(Matcher(models, **kw).scores("gt", mask))
        return float((p == gt_meta[:, 1]).mean())

    # which vectors: every model alone and the chosen pair, 12 photos or the whole catalog
    models = []
    for m in sorted(set(cfg["models"].split("+")) | set(EXTRA_MODELS)) + [cfg["models"]]:
        if all((OUT / f"emb_{x}.npz").exists() for x in m.split("+")):
            for full in (False, True):
                if not full or all((OUT / f"emb_full_{x}.npz").exists() for x in m.split("+")):
                    models.append({"models": m, "full": full,
                                   "item_accuracy": accuracy(m, full=full)})
    # which catalog photos matter, with the chosen models and catalog
    cameras = [{"cameras": "all" if c is None else "+".join(c),
                "item_accuracy": accuracy(cameras=c, full=cfg["full"])} for c in CAMERA_SETS]
    views = [{"views": v, "item_accuracy": accuracy(views=v, full=cfg["full"])}
             for v in VIEWS if cfg["full"] or v <= 12]
    rotations = {"with": accuracy(full=cfg["full"]),
                 "without": accuracy(rotations=False, full=cfg["full"])}

    # mistakes on the annotated boxes: same supercategory (look-alikes) or not
    wrong = ~right
    same_super = np.array([supercategory(int(a)) == supercategory(int(b))
                           for a, b in zip(gt_prod[wrong], gt_meta[wrong, 1])])
    by_super = defaultdict(list)
    for c, r in zip(gt_meta[:, 1], right):
        by_super[supercategory(int(c))].append(bool(r))
    confusions = Counter((names()[int(b)], names()[int(a)])
                         for a, b in zip(gt_prod[wrong], gt_meta[wrong, 1]))

    # where whole bills go wrong
    causes = Counter()
    for t in per_tray:
        if t["bill_right"]:
            continue
        if t["found"] < t["objects"]:
            causes["a product was missed"] += 1
        elif t["detections"] > t["found"]:
            causes["an extra box (piece or double)"] += 1
        else:
            causes["every box right, a wrong name"] += 1

    res = {"split": split, "config": cfg, "levels": rows, "touch_curve": touch_curve,
           "touch_curve_by_level": {k: curves[k] for k in LEVELS},
           "spread_curve": spread_curve, "spread": spread, "models": models, "cameras": cameras,
           "views": views, "rotations": rotations,
           "wrong_same_supercategory": float(same_super.mean()) if wrong.any() else None,
           "item_accuracy_by_supercategory": {k: float(np.mean(v)) for k, v in by_super.items()},
           "top_confusions": [[a, b, n] for (a, b), n in confusions.most_common(10)],
           "bill_error_causes": dict(causes), "trays": per_tray}
    write_json(OUT / f"results_{split}.json", json.loads(json.dumps(res, default=float)))

    print(f"## {split}: {len(per_tray)} trays\n")
    print("| | easy | medium | hard | all |\n|---|---|---|---|---|")
    for key, label in [("objects", "products"), ("found", "found by OWLv2"),
                       ("extra_boxes_per_tray", "extra boxes per tray"),
                       ("item_accuracy_gt_boxes", "named right (annotated boxes)"),
                       ("named_right", "named right (full pipeline)"),
                       ("count_right", "item count right"),
                       ("bill_accuracy", "whole bill right (cAcc)"),
                       ("oracle_bill_accuracy", "cAcc with the annotated boxes"),
                       ("touches_per_tray", "items checked by the attendant"),
                       ("touched_bill_accuracy", "cAcc after the attendant")]:
        vals = [rows[k][key] for k in LEVELS + ["all"]]
        fmt = (lambda v: f"{v:.2f}") if key in ("extra_boxes_per_tray", "touches_per_tray") \
            else (str if key == "objects" else pct)
        print(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
    print("\nspread rule:", {k: {kk: round(float(vv), 3) for kk, vv in v.items()}
                             for k, v in spread.items()})
    print("touch curve:", [(r["touch_rate"], round(r["touches_per_tray"], 2),
                            round(r["bill_accuracy"], 3)) for r in touch_curve])
    print("models:", [(r["models"], r["full"], round(r["item_accuracy"], 3)) for r in models])
    print("rotations:", rotations)
    print("cameras:", [(r["cameras"], round(r["item_accuracy"], 3)) for r in cameras])
    print("views:", [(r["views"], round(r["item_accuracy"], 3)) for r in views])
    print("wrong names within the same supercategory:", res["wrong_same_supercategory"])
    print("bill error causes:", dict(causes))
    print("by supercategory:", {k: round(v, 2) for k, v in
                                sorted(res["item_accuracy_by_supercategory"].items(),
                                       key=lambda kv: kv[1])})


if __name__ == "__main__":
    main()
