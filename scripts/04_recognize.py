"""Step 4: name every product on the trays and build each tray's bill.

Every choice is made on the validation trays only, so the test numbers in step 5 stay honest:
  1. which vectors name products best (model or pair of models, 12 catalog photos per product or
     the whole catalog), by item accuracy on the annotated boxes;
  2. which OWLv2 boxes count (score, similarity and the group/piece rules), by whole-bill accuracy;
  3. the attendant's margin: the TOUCH_RATE least sure items of validation get checked;
  4. when to ask for the products to be spread out (piled-up products).

Writes data/output/config.json, data/output/predictions.json and data/output/validation.json.

  python scripts/04_recognize.py
"""
import importlib
import itertools
import json

import numpy as np

from common import OUT, write_json
from tray import (Matcher, bill, ground_truth, match, overlap, read_tray, selection, top2,
                  touched_bill)

EMBEDDERS = importlib.import_module("03_embed").MODELS

# --- parameters -------------------------------------------------------------------------------
GRID_DET_SCORE = [0.10, 0.15, 0.20, 0.25, 0.30]
GRID_SIM_QUANTILE = [None, 0.05, 0.10, 0.25]  # weak boxes kept above this quantile of the
                                              # similarity of real products (None: off)
GRID_CONTAIN = [0.6, 0.7, 0.8, 0.9, 1.01]     # 1.01 = never drop a box
GRID_DUP_IOU = [0.3, 0.5, 1.01]               # 1.01 = never merge doubles
TOUCH_RATE = 0.10        # the attendant checks this share of items (the least sure ones)
GRID_SPREAD = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # ask to spread the products when a box lies
                                                   # this much inside another
# -----------------------------------------------------------------------------------------------


def candidates():
    """Every model step 3 produced, alone and in DINOv2 + CLIP pairs, with 12 catalog photos per
    product and, where step 3b ran, with the whole catalog."""
    have = [m for m in EMBEDDERS if (OUT / f"emb_{m}.npz").exists()]
    dino = [m for m in have if m.startswith("dinov2")]
    clip = [m for m in have if m.startswith("clip")]
    for models in have + [f"{d}+{c}" for d, c in itertools.product(dino, clip)]:
        yield {"models": models, "full": False}
        if all((OUT / f"emb_full_{m}.npz").exists() for m in models.split("+")):
            yield {"models": models, "full": True}


def item_accuracy(matcher, split="validation"):
    s, meta = matcher.meta("gt")
    prod, _, _ = top2(matcher.scores("gt", s == split))
    return float((prod == meta[s == split, 1]).mean())


def tray_scores(matcher, split):
    """{tray offset: (detections in step-2 order, scores (n, 200))}"""
    det = json.loads((OUT / "detections.json").read_text())[split]
    s, offs = matcher.meta("det")
    sc = matcher.scores("det", s == split)
    offs = offs[s == split]
    return {int(o): (d, sc[offs == int(o)]) for o, d in det.items()}


def read_all(scored, det_score, sim_min, contain, dup_iou):
    return {o: read_tray(d, s, det_score, sim_min, contain, dup_iou)
            for o, (d, s) in scored.items()}


def bill_accuracy(reads, gt):
    return float(np.mean([bill(items) == bill([{"product": c} for c in gt[o]["category"]])
                          for o, items in reads.items()]))


def main():
    sel = selection()
    gt = ground_truth("validation")

    print("1) vectors: item accuracy on the annotated validation boxes")
    table = []
    for cfg in candidates():
        acc = item_accuracy(Matcher(cfg["models"], full=cfg["full"]))
        table.append({**cfg, "item_accuracy": acc})
        print(f"  {cfg['models']:32} {'whole catalog' if cfg['full'] else '12 photos':14} "
              f"{acc:6.1%}", flush=True)
    best = max(table, key=lambda r: r["item_accuracy"])
    vec = {"models": best["models"], "full": best["full"]}
    print("  best:", vec)

    matcher = Matcher(vec["models"], full=vec["full"])
    scored = tray_scores(matcher, "validation")

    print("2) which boxes count: whole-bill accuracy on validation")
    s, meta = matcher.meta("gt")
    _, real_sim, _ = top2(matcher.scores("gt", s == "validation"))
    grid = []
    for ds, q, ct, du in itertools.product(GRID_DET_SCORE, GRID_SIM_QUANTILE, GRID_CONTAIN,
                                           GRID_DUP_IOU):
        sim_min = 1e9 if q is None else float(np.quantile(real_sim, q))
        acc = bill_accuracy(read_all(scored, ds, sim_min, ct, du), gt)
        grid.append({"det_score": ds, "sim_quantile": q, "sim_min": sim_min, "contain": ct,
                     "dup_iou": du, "bill_accuracy": acc})
    # ties: prefer the simplest rule (no similarity rescue, no double merge, highest score)
    top = max(grid, key=lambda r: (r["bill_accuracy"], r["sim_quantile"] is None,
                                   r["dup_iou"] > 1, r["det_score"]))
    for row in sorted(grid, key=lambda r: -r["bill_accuracy"])[:6]:
        print(f"  det_score={row['det_score']:.2f} sim_quantile={row['sim_quantile']} "
              f"contain={row['contain']:.2f} dup_iou={row['dup_iou']:.2f}  "
              f"{row['bill_accuracy']:6.1%}")
    det_score, sim_min, contain, dup_iou = (top[k] for k in ("det_score", "sim_min", "contain",
                                                              "dup_iou"))

    reads = read_all(scored, det_score, sim_min, contain, dup_iou)
    margins = np.array([it["margin"] for items in reads.values() for it in items])
    margin_min = float(np.quantile(margins, TOUCH_RATE))
    print(f"3) attendant checks items with margin < {margin_min:.4f} "
          f"({TOUCH_RATE:.0%} of validation items)")

    # 4. "please spread the products": the overlap threshold that best separates the bills that
    # will be wrong (even after the attendant) from the right ones (Youden's J = TPR - FPR)
    wrong, over = [], []
    for o, items in reads.items():
        g = gt[o]
        fixed, _ = touched_bill(items, match(items, g["bbox"]), g["category"], margin_min)
        wrong.append(fixed != bill([{"product": c} for c in g["category"]]))
        over.append(overlap([it["box"] for it in items]))
    wrong, over = np.array(wrong), np.array(over)
    spread = []
    for t in GRID_SPREAD:
        flag = over >= t
        tpr = flag[wrong].mean() if wrong.any() else 0.0
        fpr = flag[~wrong].mean() if (~wrong).any() else 0.0
        spread.append({"spread_overlap": t, "asked": float(flag.mean()), "j": float(tpr - fpr)})
    spread_overlap = max(spread, key=lambda r: r["j"])["spread_overlap"]
    print(f"4) ask to spread the products when a box lies >= {spread_overlap:.0%} inside another "
          f"({[(r['spread_overlap'], round(r['asked'], 2), round(r['j'], 2)) for r in spread]})")

    # the scale of every other model too, so tillscope.py can pair any two of them
    norm = dict(matcher.norm)
    for m in EMBEDDERS:
        if m not in norm and (OUT / f"emb_{m}.npz").exists():
            norm[m] = Matcher(f"{m}+{m}").norm[m]
    config = {**vec, "norm": norm, "det_score": det_score, "sim_min": sim_min,
              "contain": contain,
              "dup_iou": dup_iou, "margin_min": margin_min, "touch_rate": TOUCH_RATE,
              "spread_overlap": spread_overlap}
    write_json(OUT / "config.json", config)
    write_json(OUT / "validation.json", {"vectors": table, "detection": grid, "spread": spread})

    preds = {}
    for split in sel["trays"]:
        sc = scored if split == "validation" else tray_scores(matcher, split)
        preds[split] = {str(o): items
                        for o, items in read_all(sc, det_score, sim_min, contain,
                                                 dup_iou).items()}
    write_json(OUT / "predictions.json", preds)
    print("config:", config)


if __name__ == "__main__":
    main()
