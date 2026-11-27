"""Use the study on your own photos: a catalog folder and a photo of a checkout tray.

  # one folder per product, a few photos each, from several angles
  python scripts/tillscope.py build-catalog my_catalog/ --out my_catalog.npz

  # read a tray; with --expect, compare with what was rung up (one product name per line)
  python scripts/tillscope.py check tray.jpg --catalog my_catalog.npz --expect receipt.txt \
      --out tray_checked.jpg

The thresholds come from data/output/config.json when the study was run, else the defaults below.
"""
import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from common import OUT
from tray import overlap, read_tray

detect_mod = importlib.import_module("02_detect")
embed_mod = importlib.import_module("03_embed")

# --- parameters (used when data/output/config.json does not exist) ----------------------------
DEFAULTS = {"models": "dinov2-448+clip-large-336",          # the values the study chose
            "norm": {"dinov2-448": [0.397, 0.175], "clip-large-336": [0.782, 0.051]},
            "det_score": 0.3, "sim_min": 1.85, "contain": 0.6, "dup_iou": 1.01,
            "margin_min": 0.055, "spread_overlap": 0.5}
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
# -----------------------------------------------------------------------------------------------


def config():
    path = OUT / "config.json"
    cfg = dict(DEFAULTS)
    if path.exists():
        cfg.update({k: v for k, v in json.loads(path.read_text()).items() if k in DEFAULTS})
    return cfg


def load_detector():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(detect_mod.MODEL)
    owl = Owlv2ForObjectDetection.from_pretrained(detect_mod.MODEL).to(device).eval()
    return device, proc, owl


def vectors(emb, img, boxes, rotations=(0,)):
    crops = [embed_mod.square_crop(img, b, emb.size) for b in boxes]
    crops = [c.rotate(r) if r else c for c in crops for r in rotations]
    out = [emb(crops[i:i + embed_mod.BATCH])["cls"]
           for i in range(0, len(crops), embed_mod.BATCH)]
    return np.concatenate(out).reshape(len(boxes), len(rotations), -1)


def build_catalog(args):
    cfg = config()
    models = (args.models or cfg["models"]).split("+")
    device, proc, owl = load_detector()
    jobs = []  # (product, photo, product box)
    folders = sorted(p for p in Path(args.folder).iterdir() if p.is_dir())
    for folder in folders:
        for photo in sorted(folder.iterdir()):
            if photo.suffix.lower() not in PHOTO_TYPES:
                continue
            img = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
            dets = [d for d in detect_mod.detect(owl, proc, img, device)
                    if d[4] >= cfg["det_score"]]
            # the product is the biggest confident box; without one, the whole photo
            box = max(dets, key=lambda d: d[2] * d[3])[:4] if dets else [0, 0, *img.size]
            jobs.append((folder.name, img, box))
        print(f"{folder.name}: {sum(j[0] == folder.name for j in jobs)} photos")
    arrays = {"product": np.array([j[0] for j in jobs]), "models": "+".join(models)}
    for m in models:
        emb = embed_mod.Embedder(m, device)
        arrays[f"x_{m}"] = np.stack([vectors(emb, img, [box], embed_mod.ROTATIONS)[0]
                                     for _, img, box in jobs])
        del emb
    np.savez_compressed(args.out, **arrays)
    print(f"{len(jobs)} photos of {len(folders)} products, {'+'.join(models)} -> {args.out}")


def check(args):
    cfg = config()
    cat = np.load(args.catalog)
    models = str(cat["models"]).split("+")
    names = sorted(set(cat["product"].tolist()))
    device, proc, owl = load_detector()
    img = ImageOps.exif_transpose(Image.open(args.photo)).convert("RGB")
    dets = detect_mod.detect(owl, proc, img, device)
    scores = np.full((len(dets), max(len(names), 2)), -1e9, np.float32)
    if dets:
        per_model = []
        for m in models:
            emb = embed_mod.Embedder(m, device)
            q = vectors(emb, img, [d[:4] for d in dets])[:, 0]
            del emb
            x = cat[f"x_{m}"]
            s = q @ x.reshape(-1, x.shape[-1]).T
            owner = np.repeat(cat["product"], x.shape[1])
            s = np.stack([s[:, owner == n].max(1) for n in names], 1)
            if len(models) > 1:  # put both models on one scale (constants from the study)
                mu, sd = cfg["norm"].get(m, [0.0, 1.0])
                s = (s - mu) / sd
            per_model.append(s)
        scores[:, :len(names)] = np.mean(per_model, axis=0)
    items = read_tray(dets, scores, cfg["det_score"], cfg["sim_min"], cfg["contain"],
                      cfg["dup_iou"])
    for it in items:
        it["name"] = names[it["product"]]
        it["check"] = it["margin"] < cfg["margin_min"]

    print(f"{len(items)} products on the tray")
    for name, n in sorted(Counter(it["name"] for it in items).items()):
        doubt = sum(it["check"] for it in items if it["name"] == name)
        print(f"  {n} x {name}" + (f"   ({doubt} to confirm)" if doubt else ""))
    spread = overlap([it["box"] for it in items]) >= cfg["spread_overlap"]
    if spread:
        print("! products are piled up or touching: please spread them out and take a new photo")
    if args.expect:
        rung = Counter(l.strip() for l in open(args.expect) if l.strip())
        seen = Counter(it["name"] for it in items)
        for name in sorted(set(rung) | set(seen)):
            if seen[name] > rung[name]:
                print(f"! on the tray but not rung up: {seen[name] - rung[name]} x {name}")
            elif rung[name] > seen[name]:
                print(f"! rung up but not seen on the tray: {rung[name] - seen[name]} x {name}")
        if seen == rung:
            print("the tray matches what was rung up")
    if args.out:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=max(14, img.size[0] // 60))
        for it in items:
            x0, y0, w, h = it["box"]
            color = (255, 170, 50) if it["check"] else (80, 225, 255)
            draw.rectangle([x0, y0, x0 + w, y0 + h], outline=color, width=4)
            draw.text((x0 + 4, y0 + 2), it["name"], fill=color, font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        img.save(args.out, quality=92)
        print("annotated photo:", args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-catalog", help="vectors of a catalog folder (one folder per product)")
    b.add_argument("folder")
    b.add_argument("--out", default="catalog.npz")
    b.add_argument("--models", help="embedders joined by +, e.g. dinov2+clip (default: the "
                                    "study's choice, " + DEFAULTS["models"] + ")")
    c = sub.add_parser("check", help="read a tray photo and compare it with a receipt")
    c.add_argument("photo")
    c.add_argument("--catalog", required=True)
    c.add_argument("--expect", help="text file, one rung-up product name per line")
    c.add_argument("--out", help="write the photo with the boxes")
    args = ap.parse_args()
    {"build-catalog": build_catalog, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
