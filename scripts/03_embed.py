"""Step 3: turn every product crop into a vector ("visual fingerprint").

Three kinds of crops:
  catalog   one per catalog photo (the RPC box around the single product), in 4 rotations,
            because on the tray a product can lie at any angle
  tray_gt   every annotated product on the trays (to measure recognition alone)
  tray_det  every OWLv2 box from step 2 (the real pipeline)

Writes data/output/emb_<name>.npz (float16). Runs on GPU when there is one, else CPU.

  python scripts/03_embed.py [name ...]      # names from MODELS; default: all of them
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from common import CATALOG, INDEX, OUT, SELECTION, TRAYS, read_jsonl

# --- parameters -------------------------------------------------------------------------------
MODELS = {                  # name: (Hugging Face repository, input side in pixels)
    "dinov2": ("facebook/dinov2-base", 224),
    "dinov2-448": ("facebook/dinov2-base", 448),
    "dinov2-large": ("facebook/dinov2-large", 224),
    "dinov2-large-448": ("facebook/dinov2-large", 448),
    "dinov2-giant": ("facebook/dinov2-giant", 224),
    "dinov2-giant-448": ("facebook/dinov2-giant", 448),
    "clip": ("openai/clip-vit-base-patch32", 224),
    "clip-large-336": ("openai/clip-vit-large-patch14-336", 336),
}
PAD = 0.04                  # context added around each box, as a fraction of its size
ROTATIONS = (0, 90, 180, 270)
BATCH = 64
# -----------------------------------------------------------------------------------------------


def square_crop(img, box, size=224):
    """The box plus a little context, padded with white to a square, resized to `size`."""
    x, y, w, h = box
    p = PAD * max(w, h)
    c = img.crop((int(x - p), int(y - p), int(x + w + p), int(y + h + p)))
    side = max(c.size)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(c, ((side - c.size[0]) // 2, (side - c.size[1]) // 2))
    return sq.resize((size, size), Image.BICUBIC)


class Embedder:
    def __init__(self, name, device):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel
        repo, self.size = MODELS[name]
        self.clip, self.device = name.startswith("clip"), device
        self.proc = AutoImageProcessor.from_pretrained(repo)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = CLIPModel if self.clip else AutoModel
        self.model = model.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, crops):
        # the crops are already square at the model's size: no resize or center crop that would
        # cut the edges
        x = self.proc(images=crops, return_tensors="pt", do_resize=False,
                      do_center_crop=False)["pixel_values"].to(self.device, self.dtype)
        if self.clip:
            out = self.model.get_image_features(pixel_values=x)
            feats = {"cls": out if torch.is_tensor(out) else out.pooler_output}
        else:
            h = self.model(pixel_values=x).last_hidden_state
            feats = {"cls": h[:, 0], "mean": h[:, 1:].mean(1)}
        return {k: torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
                for k, v in feats.items()}


def run(emb, items):
    """items: list of (image_path, box, rotation). Returns {feature: array [N, D]}."""
    out, cache = defaultdict(list), {}
    for i in range(0, len(items), BATCH):
        crops = []
        for path, box, rot in items[i:i + BATCH]:
            if path not in cache:
                cache.clear()
                cache[path] = Image.open(path).convert("RGB")
            c = square_crop(cache[path], box, emb.size)
            crops.append(c.rotate(rot) if rot else c)
        for k, v in emb(crops).items():
            out[k].append(v.astype(np.float16))
        if (i // BATCH) % 50 == 0:
            print(f"  {i + len(crops)}/{len(items)}", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    names = sys.argv[1:] or list(MODELS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sel = json.loads(SELECTION.read_text())
    det = json.loads((OUT / "detections.json").read_text())
    rows = {s: {r["offset"]: r for r in read_jsonl(INDEX / f"{s}.jsonl")}
            for s in ("train", *sel["trays"])}

    cat_items, cat_meta = [], []
    for c, offs in sel["catalog"].items():
        for o in offs:
            box = rows["train"][o]["bbox"][0]
            for rot in ROTATIONS:
                cat_items.append((CATALOG / f"{int(c):03d}" / f"{o}.jpg", box, rot))
            cat_meta.append((int(c), o))
    gt_items, gt_meta, det_items, det_meta = [], [], [], []
    for split, levels in sel["trays"].items():
        for lv, offs in levels.items():
            for o in offs:
                path = TRAYS / split / f"{o}.jpg"
                for box, c in zip(rows[split][o]["bbox"], rows[split][o]["category"]):
                    gt_items.append((path, box, 0))
                    gt_meta.append((split, o, c))
                for d in det[split][str(o)]:
                    det_items.append((path, d[:4], 0))
                    det_meta.append((split, o))

    for name in names:
        t0 = time.time()
        emb = Embedder(name, device)
        cat = run(emb, cat_items)
        gt, dt = run(emb, gt_items), run(emb, det_items)
        arrays = {f"catalog_{k}": v.reshape(len(cat_meta), len(ROTATIONS), -1)
                  for k, v in cat.items()}
        arrays.update({f"tray_gt_{k}": v for k, v in gt.items()})
        arrays.update({f"tray_det_{k}": v for k, v in dt.items()})
        np.savez_compressed(
            OUT / f"emb_{name}.npz", **arrays,
            catalog_meta=np.array(cat_meta),                     # (product, row offset)
            tray_gt_split=np.array([m[0] for m in gt_meta]),
            tray_gt_meta=np.array([m[1:] for m in gt_meta]),     # (tray offset, product)
            tray_det_split=np.array([m[0] for m in det_meta]),
            tray_det_meta=np.array([m[1] for m in det_meta]))    # tray offset
        print(f"{name}: {len(cat_items)} catalog crops, {len(gt_items)} tray products, "
              f"{len(det_items)} detections in {time.time() - t0:.0f} s on {device}", flush=True)
        del emb
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
