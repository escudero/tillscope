"""Reading a checkout tray from the vectors of steps 2 and 3: name each box, drop the pieces,
build the bill. Shared by 04_recognize.py, 05_evaluate.py and 06_figures.py (pure NumPy)."""
import functools
import json
from collections import Counter

import numpy as np

from common import INDEX, OUT, SELECTION, read_jsonl

N_PRODUCTS = 200


@functools.lru_cache(maxsize=None)
def load_embeddings(model, full=False):
    """Vectors of step 3; with full=True, the catalog part comes from step 3b (all photos)."""
    def read(path):
        z = np.load(path)
        return {k: z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k] for k in z.files}
    emb = read(OUT / f"emb_{model}.npz")
    if full:
        emb = {k: v for k, v in emb.items() if not k.startswith("catalog_")}
        emb.update(read(OUT / f"emb_full_{model}.npz"))
    return emb


def selection():
    return json.loads(SELECTION.read_text())


def ground_truth(split):
    return {r["offset"]: r for r in read_jsonl(INDEX / f"{split}.jsonl")}


class Catalog:
    """The catalog vectors: every view of every product, optionally only some cameras."""

    def __init__(self, emb, feature="cls", cameras=None, rotations=True, views=None):
        x = emb[f"catalog_{feature}"]                  # (views, rotations, dim)
        meta = emb["catalog_meta"]                     # (views, 2): product, row offset
        if "catalog_camera" in emb:                    # the whole catalog (step 3b)
            cam = emb["catalog_camera"]
        else:                                          # the 12 selected views (step 3)
            sel = selection()
            cam_of = {o: cam for c, offs in sel["catalog"].items()
                      for o, cam in zip(offs, sel["camera"][c])}
            cam = np.array([cam_of[int(o)] for _, o in meta])
        self.camera = cam
        keep = np.ones(len(meta), bool)
        if cameras:
            keep &= np.isin(cam, list(cameras))
        if views:  # `views` photos per product, evenly spread over its kept photos
            for p in np.unique(meta[:, 0]):
                idx = np.flatnonzero(keep & (meta[:, 0] == p))
                pick = idx[(np.arange(views) * len(idx) / views + len(idx) / views / 2)
                           .astype(int)] if len(idx) > views else idx
                keep[idx] = False
                keep[pick] = True
        x = x[keep] if rotations else x[keep][:, :1]
        product = np.repeat(meta[keep, 0], x.shape[1])
        order = np.argsort(product, kind="stable")   # one contiguous run of columns per product
        self.product = product[order]
        self.x = np.ascontiguousarray(x.reshape(-1, x.shape[-1])[order], dtype=np.float32)
        self.present, self.starts = np.unique(self.product, return_index=True)

    def scores(self, q, chunk=256):
        """Cosine similarity of each query to each product (its best view): (n, 200)."""
        out = np.full((len(q), N_PRODUCTS), -1.0, np.float32)
        for i in range(0, len(q), chunk):
            s = q[i:i + chunk] @ self.x.T
            out[i:i + chunk, self.present] = np.maximum.reduceat(s, self.starts, axis=1)
        return out


class Matcher:
    """Scores of tray crops against the catalog for one model or several ("dinov2+clip").
    Each model's cosine similarities live in their own range (CLIP's sit around 0.77, DINOv2's
    around 0.36), so with several models each one is standardized by its mean and spread on the
    validation boxes, then averaged. The constants are kept in `norm` (and in config.json)."""

    def __init__(self, models, cameras=None, rotations=True, views=None, full=False, norm=None):
        self.models = models.split("+")
        self.embs = {m: load_embeddings(m, full) for m in self.models}
        self.cats = {m: Catalog(e, "cls", cameras, rotations, views) for m, e in self.embs.items()}
        self.norm = norm or {}
        if len(self.models) > 1:
            for m in self.models:
                if m not in self.norm:
                    s = self._raw(m, "gt", self.embs[m]["tray_gt_split"] == "validation")
                    self.norm[m] = [float(s.mean()), float(s.std())]

    def meta(self, kind):
        e = self.embs[self.models[0]]
        return e[f"tray_{kind}_split"], e[f"tray_{kind}_meta"]

    def _raw(self, m, kind, mask):
        q = self.embs[m][f"tray_{kind}_cls"]
        return self.cats[m].scores(q if mask is None else q[mask])

    def scores(self, kind, mask=None):
        """kind: "gt" (annotated boxes) or "det" (OWLv2 boxes); mask selects rows."""
        if len(self.models) == 1:
            return self._raw(self.models[0], kind, mask)
        return np.mean([(self._raw(m, kind, mask) - self.norm[m][0]) / self.norm[m][1]
                        for m in self.models], axis=0)


def top2(scores):
    order = np.argsort(-scores, axis=1)
    rows = np.arange(len(scores))
    s1, s2 = scores[rows, order[:, 0]], scores[rows, order[:, 1]]
    return order[:, 0], s1, s1 - s2


def inside(a, b):
    """Fraction of box a ([x, y, w, h]) that lies inside box b."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / (a[2] * a[3] + 1e-9)


def iou(a, b):
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def read_tray(dets, scores, det_score=0.2, sim_min=1e9, contain=0.8, dup_iou=0.3, floor=0.1):
    """dets: [[x, y, w, h, score], ...]; scores: (len(dets), 200) product similarities.

    1. A box counts when OWLv2 is sure (score >= det_score), or when it is weaker (>= floor) but
       looks a lot like a catalog product (similarity >= sim_min).
    2. Group: a box holding two or more boxes that OWLv2 is surer of is a group of products,
       dropped.
    3. Piece: a box inside a surer, bigger box with the same name is a piece of that product,
       dropped. Inside a box with another name it stays: a product on top of another.
    4. Double: two boxes with the same name overlapping by IoU >= dup_iou are one product.
    """
    if not len(dets):
        return []
    prod, sim, margin = top2(scores)
    keep = [i for i, d in enumerate(dets)
            if d[4] >= det_score or (d[4] >= floor and sim[i] >= sim_min)]
    items = [{"box": dets[i][:4], "det": dets[i][4], "product": int(prod[i]),
              "sim": float(sim[i]), "margin": float(margin[i])} for i in keep]
    drop = set()
    for a, ia in enumerate(items):
        inner = [b for b, ib in enumerate(items)
                 if b != a and inside(ib["box"], ia["box"]) >= contain]
        if sum(items[b]["det"] >= ia["det"] for b in inner) >= 2:
            drop.add(a)                                                  # group
        else:
            drop.update(b for b in inner if items[b]["product"] == ia["product"]
                        and items[b]["det"] < ia["det"])                 # pieces
    items = [it for k, it in enumerate(items) if k not in drop]
    items.sort(key=lambda it: -it["det"])
    out = []
    for it in items:                                                     # doubles
        if not any(o["product"] == it["product"] and iou(o["box"], it["box"]) >= dup_iou
                   for o in out):
            out.append(it)
    return out


def match(items, gt_boxes, thr=0.5):
    """Greedy one-to-one matching by IoU, most confident box first. Returns {item: gt index}."""
    pairs, used = {}, set()
    for k in sorted(range(len(items)), key=lambda k: -items[k]["det"]):
        best, bi = thr, None
        for g, gb in enumerate(gt_boxes):
            if g not in used:
                v = iou(items[k]["box"], gb)
                if v >= best:
                    best, bi = v, g
        if bi is not None:
            pairs[k] = bi
            used.add(bi)
    return pairs


def bill(items):
    return Counter(it["product"] for it in items)


def touched_bill(items, pairs, gt_products, margin_min):
    """The attendant checks every item named with a margin below margin_min: a real product gets
    its right name, a box that is not a product (a piece, a double) is deleted. Missed products
    stay missed. Returns the corrected bill and how many items were checked."""
    out, touches = Counter(), 0
    for k, it in enumerate(items):
        if it["margin"] < margin_min:
            touches += 1
            if k in pairs:
                out[gt_products[pairs[k]]] += 1
        else:
            out[it["product"]] += 1
    return out, touches


def overlap(boxes):
    """Largest share of a box covered by another box on the tray: products piled up or touching."""
    best = 0.0
    for a in range(len(boxes)):
        for b in range(len(boxes)):
            if a != b:
                best = max(best, inside(boxes[a], boxes[b]))
    return best
