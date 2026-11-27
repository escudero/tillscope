"""Step 3b (optional, GPU): vectors of the WHOLE RPC catalog, all 53,739 photos, not only 12 per
product. It reads the 19 catalog Parquet files from Hugging Face one at a time (9.2 GB in all;
each is deleted after use), crops each product by its RPC box, in 4 rotations, and embeds it.

Writes data/output/emb_full_<name>.npz: catalog_cls (53739, 4, dim) float16, catalog_meta
(product, row offset) and catalog_camera. Needs `pyarrow` and `huggingface_hub`.
It took ~1 h on a Kaggle T4 (scripts/kaggle/run_on_kaggle.py --full-catalog).

  python scripts/03b_embed_full_catalog.py <name>     # a name from 03_embed.MODELS
"""
import importlib
import io
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

from common import DATASET, INDEX, OUT, read_jsonl

embed = importlib.import_module("03_embed")

# --- parameters -------------------------------------------------------------------------------
SHARDS = [f"data/train-{i:05d}-of-00019.parquet" for i in range(19)]
CAMERAS = ["side", "45deg", "top", "30deg"]   # order of the camera blocks inside each product
WORKERS = 4                                   # processes decoding and cropping the photos
BATCH = 64                                    # photos per step (x4 rotations on the GPU)
# -----------------------------------------------------------------------------------------------

SIZE = None


def cameras():
    """Camera of every catalog row: each product is a block of 160 rows per side photographed
    (40 per camera, camera after camera), so the position in the block gives the camera."""
    blocks = defaultdict(list)
    for r in read_jsonl(INDEX / "train.jsonl"):
        blocks[r["category"][0]].append(r["offset"])
    cam = {}
    for offs in blocks.values():
        offs.sort()
        passes = max(1, round(len(offs) / 160))
        seg = len(offs) / (4 * passes)
        for i, o in enumerate(offs):
            cam[o] = CAMERAS[int(i / seg) % 4]
    return cam


def crops(job):
    data, box = job
    img = Image.open(io.BytesIO(data)).convert("RGB")
    c = embed.square_crop(img, box, SIZE)
    return [c.rotate(r) if r else c for r in embed.ROTATIONS]


def init(size):
    global SIZE
    SIZE = size


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    name = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam = cameras()
    truth = {r["offset"]: r["category"][0] for r in read_jsonl(INDEX / "train.jsonl")}
    feats, meta, offset, t0 = [], [], 0, time.time()
    # the pool forks before the model touches the GPU
    with Pool(WORKERS, initializer=init, initargs=(embed.MODELS[name][1],)) as pool:
        emb = embed.Embedder(name, device)
        for shard in SHARDS:
            path = hf_hub_download(DATASET, shard, repo_type="dataset")
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=BATCH, columns=["image", "objects"]):
                rows = batch.to_pylist()
                jobs = [(r["image"]["bytes"], r["objects"]["bbox"][0]) for r in rows]
                flat = [c for cs in pool.map(crops, jobs) for c in cs]
                v = emb(flat)["cls"].astype(np.float16)
                feats.append(v.reshape(len(rows), len(embed.ROTATIONS), -1))
                for r in rows:
                    c = r["objects"]["category"][0]
                    assert truth[offset] == c, f"row {offset}: Parquet order differs from the index"
                    meta.append((c, offset))
                    offset += 1
            os.remove(os.path.realpath(path))  # keep the disk free: 0.4-0.5 GB per shard
            print(f"{shard}: {offset} photos, {(time.time() - t0) / 60:.1f} min", flush=True)
    meta = np.array(meta)
    np.savez(OUT / f"emb_full_{name}.npz", catalog_cls=np.concatenate(feats),
             catalog_meta=meta, catalog_camera=np.array([cam[int(o)] for o in meta[:, 1]]))
    print(f"done: {offset} catalog photos in {(time.time() - t0) / 60:.1f} min on {device}")


if __name__ == "__main__":
    main()
