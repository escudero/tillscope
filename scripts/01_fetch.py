"""Step 1: fetch only the part of RPC the study needs, row by row, from the Hugging Face rows API.

The full dataset is 15.3 GB of Parquet. The study needs a few catalog photos per product and a
few hundred checkout trays, about 0.4 GB, so it reads the rows API instead.

  python scripts/01_fetch.py index [split ...]  # labels of every row (JSON, no images): ~840 calls
  python scripts/01_fetch.py select    # which catalog photos and trays to use
  python scripts/01_fetch.py download  # the selected images (signed URLs, fetched fresh)
"""
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

from common import CATALOG, DATASET, INDEX, RAW, SELECTION, TRAYS, write_json

# --- parameters -------------------------------------------------------------------------------
API = "https://datasets-server.huggingface.co"
PAGE = 100                 # rows per call (API maximum)
PAUSE = 0.6                # pause after each rows call while indexing
MIN_INTERVAL = 0.4         # at most one request (rows call or photo) every 0.4 s: the photo
                           # links are on the same server, and it answers 429 above ~1000 per 5 min
WORKERS = 4                # parallel photo downloads
SPLITS = {"train": 53739, "validation": 6000, "test": 24000}
VIEWS_PER_PRODUCT = 12     # catalog photos kept per product, spread over the product's block
CAMERAS = ["side", "45deg", "top", "30deg"]   # order of the camera blocks inside each product
TRAYS_PER_LEVEL = {"validation": 30, "test": 100}   # tuning set and reported set
SEED = 0
# Clutter level is not in the Hugging Face copy. The RPC paper (Table 2) defines it by
# instances per photo: easy 3-10, medium 10-15, hard 15-20. The ranges touch, so we bin by count.
LEVELS = {"easy": (3, 10), "medium": (11, 15), "hard": (16, 99)}
# -----------------------------------------------------------------------------------------------


def auth():
    """Your Hugging Face token, if you are logged in: the rows API allows more calls with it."""
    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        token = None
    return {"Authorization": f"Bearer {token}"} if token else {}


HEADERS = auth()
_lock, _last = threading.Lock(), [0.0]


def request(url, params=None, tries=20):
    """GET with a shared pace for all threads; on 429 (rate limit) wait a minute and retry."""
    for k in range(tries):
        with _lock:
            time.sleep(max(0.0, _last[0] + MIN_INTERVAL - time.time()))
            _last[0] = time.time()
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r
            wait = 60 if r.status_code == 429 else 2 * (k + 1)
        except requests.RequestException:
            wait = 2 * (k + 1)
        time.sleep(wait)
    raise SystemExit(f"request failed: {url} {params}")


def get(path, params):
    return request(f"{API}/{path}", params).json()


def rows(split, offset, length):
    return get("rows", {"dataset": DATASET, "config": "default", "split": split,
                        "offset": offset, "length": length})


def index():
    INDEX.mkdir(parents=True, exist_ok=True)
    info = get("info", {"dataset": DATASET})
    names = info["dataset_info"]["default"]["features"]["objects"]["category"][0]["names"]
    write_json(RAW / "categories.json", names)
    for split, total in SPLITS.items():
        if len(sys.argv) > 2 and split not in sys.argv[2:]:
            continue
        path = INDEX / f"{split}.jsonl"
        done = sum(1 for _ in open(path)) if path.exists() else 0
        with open(path, "a") as f:
            for off in range(done, total, PAGE):
                for r in rows(split, off, PAGE)["rows"]:
                    o = r["row"]["objects"]
                    f.write(json.dumps({"offset": r["row_idx"], "category": o["category"],
                                        "bbox": [[round(v, 1) for v in b] for b in o["bbox"]],
                                        "size": [r["row"]["image"]["width"],
                                                 r["row"]["image"]["height"]]}) + "\n")
                f.flush()
                print(f"\r{split}: {min(off + PAGE, total)}/{total}", end="", flush=True)
                time.sleep(PAUSE)
        print()


def level_of(n):
    return next(k for k, (lo, hi) in LEVELS.items() if lo <= n <= hi)


def select():
    from common import read_jsonl
    rng = random.Random(SEED)
    sel = {"catalog": {}, "trays": {}}
    # catalog: each product is a contiguous block of rows, 40 per camera, camera after camera
    # (side, 45 degrees, top, 30 degrees; twice for boxes and bags, one pass per side).
    # Keeping VIEWS_PER_PRODUCT rows evenly spread over the block keeps every camera.
    blocks = defaultdict(list)
    for r in read_jsonl(INDEX / "train.jsonl"):
        blocks[r["category"][0]].append(r["offset"])
    sel["camera"] = {}
    per_cam = VIEWS_PER_PRODUCT // len(CAMERAS)
    for c, offs in sorted(blocks.items()):
        offs.sort()
        passes = max(1, round(len(offs) / 160))   # 160 rows = 4 cameras x 40 turntable steps
        seg = len(offs) / (4 * passes)
        sel["catalog"][c], sel["camera"][c] = [], []
        for k, cam in enumerate(CAMERAS):
            rows_k = [i for p in range(passes)
                      for i in range(int((4 * p + k) * seg), int((4 * p + k + 1) * seg))]
            step = len(rows_k) / per_cam
            for j in range(per_cam):
                sel["catalog"][c].append(offs[rows_k[int(j * step + step / 2)]])
                sel["camera"][c].append(cam)
    for split, n in TRAYS_PER_LEVEL.items():
        # RPC photographs each basket 2 or 3 times, rearranged on the tray: keep one photo per
        # basket, so the sample does not count the same purchase twice
        by_level, seen = defaultdict(list), set()
        for r in sorted(read_jsonl(INDEX / f"{split}.jsonl"), key=lambda r: r["offset"]):
            basket = tuple(sorted(r["category"]))
            if basket not in seen:
                seen.add(basket)
                by_level[level_of(len(r["category"]))].append(r["offset"])
        sel["trays"][split] = {lv: sorted(rng.sample(v, n)) for lv, v in by_level.items()}
        print(split, {lv: len(v) for lv, v in by_level.items()}, "rows per level")
    write_json(SELECTION, sel)
    print(f"{len(blocks)} products, {sum(map(len, sel['catalog'].values()))} catalog photos; "
          f"trays: { {s: sum(map(len, v.values())) for s, v in sel['trays'].items()} }")


def windows(offsets, span=PAGE):
    """Group sorted row offsets into windows of at most `span` rows: one API call each."""
    out = []
    for o in sorted(offsets):
        if out and o - out[-1][0] < span:
            out[-1].append(o)
        else:
            out.append([o])
    return out


def save(url, dest):
    r = request(url, tries=5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return len(r.content)


def download():
    """One rows call per window of nearby rows (it returns a signed link for each photo), then
    the photos themselves, a few at a time. The links expire, so they are used right away."""
    sel = json.loads(SELECTION.read_text())
    want = defaultdict(dict)  # split -> {offset: destination}; trays first, step 2 needs them
    for split, levels in sel["trays"].items():
        for offs in levels.values():
            for o in offs:
                want[split][o] = TRAYS / split / f"{o}.jpg"
    for c, offs in sel["catalog"].items():
        for o in offs:
            want["train"][o] = CATALOG / f"{int(c):03d}" / f"{o}.jpg"
    todo = {s: {o: d for o, d in m.items() if not d.exists()} for s, m in want.items()}
    n_all, n_new, total = sum(map(len, want.values())), sum(map(len, todo.values())), 0
    print(f"{n_all - n_new} of {n_all} images already here")
    done = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for split, dests in todo.items():
            for win in windows(dests):
                page = rows(split, win[0], win[-1] - win[0] + 1)["rows"]
                links = {r["row_idx"]: r["row"]["image"]["src"] for r in page}
                for got in pool.map(lambda o: save(links[o], dests[o]), win):
                    total += got
                done += len(win)
                print(f"\r{done}/{n_new} images, {total / 1e6:.0f} MB", end="", flush=True)
    print()


if __name__ == "__main__":
    {"index": index, "select": select, "download": download}[sys.argv[1]]()
