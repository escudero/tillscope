"""Paths and helpers shared by the pipeline scripts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "output"
INDEX = RAW / "index"          # one JSON line per dataset row: offset, categories, boxes
CATALOG = RAW / "catalog"      # catalog photos, one folder per product
TRAYS = RAW / "trays"          # checkout tray photos, one folder per split
SELECTION = RAW / "selection.json"

DATASET = "benjamintli/retail-product-checkout"  # free copy of RPC on Hugging Face

# RPC product names are anonymous: "<id>_<supercategory>", 17 supercategories.
# The rows API returns the class index (0..199); NAMES[i] is its label.
NAMES = None


def names():
    global NAMES
    if NAMES is None:
        NAMES = json.loads((RAW / "categories.json").read_text())
    return NAMES


def supercategory(c):
    return names()[c].split("_", 1)[1]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
