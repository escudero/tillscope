"""Step 2: find every product on each checkout tray with OWLv2, zero-shot: text prompts, no
training.

Writes data/output/detections.json: {split: {offset: [[x, y, w, h, score], ...]}}.
Boxes are kept with a low score floor; step 4 decides which ones count.

  python scripts/02_detect.py
"""
import json
import time

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from common import OUT, SELECTION, TRAYS, write_json

# --- parameters -------------------------------------------------------------------------------
MODEL = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["a product", "a package", "a bottle", "a box", "a bag", "a cup",
           "a chocolate bar", "a can"]
SCORE_FLOOR = 0.10          # keep weak boxes too; the threshold is tuned on validation in step 4
NMS_IOU = 0.5               # across prompts: one box per object
AREA = (0.002, 0.25)        # drop boxes smaller/larger than this fraction of the photo
# -----------------------------------------------------------------------------------------------


def detect(model, proc, img, device):
    inputs = proc(text=[PROMPTS], images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    side = max(img.size)  # the processor pads the photo to a square before resizing
    res = proc.post_process_grounded_object_detection(
        outputs=out, threshold=SCORE_FLOOR, target_sizes=torch.tensor([[side, side]]))[0]
    keep = nms(res["boxes"].float(), res["scores"].float(), NMS_IOU)
    area_img = img.size[0] * img.size[1]
    dets = []
    for (x1, y1, x2, y2), s in zip(res["boxes"][keep].tolist(), res["scores"][keep].tolist()):
        a = (x2 - x1) * (y2 - y1)
        if AREA[0] * area_img < a < AREA[1] * area_img:
            dets.append([round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1),
                         round(s, 4)])
    return dets


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval()
    sel = json.loads(SELECTION.read_text())
    path = OUT / "detections.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    t0, n = time.time(), 0
    for split, levels in sel["trays"].items():
        result.setdefault(split, {})
        for offs in levels.values():
            for off in offs:
                if str(off) in result[split]:
                    continue
                img = Image.open(TRAYS / split / f"{off}.jpg").convert("RGB")
                result[split][str(off)] = detect(model, proc, img, device)
                n += 1
                if n % 20 == 0:
                    write_json(path, result)
                    print(f"{n} trays, {(time.time() - t0) / n:.2f} s per tray", flush=True)
    write_json(path, result)
    print(f"done: {n} trays on {device}, {(time.time() - t0) / max(n, 1):.2f} s per tray")


if __name__ == "__main__":
    main()
