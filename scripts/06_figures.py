"""Step 6: the images and charts of the README, in docs/media/.

  python scripts/06_figures.py
"""
import json

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import CATALOG, OUT, ROOT, TRAYS, names, write_json
from tray import ground_truth, match, selection

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MEDIA = ROOT / "docs" / "media"
# --- parameters -------------------------------------------------------------------------------
RIGHT, WRONG, MISSED, CHECK = (80, 225, 255), (255, 170, 50), (255, 70, 70), (255, 235, 175)
BLUE, AQUA, GRAY, DARK = "#257de0", "#00b894", "#b5b5b5", "#666666"
LEVEL_COLORS = {"easy": "#257de0", "medium": "#f47230", "hard": "#00b894"}
TEXT, TEXT2 = "#191919", "#666666"
# RPC paper (Wei et al., 2019), Table 3, cAcc; CommNet v2 from the RPC leaderboard (average only)
RPC_SINGLE = {"easy": 0.0002, "medium": 0.0, "hard": 0.0, "all": 0.0001}
RPC_BEST = {"easy": 0.7317, "medium": 0.5469, "hard": 0.4248, "all": 0.5668}
COMMNET_V2 = 0.9311
# -----------------------------------------------------------------------------------------------


def font(size, bold=False):
    for f in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def short(c):
    i, sup = names()[c].split("_", 1)
    return f"#{i} {sup.replace('_', ' ')}"


def draw_tray(split, offset, items, scale=0.5, labels=True):
    """The tray photo with each box: cyan right, orange wrong name, red dashed = missed."""
    g = ground_truth(split)[offset]
    img = Image.open(TRAYS / split / f"{offset}.jpg").convert("RGB")
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    pairs = match(items, g["bbox"])
    f = font(15, bold=True)
    for k, it in enumerate(items):
        x, y, w, h = [v * scale for v in it["box"]]
        ok = k in pairs and g["category"][pairs[k]] == it["product"]
        col = RIGHT if ok else WRONG
        d.rectangle([x, y, x + w, y + h], outline=col, width=3)
        if labels:
            text = short(it["product"])
            tw = d.textlength(text, font=f)
            d.rectangle([x, y, x + tw + 8, y + 20], fill=col)
            d.text((x + 4, y + 1), text, fill=(6, 14, 22), font=f)
    found = set(pairs.values())
    for gi, (box, c) in enumerate(zip(g["bbox"], g["category"])):
        if gi not in found:
            x, y, w, h = [v * scale for v in box]
            for t in range(0, int(2 * (w + h)), 16):  # dashed outline
                pts = perimeter(x, y, w, h, t, 8)
                d.line(pts, fill=MISSED, width=3)
    return img


def perimeter(x, y, w, h, t, n):
    def at(s):
        s %= 2 * (w + h)
        if s < w:
            return (x + s, y)
        if s < w + h:
            return (x + w, y + s - w)
        if s < 2 * w + h:
            return (x + w - (s - w - h), y + h)
        return (x, y + h - (s - 2 * w - h))
    return [at(t), at(t + n)]


def catalog_photo(product, camera="top", size=150):
    sel = selection()
    offs = [o for o, c in zip(sel["catalog"][str(product)], sel["camera"][str(product)])
            if c == camera]
    from common import INDEX, read_jsonl
    box = next(r["bbox"][0] for r in read_jsonl(INDEX / "train.jsonl") if r["offset"] == offs[0])
    img = Image.open(CATALOG / f"{product:03d}" / f"{offs[0]}.jpg").convert("RGB")
    x, y, w, h = box
    p = 0.08 * max(w, h)
    c = img.crop((int(x - p), int(y - p), int(x + w + p), int(y + h + p)))
    c.thumbnail((size, size))
    sq = Image.new("RGB", (size, size), (10, 20, 28))
    sq.paste(c, ((size - c.width) // 2, (size - c.height) // 2))
    return sq


def hero(res, preds):
    """A hard test tray read right, and some of its crops next to the catalog photo they matched."""
    right = [t for t in res["trays"] if t["bill_right"]]
    t = max(right or res["trays"], key=lambda t: (t["objects"], t["named_right"]))
    items = preds["test"][str(t["offset"])]
    tray = draw_tray("test", t["offset"], items, scale=900 / 1830)
    side, cell = 170, 150
    distinct = []
    for it in sorted(items, key=lambda it: -it["sim"]):
        if it["product"] not in [d["product"] for d in distinct]:
            distinct.append(it)
    distinct = distinct[:5]
    canvas = Image.new("RGB", (tray.width + 2 * side + 40, max(tray.height, 5 * (cell + 24) + 50)),
                       (10, 20, 28))
    canvas.paste(tray, (0, 0))
    d = ImageDraw.Draw(canvas)
    x0 = tray.width + 20
    d.text((x0, 8), "on the tray", fill=(150, 170, 185), font=font(16))
    d.text((x0 + side, 8), "catalog photo (45°)", fill=(150, 170, 185), font=font(16))
    src = Image.open(TRAYS / "test" / f"{t['offset']}.jpg").convert("RGB")
    for k, it in enumerate(distinct):
        y = 36 + k * (cell + 24)
        x, yy, w, h = it["box"]
        c = src.crop((int(x), int(yy), int(x + w), int(yy + h)))
        c.thumbnail((cell, cell))
        canvas.paste(c, (x0 + (cell - c.width) // 2, y + (cell - c.height) // 2))
        canvas.paste(catalog_photo(it["product"], "45deg", size=cell), (x0 + side, y))
        d.text((x0, y + cell + 2), short(it["product"]), fill=RIGHT, font=font(14, bold=True))
    canvas.save(MEDIA / "hero.jpg", quality=90)
    return t["offset"]


def failure(res, preds):
    """A test tray with the typical mistakes."""
    def kinds(t):  # missed, wrong name, extra box: show a tray with all three, not too crowded
        return ((t["found"] < t["objects"]) + (t["named_right"] < t["found"])
                + (t["detections"] > t["found"]))
    bad = [t for t in res["trays"] if not t["bill_right"] and 9 <= t["objects"] <= 13]
    t = max(bad or res["trays"], key=lambda t: (kinds(t), -t["detections"]))
    img = draw_tray("test", t["offset"], preds["test"][str(t["offset"])], scale=900 / 1830)
    img.save(MEDIA / "mistakes.jpg", quality=90)
    return t["offset"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=TEXT2, labelsize=10)
    ax.yaxis.grid(True, color="#eeeeee")
    ax.set_axisbelow(True)


def levels_chart(res):
    lv = ["easy", "medium", "hard", "all"]
    series = [
        ("RPC 2019, trained on the catalog only", [RPC_SINGLE[k] for k in lv], GRAY),
        ("tillscope, no training", [res["levels"][k]["bill_accuracy"] for k in lv], BLUE),
        ("tillscope + attendant checks the least sure items",
         [res["levels"][k]["touched_bill_accuracy"] for k in lv], AQUA),
        ("RPC 2019 best, trained on 200k synthetic trays", [RPC_BEST[k] for k in lv], DARK),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    w = 0.2
    x = np.arange(len(lv))
    for i, (label, vals, col) in enumerate(series):
        pos = x + (i - 1.5) * w
        ax.bar(pos, vals, w * 0.92, color=col, label=label)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.012, f"{100 * v:.0f}%" if v >= 0.005 else "0%", ha="center",
                    va="bottom", fontsize=8.5, color=TEXT)
    ax.axhline(COMMNET_V2, color=TEXT2, lw=1, ls=(0, (4, 3)))
    ax.text(len(lv) - 0.5, COMMNET_V2 + 0.015,
            f"best on the leaderboard, trained: {COMMNET_V2:.0%}", ha="right", fontsize=8.5,
            color=TEXT2)
    ax.set_xticks(x, ["easy\n(3–10 items)", "medium\n(11–15)", "hard\n(16–20)", "all"])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("trays with the whole bill right (cAcc)", color=TEXT2, fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=2)
    fig.tight_layout()
    fig.savefig(MEDIA / "levels.png")
    plt.close(fig)


def attendant_chart(per_level):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for lv, curve in per_level.items():
        xs = [r["touches_per_tray"] for r in curve]
        ys = [r["bill_accuracy"] for r in curve]
        ax.plot(xs, ys, color=LEVEL_COLORS.get(lv, BLUE), lw=2, marker="o", ms=4, label=lv)
        ax.text(xs[-1] + 0.1, ys[-1], lv, color=TEXT, fontsize=9, va="center")
    ax.set_xlabel("items the attendant checks per tray (the least sure ones)", color=TEXT2)
    ax.set_ylabel("trays with the whole bill right", color=TEXT2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(MEDIA / "attendant.png")
    plt.close(fig)


def cameras_chart(res):
    label = {"all": "all 4 cameras", "top": "top only", "side": "side only",
             "45deg": "45° only", "30deg": "30° only", "top+45deg": "top + 45°"}
    rows = sorted(res["cameras"], key=lambda r: r["item_accuracy"])
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    y = np.arange(len(rows))
    ax.barh(y, [r["item_accuracy"] for r in rows], 0.6,
            color=[BLUE if r["cameras"] == "all" else GRAY for r in rows])
    for i, r in enumerate(rows):
        ax.text(r["item_accuracy"] + 0.01, i, f"{100 * r['item_accuracy']:.0f}%", va="center",
                fontsize=9, color=TEXT)
    ax.set_yticks(y, [label.get(r["cameras"], r["cameras"]) for r in rows])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("tray products named right (annotated boxes)", color=TEXT2)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#eeeeee")
    fig.tight_layout()
    fig.savefig(MEDIA / "cameras.png")
    plt.close(fig)


def views_chart(res):
    rows = res["views"]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    xs = [r["views"] for r in rows]
    ys = [r["item_accuracy"] for r in rows]
    ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=5)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.025, f"{100 * y:.0f}%", ha="center", fontsize=9, color=TEXT)
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("catalog photos per product", color=TEXT2)
    ax.set_ylabel("tray products named right", color=TEXT2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    style(ax)
    fig.tight_layout()
    fig.savefig(MEDIA / "views.png")
    plt.close(fig)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "results_test.json").read_text())
    preds = json.loads((OUT / "predictions.json").read_text())
    picked = {"hero": hero(res, preds), "mistakes": failure(res, preds)}
    levels_chart(res)
    attendant_chart(res["touch_curve_by_level"])
    cameras_chart(res)
    views_chart(res)
    write_json(OUT / "figures.json", picked)
    print("figures in", MEDIA, picked)


if __name__ == "__main__":
    main()
