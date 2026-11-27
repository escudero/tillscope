"""Optional: run steps 1-3 (download, OWLv2, DINOv2 and CLIP) on a free Kaggle GPU.

Needs the Kaggle CLI with credentials (kaggle.com > Settings > API) and a phone-verified
account (GPU and internet in notebooks). Run steps `01_fetch.py index` and `select` locally first.

  python scripts/kaggle/run_on_kaggle.py <kaggle-username> [embedder ...]   # step 3 (default: all)
  python scripts/kaggle/run_on_kaggle.py <kaggle-username> --full <embedder>  # step 3b

Pushes a private dataset (scripts, selection, label index and any images already downloaded)
and a private notebook,
waits for it, and downloads the outputs into data/output/.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "tillscope-steps-1-3"
POLL = 60  # seconds


def kaggle(*args, check=True):
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def main():
    user = sys.argv[1]
    args = sys.argv[2:]
    full = args[args.index("--full") + 1:] if "--full" in args else []
    embedders = " ".join(a for a in args if a != "--full" and a not in full)
    slug = SLUG + ("-full-" + "-".join(full) if full else "")
    ds_id, kn_id = f"{user}/{SLUG}-inputs", f"{user}/{slug}"
    stage = Path(tempfile.mkdtemp(prefix="tillscope-kaggle-"))
    ds, kn = stage / "dataset", stage / "kernel"
    shutil.copytree(ROOT / "scripts", ds / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "kaggle"))
    for f in ("selection.json", "categories.json"):
        shutil.copy(ROOT / "data/raw" / f, ds / f)
    shutil.copytree(ROOT / "data/raw/index", ds / "index")
    if (ROOT / "data/output/detections.json").exists():  # skip step 2 when it is done
        shutil.copy(ROOT / "data/output/detections.json", ds / "detections.json")
    for d in ("catalog", "trays"):  # send the images when they are already here (~0.45 GB)
        if (ROOT / "data/raw" / d).is_dir() and not full:
            shutil.copytree(ROOT / "data/raw" / d, ds / d)
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"{SLUG}-inputs", "id": ds_id, "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}))
    if "ready" in kaggle("datasets", "status", ds_id, check=False):
        kaggle("datasets", "version", "-p", str(ds), "-r", "zip", "-m", "update")
    else:
        kaggle("datasets", "create", "-p", str(ds), "-r", "zip")
    while "ready" not in kaggle("datasets", "status", ds_id, check=False):
        time.sleep(15)
    time.sleep(30)  # a new dataset version takes a moment to be mountable

    kn.mkdir()
    code = (Path(__file__).parent / "kernel.py").read_text()
    (kn / "kernel.py").write_text(code.replace("__EMBEDDERS__", repr(embedders))
                                  .replace("__FULL__", repr(" ".join(full))))
    (kn / "kernel-metadata.json").write_text(json.dumps({
        "id": kn_id, "title": slug, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [ds_id], "machine_shape": "NvidiaTeslaT4"}))
    kaggle("kernels", "push", "-p", str(kn))
    print(f"pushed https://www.kaggle.com/code/{kn_id}", flush=True)
    while True:
        time.sleep(POLL)
        st = kaggle("kernels", "status", kn_id, check=False)
        print(time.strftime("%H:%M"), st.strip().splitlines()[-1], flush=True)
        if "COMPLETE" in st.upper() or "ERROR" in st.upper() or "CANCEL" in st.upper():
            break
    out = ROOT / "data/output"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kn_id, "-p", str(out), "-o")
    print("outputs in", out)


if __name__ == "__main__":
    main()
