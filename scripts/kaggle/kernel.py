# tillscope on a Kaggle GPU: steps 1 (download), 2 (detect, if not done), 3 (embed) or 3b.
# Input: the private dataset pushed by run_on_kaggle.py (scripts, selection, index and, if they
# were downloaded locally, the images). Missing images are fetched inside the kernel into /tmp;
# only the outputs go to /kaggle/working.
import glob, os, shutil, subprocess, sys, time

EMBEDDERS = __EMBEDDERS__   # filled in by run_on_kaggle.py: step 3 for these
FULL = __FULL__             # and step 3b (the whole catalog) for these
T0 = time.time()
WORK = "/tmp/tillscope"
OUTDIR = "/kaggle/working"


def sh(cmd):
    print(f"[{(time.time() - T0) / 60:5.1f} min] $ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")


src = os.path.dirname(glob.glob("/kaggle/input/**/selection.json", recursive=True)[0])
os.makedirs(f"{WORK}/data/raw/index", exist_ok=True)
os.makedirs(f"{WORK}/data/output", exist_ok=True)
shutil.copytree(f"{src}/scripts", f"{WORK}/scripts", dirs_exist_ok=True)
for f in ("selection.json", "categories.json"):
    shutil.copy(f"{src}/{f}", f"{WORK}/data/raw/{f}")
for f in glob.glob(f"{src}/index/*.jsonl"):
    shutil.copy(f, f"{WORK}/data/raw/index/")
if os.path.exists(f"{src}/detections.json"):  # step 2 already done locally
    shutil.copy(f"{src}/detections.json", f"{WORK}/data/output/detections.json")
for d in ("catalog", "trays"):  # images uploaded with the dataset, if any; the rest is fetched
    if os.path.isdir(f"{src}/{d}"):
        shutil.copytree(f"{src}/{d}", f"{WORK}/data/raw/{d}", dirs_exist_ok=True)

sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"{sys.executable} -m pip install -q transformers==5.22.1")
if EMBEDDERS or not FULL:  # step 3b reads the photos from Hugging Face itself
    sh(f"cd {WORK} && {sys.executable} scripts/01_fetch.py download")
if not os.path.exists(f"{WORK}/data/output/detections.json"):
    sh(f"cd {WORK} && {sys.executable} scripts/02_detect.py")
if EMBEDDERS or not FULL:  # no names: every model of 03_embed.MODELS
    sh(f"cd {WORK} && {sys.executable} scripts/03_embed.py {EMBEDDERS}")
for name in FULL.split():
    sh(f"cd {WORK} && {sys.executable} scripts/03b_embed_full_catalog.py {name}")
for f in glob.glob(f"{WORK}/data/output/emb_*.npz") + glob.glob(f"{WORK}/data/output/det*.json"):
    shutil.copy(f, OUTDIR)
print(f"done in {(time.time() - T0) / 60:.1f} min", flush=True)
