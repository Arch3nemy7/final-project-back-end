#!/usr/bin/env python
"""Seed run r-imported-378 with the values reported in the paper, marking every
pipeline stage complete. Values are transcribed from:
  - Table I   (dataset crops)
  - Table IV  (per-tick FID for GN/GP, best checkpoints)
  - Table V   (macro-F1 per architecture x scenario)

FID curve x-axis is stored in kimg (1 tick = 4 kimg; the UI divides by 4 to show
ticks), so x = tick * 4.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from importer import _save_sample  # downscales huge fakes grids to light JPEGs

RUN_ID = "r-imported-378"
API = "http://localhost:8000"
RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / RUN_ID

K = 4  # kimg per tick

# --- Table IV: per-tick FID (tick -> fid) -------------------------------------
GN = {0: 302.62, 50: 18.46, 100: 15.11, 150: 13.68, 200: 12.83, 250: 13.86,
      300: 12.60, 350: 13.14, 400: 11.34, 450: 10.82, 500: 10.88, 550: 9.74,
      600: 8.88, 650: 9.81, 700: 9.86, 750: 8.91, 800: 8.72, 850: 7.88,
      900: 7.35, 950: 7.88, 1000: 7.63}
GP = {0: 298.35, 50: 26.00, 100: 15.19, 150: 13.75, 200: 11.83, 250: 10.69,
      300: 9.55, 350: 10.17, 400: 9.55, 450: 9.24, 500: 8.93, 550: 9.09,
      600: 8.56, 650: 8.72, 700: 8.22, 750: 8.40, 800: 8.20, 850: 8.64,
      900: 7.80, 950: 7.74, 1000: 7.82}


def curve(d):
    return [{"x": t * K, "y": v} for t, v in sorted(d.items())]


# best checkpoints (Table IV daggers): GN tick 900 (7.35), GP tick 950 (7.74)
GN_BEST_TICK, GN_BEST_FID = 900, 7.35
GP_BEST_TICK, GP_BEST_FID = 950, 7.74

art = lambda rel: f"/api/runs/{RUN_ID}/artifacts/{rel}"

results = {
    "format": {"res": 256, "cropsPos": 6054, "cropsNeg": 13141},  # Table I, combined totals
    "train": {
        "gn": {"bestFid": GN_BEST_FID, "bestTick": GN_BEST_TICK * K, "sample": art("gn/sample.png")},
        "gp": {"bestFid": GP_BEST_FID, "bestTick": GP_BEST_TICK * K, "sample": art("gp/sample.png")},
        "curveGN": curve(GN),
        "curveGP": curve(GP),
    },
    "fidelity": {
        "gn": GN_BEST_FID, "gp": GP_BEST_FID,
        "gnTick": GN_BEST_TICK * K, "gpTick": GP_BEST_TICK * K,
    },
    "feasibility": {
        "archs": ["ResNet-50", "DenseNet-121", "VGG-16", "MobileNetV3", "InceptionV3"],
        "rows": [                                  # Table V: [Sc I, II, III, IV]
            [0.9425, 0.9245, 0.9150, 0.9008],      # ResNet-50
            [0.9466, 0.9431, 0.9320, 0.8958],      # DenseNet-121
            [0.8497, 0.8547, 0.4064, 0.4064],      # VGG-16 (collapse at III/IV)
            [0.9464, 0.9335, 0.9105, 0.9115],      # MobileNetV3
            [0.9323, 0.9247, 0.9202, 0.8980],      # InceptionV3
        ],
        "top": "DenseNet-121", "topF1": 0.9466,    # highest macro-F1 in the table
    },
    "generate": {
        "total": 10000,                            # 5,000 per class
        "gallery": [
            {"src": art("gn/sample.png"), "cls": "neg"},
            {"src": art("gp/sample.png"), "cls": "pos"},
        ],
    },
}

# --- copy every snapshot's fakes grid so each checkpoint has a sample image ----
def _kimg(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def copy_fakes(cls):
    src = Path((RUN_DIR / cls / "SRC_DIR.txt").read_text().strip())
    sdir = RUN_DIR / cls / "trainsamples"
    sdir.mkdir(parents=True, exist_ok=True)
    for old in sdir.glob("*.png"):      # drop earlier full-size copies
        old.unlink()
    out = []
    for f in sorted(src.glob("fakes*.png"), key=lambda p: _kimg(p.name)):
        k = _kimg(f.name)
        if k < 0:                       # skip fakes_init.png
            continue
        if _save_sample(f, sdir / f"{k}.jpg"):
            out.append({"kimg": k, "url": art(f"{cls}/trainsamples/{k}.jpg")})
    return out


gn_samples, gp_samples = copy_fakes("gn"), copy_fakes("gp")
results["train"]["trainSamplesGN"] = gn_samples
results["train"]["trainSamplesGP"] = gp_samples
# point the single "latest grid" alias at the downscaled jpg, and drop the old
# full-size sample.png copies (~29 MB each) left by the original import.
if gn_samples:
    results["train"]["gn"]["sample"] = gn_samples[-1]["url"]
if gp_samples:
    results["train"]["gp"]["sample"] = gp_samples[-1]["url"]
for cls in ("gn", "gp"):
    old = RUN_DIR / cls / "sample.png"
    if old.exists():
        old.unlink()
# per-class training duration (kimg) — gn and gp trained for different lengths
results["train"]["gn"]["durKimg"] = gn_samples[-1]["kimg"] if gn_samples else GN_BEST_TICK * K
results["train"]["gp"]["durKimg"] = gp_samples[-1]["kimg"] if gp_samples else GP_BEST_TICK * K
print(f"copied {len(gn_samples)} gn + {len(gp_samples)} gp checkpoint samples; "
      f"dur gn={results['train']['gn']['durKimg']} gp={results['train']['gp']['durKimg']} kimg")

# 1) write results.json (served by GET /api/runs/{id}/results)
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "results.json").write_text(json.dumps(results, indent=2))
print("wrote", RUN_DIR / "results.json")

# 2) update the run's pipe in the DB (all stages done + results) via the API
run = json.loads(urllib.request.urlopen(f"{API}/api/runs/{RUN_ID}").read())
cfg = dict(run.get("config") or {})
cfg["ticks"] = "4000"          # 1,000 ticks x 4 = 4,000 kimg (paper training duration)
cfg["metrics"] = "fid50k_full"
pipe = dict(run.get("pipe") or {})
pipe.update({
    "stage": 5, "done": [True, True, True, True, True, True],
    "fmtDone": True, "gnDone": True, "gpDone": True,
    "fidDone": True, "generated": True, "feasDone": True,
    "cfg": cfg, "genN": "5000",
    "posFile": {"name": "gram_positive.zip", "mb": "—", "count": 6054},
    "negFile": {"name": "gram_negative.zip", "mb": "—", "count": 13141},
    "fidData": {"gn": "", "gp": "", "num": "50000"},
    "results": results,
})
payload = {
    "id": RUN_ID,
    "name": run.get("name") or "Imported · PLA generators",
    "dataset": run.get("dataset") or "Imported StyleGAN2-ADA · 256² · ffhq256",
    "config": cfg,
    "pipe": pipe,
}
req = urllib.request.Request(f"{API}/api/runs", method="POST",
                            data=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())
print("upserted run; pipe.done =", resp.get("pipe", {}).get("done"))
