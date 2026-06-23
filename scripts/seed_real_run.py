#!/usr/bin/env python
"""Seed ONE real-execution run (not a demo replay): a fresh 0/5 run wired with
predefined datasets, so you can click straight through the REAL pipeline on the
GPU — real training, real fidelity sweep, then real synthesis + the 5-CNN
feasibility study — without entering any dataset paths.

Datasets are pre-staged in the server's data/ dir:
  data/gram_positive.zip, data/gram_negative.zip   GAN training crops (Format/Train)
  data/real  -> real_base/train                     classifier training crops (Feasibility)
  data/test  -> real_base/test                       held-out test split (Feasibility)

The run carries a `.real` marker so `Restore demo template` never deletes/resets it,
and it has NO `.demo` marker, so every stage runs for real on the GPU.

Run with the backend up:  cd /workspace/final-project-back-end && python scripts/seed_real_run.py
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from config import settings
import _realdata as RD

API = "http://localhost:8000"
RUN_ID = "r-real-pipeline"
NAME = "⚡ Real pipeline · full GPU run (predefined datasets)"

# Full thesis-length training; metrics=none so training is fast (FID is measured
# by the Fidelity stage instead), snapshots every 50 ticks for the FID sweep.
CFG = {
    "cfg": "auto", "res": "256", "ticks": "8800", "snap": "50", "batch": "32", "gpus": "1",
    "aug": "ada", "target": "0.6", "resume": "ffhq256", "augpipe": "bgc", "gamma": "",
    "metrics": "none", "seed": "0", "mirror": False, "subset": "", "freezed": "0",
    "fp32": False, "tf32": False, "workers": "3",
}

pipe = {
    "stage": 0, "done": [False, False, False, False, False],
    "fmtDone": False, "gnDone": False, "gpDone": False,
    "fidDone": False, "generated": False, "feasDone": False,
    "cfg": CFG, "genN": "2000",
    "posFile": {"name": "gram_positive.zip", "mb": "—", "count": RD.CROPS_POS},
    "negFile": {"name": "gram_negative.zip", "mb": "—", "count": RD.CROPS_NEG},
    # blank -> the backend falls back to the pre-staged datasets (no input needed)
    "fidData": {"gn": "", "gp": "", "num": "5000"},
    "feasData": {"real": "", "test": ""},
    "results": {},
}

req = urllib.request.Request(
    API + "/api/runs", method="POST",
    data=json.dumps({
        "id": RUN_ID, "name": NAME,
        "dataset": "Real · StyleGAN2-ADA 256² + 5-CNN feasibility (predefined)",
        "config": CFG, "pipe": pipe}).encode(),
    headers={"Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(req).read())

run_dir = settings.runs_dir / RUN_ID
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "results.json").write_text("{}")
(run_dir / ".real").touch()                 # preserved by reset-template
if (run_dir / ".demo").exists():
    (run_dir / ".demo").unlink()            # NOT a demo -> real execution

print(f"seeded REAL run {RUN_ID}: stage 0/5, done={resp['pipe']['done']}")
print("  datasets predefined: gram_positive.zip / gram_negative.zip + data/real + data/test")
print("  .real marker set (survives Restore demo template); no .demo -> real GPU execution")
