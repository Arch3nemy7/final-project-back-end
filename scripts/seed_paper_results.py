#!/usr/bin/env python
"""Seed the canonical COMPLETED run (r-imported-378 — all 6 pipeline stages done)
with the REAL results measured on THIS instance:

  - Train/Fidelity : best FID + full FID-vs-kimg convergence curve per generator
                     (gram_negative + gram_positive), and the real fakes-grid
                     samples copied per checkpoint by the importer.
  - Feasibility    : the 5-architecture x 4-scenario macro-F1 table AND the
                     per-epoch TRAINING LOSS curves (history[].train_loss).
  - Generate       : a gallery of real generated crops (so every image loads).

Replaces the old hard-coded "paper" placeholders. Run with the backend up:

    cd /workspace/final-project-back-end && python scripts/seed_paper_results.py
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

RUN_ID = "r-imported-378"
API = "http://localhost:8000"
NAME = "Full pipeline · complete (5/5) · real results"


def api_post(path, payload):
    req = urllib.request.Request(
        API + path, method="POST",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


# 1) Import the two real trained generators. This copies each snapshot's fakes
#    grid into runs/<id>/{gn,gp}/trainsamples and writes a partial results.json.
run = api_post("/api/import", {
    "gn": str(RD.GN_RUN), "gp": str(RD.GP_RUN), "id": RUN_ID, "name": NAME})
run_dir = settings.runs_dir / RUN_ID
results = json.loads((run_dir / "results.json").read_text())

# 2) Real best FID + convergence curves (Train + Fidelity steps).
gn_k, gn_fid = RD.best_fid(RD.GN_RUN)
gp_k, gp_fid = RD.best_fid(RD.GP_RUN)
results["train"]["gn"].update({"bestFid": gn_fid, "bestTick": gn_k})
results["train"]["gp"].update({"bestFid": gp_fid, "bestTick": gp_k})
results["train"]["curveGN"] = RD.fid_curve(RD.GN_RUN)
results["train"]["curveGP"] = RD.fid_curve(RD.GP_RUN)
# show the per-checkpoint grids on the Fidelity step too
results["train"]["fidSamplesGN"] = results["train"].get("trainSamplesGN", [])
results["train"]["fidSamplesGP"] = results["train"].get("trainSamplesGP", [])

# 3) Format counts, Fidelity result, Feasibility (F1 + training loss), Generate.
results["format"] = {"res": 256, "cropsPos": RD.CROPS_POS, "cropsNeg": RD.CROPS_NEG}
results["fidelity"] = {"gn": gn_fid, "gp": gp_fid, "gnTick": gn_k, "gpTick": gp_k}
archs, rows, losses, top, topF1 = RD.feasibility()
results["feasibility"] = {"archs": archs, "rows": rows, "losses": losses, "top": top, "topF1": topF1}
gallery = RD.build_gallery(run_dir, RUN_ID, n=24)
results["generate"] = {"total": RD.gen_total(), "gallery": gallery}

(run_dir / "results.json").write_text(json.dumps(results, indent=2))
print(f"wrote {run_dir/'results.json'}  | gallery {len(gallery)} imgs"
      f"  | GN best {gn_fid}@{gn_k}k  GP best {gp_fid}@{gp_k}k  | top {top} {topF1}")

# 4) Mark all 6 stages complete and embed the results, via the API.
cfg = dict(run.get("config") or {})
# Training DURATION = the full run length in kimg (both generators trained to
# 8800 kimg = 2200 ticks), NOT the best-checkpoint kimg. The UI uses cfg.ticks
# as the FID-curve x-extent, so this must span the whole sweep the best is
# picked from — otherwise the curve is clipped at the best checkpoint.
dur = max(results["train"]["gn"].get("durKimg") or 0,
          results["train"]["gp"].get("durKimg") or 0, gn_k, gp_k)
cfg.update({"res": "256", "metrics": "fid50k_full", "ticks": str(dur)})
pipe = dict(run.get("pipe") or {})
pipe.update({
    "stage": 4, "done": [True, True, True, True, True],
    "fmtDone": True, "gnDone": True, "gpDone": True,
    "fidDone": True, "generated": True, "feasDone": True,
    "cfg": cfg, "genN": "5000",
    "posFile": {"name": "gram_positive.zip", "mb": "—", "count": RD.CROPS_POS},
    "negFile": {"name": "gram_negative.zip", "mb": "—", "count": RD.CROPS_NEG},
    "fidData": {"gn": "", "gp": "", "num": "50000"},
    "feasData": {"real": "", "test": ""},
    "results": results,
})
resp = api_post("/api/runs", {
    "id": RUN_ID, "name": NAME,
    "dataset": "StyleGAN2-ADA · 256² · ffhq256 · gn/gp 8800 kimg",
    "config": cfg, "pipe": pipe})
print("seeded", RUN_ID, "| pipe.done =", resp.get("pipe", {}).get("done"))
