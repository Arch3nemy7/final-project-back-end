#!/usr/bin/env python
"""Create the staged DEMO runs that show the app frozen at each pipeline
checkpoint (the "x/6" progress states):

    r-demo-1-start        0/6  new run, about to train
    r-demo-2-trained      2/6  trained        -> ready for fidelity
    r-demo-3-fidelity     3/6  fidelity done  -> ready to generate
    r-demo-4-generated    4/6  generated      -> ready for feasibility
    r-demo-5-feasibility  5/6  feasibility done -> ready for results
    (6/6 complete = r-imported-378, seeded by seed_paper_results.py)

Each run reuses the completed run's REAL results + artifacts (no image copies),
truncated to the stages it has finished — so the demos use real numbers/images.
Run seed_paper_results.py FIRST, then:

    cd /workspace/final-project-back-end && python scripts/seed_scenarios.py
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
FULL_ID = "r-imported-378"


def api_post(path, payload):
    req = urllib.request.Request(
        API + path, method="POST",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


full_path = settings.runs_dir / FULL_ID / "results.json"
if not full_path.is_file():
    sys.exit(f"ERROR: {full_path} not found — run seed_paper_results.py first.")
full = json.loads(full_path.read_text())   # artifact URLs already point at FULL_ID

# A "trained but not yet FID-tested" train result: keep the checkpoint samples but
# drop best FID / curve, exactly like a real metrics=none training run. The best
# FID only appears after the fidelity test.
train_no_fid = json.loads(json.dumps(full.get("train", {})))
for _c in ("gn", "gp"):
    if _c in train_no_fid:
        train_no_fid[_c] = {**train_no_fid[_c], "bestFid": None, "bestTick": None}
for _k in ("curveGN", "curveGP", "fidSamplesGN", "fidSamplesGP"):
    train_no_fid.pop(_k, None)


def build_results(keys):
    """Cumulative results truncated to the stages a demo run has completed."""
    r = {}
    if "format" in keys:
        r["format"] = full.get("format")
    if "train" in keys:
        r["train"] = full.get("train") if "fidelity" in keys else train_no_fid
    if "fidelity" in keys:
        r["fidelity"] = full.get("fidelity")
    if "feasibility" in keys:
        r["feasibility"] = full.get("feasibility")
        if "generate" in full:
            r["generate"] = full["generate"]   # synth gallery, shown in Results
    return r


BASE_CFG = {
    "cfg": "auto", "res": "256", "ticks": "8800", "snap": "50", "batch": "32", "gpus": "1",
    "aug": "ada", "target": "0.6", "resume": "ffhq256", "augpipe": "bgc", "gamma": "",
    "metrics": "none", "seed": "0", "mirror": False, "subset": "", "freezed": "0",
    "fp32": False, "tf32": False, "workers": "3",
}

# 5-stage pipeline: Format · Train · Fidelity · Feasibility · Results.
# (id, name, done[5], stage, stage-flags, completed-result-keys)
SCEN = [
    ("r-demo-1-start", "① New run · about to train (0/5)",
        [False, False, False, False, False], 0, {}, []),
    ("r-demo-2-trained", "② Trained · ready for fidelity (2/5)",
        [True, True, False, False, False], 2,
        {"fmtDone": True, "gnDone": True, "gpDone": True},
        ["format", "train"]),
    ("r-demo-3-fidelity", "③ Fidelity done · ready for feasibility (3/5)",
        [True, True, True, False, False], 3,
        {"fmtDone": True, "gnDone": True, "gpDone": True, "fidDone": True},
        ["format", "train", "fidelity"]),
    ("r-demo-4-feasibility", "④ Feasibility done · ready for results (4/5)",
        [True, True, True, True, False], 4,
        {"fmtDone": True, "gnDone": True, "gpDone": True, "fidDone": True, "feasDone": True},
        ["format", "train", "fidelity", "feasibility"]),
]

for rid, name, done, stage, flags, keys in SCEN:
    # Results truncated to the stages this run has completed — so e.g. a trained
    # run shows no best FID until the fidelity test runs. The demo replay fills in
    # later stages from the canonical completed run.
    res = build_results(keys)
    run_dir = settings.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(res, indent=2))
    (run_dir / ".demo").touch()   # mark as a presentation run -> stages replay, no GPU
    pipe = {
        "stage": stage, "done": done,
        "fmtDone": False, "gnDone": False, "gpDone": False,
        "fidDone": False, "generated": False, "feasDone": False,
        "cfg": BASE_CFG, "genN": "5000",
        "posFile": {"name": "gram_positive.zip", "mb": "—", "count": RD.CROPS_POS},
        "negFile": {"name": "gram_negative.zip", "mb": "—", "count": RD.CROPS_NEG},
        "fidData": {"gn": "", "gp": "", "num": "50000"},
        "feasData": {"real": "", "test": ""},
        "results": res,
    }
    pipe.update(flags)
    resp = api_post("/api/runs", {
        "id": rid, "name": name,
        "dataset": "StyleGAN2-ADA · 256² · ffhq256", "config": BASE_CFG, "pipe": pipe})
    print(f"seeded {rid:22s} {sum(done)}/5  stage={stage}  results={list(res.keys())}")

# Snapshot all template runs so POST /api/reset-template can restore them.
TEMPLATE_IDS = [FULL_ID] + [s[0] for s in SCEN]
for tid in TEMPLATE_IDS:                       # the completed run replays its stages too
    (settings.runs_dir / tid).mkdir(parents=True, exist_ok=True)
    (settings.runs_dir / tid / ".demo").touch()
snap = [json.loads(urllib.request.urlopen(f"{API}/api/runs/{tid}").read()) for tid in TEMPLATE_IDS]
(settings.runs_dir / "_template_snapshot.json").write_text(json.dumps(snap, indent=2))
print(f"done — {len(SCEN)} staged demo runs (5/5 = {FULL_ID}); wrote template snapshot ({len(snap)} runs).")
