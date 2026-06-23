"""Extract the REAL study results measured on THIS instance, for seeding the UI.

Pure helpers shared by:
  - seed_paper_results.py  (the completed 6/6 run)
  - seed_scenarios.py      (the staged demo runs)

Every value comes from an actual file on disk — nothing is fabricated:
  - StyleGAN FID:   stylegan2-ada-pytorch/training-runs/<run>/metric-fid50k_full.jsonl
                    + fid_ranking.json (best checkpoint)
  - Feasibility F1: cnn-pytorch/results/result_scenario_<s>_<arch>.json
                    (test.f1_macro) + per-epoch history[].train_loss
  - Gallery crops:  cnn-pytorch/data/synthetic/<class>/*.png  (real generated images)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent        # final-project-back-end/
WORKSPACE = ROOT.parent                               # /workspace
SG = WORKSPACE / "stylegan2-ada-pytorch" / "training-runs"
GN_RUN = SG / "00001-gram_negative-auto1-resumeffhq256"
GP_RUN = SG / "00001-gram_positive-auto1-resumeffhq256"
CNN = WORKSPACE / "cnn-pytorch"
RESULTS = CNN / "results"
SYNTH = CNN / "data" / "synthetic"

sys.path.insert(0, str(ROOT))
from importer import _save_sample  # downscale a crop/grid to a light JPEG for the UI

# Front-end fixed labels (match src/store/data.js F1_ARCHS); files use the torch names.
ARCH_FILES = ["resnet50", "densenet121", "vgg16", "mobilenet_v3", "inception_v3"]
ARCH_LABELS = ["ResNet-50", "DenseNet-121", "VGG-16", "MobileNetV3", "InceptionV3"]
SCENARIOS = [1, 2, 3, 4]
CROPS_POS, CROPS_NEG = 6054, 13141     # full per-class crop counts (gram_positive/negative.zip)


def _kimg(s):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def fid_curve(run_dir):
    """[{x: kimg, y: fid}] for every evaluated checkpoint, sorted by kimg.
    x is stored in kimg; the UI divides by 4 (kimg_per_tick) to show ticks."""
    pts = []
    for line in (run_dir / "metric-fid50k_full.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pts.append({"x": _kimg(d["snapshot_pkl"]), "y": round(d["results"]["fid50k_full"], 4)})
    pts.sort(key=lambda p: p["x"])
    return pts


def best_fid(run_dir):
    """(kimg, fid) of the lowest-FID checkpoint from fid_ranking.json."""
    b = json.loads((run_dir / "fid_ranking.json").read_text())["best"]
    return b["kimg"], round(b["fid"], 4)


def feasibility():
    """Returns (archs, rows, losses, top, topF1):
      rows[arch][scenario]            = test macro-F1
      losses[arch][scenario][epoch]   = per-epoch training (cross-entropy) loss
      top, topF1                      = architecture with the single highest macro-F1
    """
    rows, losses = [], []
    for af in ARCH_FILES:
        row, lrow = [], []
        for s in SCENARIOS:
            d = json.loads((RESULTS / f"result_scenario_{s}_{af}.json").read_text())
            row.append(round(d["test"]["f1_macro"], 4))
            lrow.append([round(h["train_loss"], 4) for h in d["history"]])
        rows.append(row)
        losses.append(lrow)
    top_gi = max(range(len(rows)), key=lambda i: max(rows[i]))
    topF1 = round(max(max(r) for r in rows), 4)
    return ARCH_LABELS, rows, losses, ARCH_LABELS[top_gi], topF1


def build_gallery(run_dir, run_id, n=24):
    """Copy n real generated crops per class into <run_dir>/gallery as light JPEGs
    and return [{src, cls}] artifact entries. Real files => they load in the UI."""
    gdir = run_dir / "gallery"
    gdir.mkdir(parents=True, exist_ok=True)
    for old in gdir.glob("*"):
        old.unlink()
    items = []
    for folder, cls in [("gram_negative", "neg"), ("gram_positive", "pos")]:
        srcs = sorted((SYNTH / folder).glob("*.png"))[:n]
        for i, src in enumerate(srcs):
            dst = gdir / f"{cls}-{i}.jpg"
            if not _save_sample(src, dst):
                continue
            items.append({"src": f"/api/runs/{run_id}/artifacts/gallery/{cls}-{i}.jpg", "cls": cls})
    return items


def gen_total():
    """Total real synthetic crops generated (gram_negative + gram_positive)."""
    g = len(list((SYNTH / "gram_negative").glob("*.png")))
    p = len(list((SYNTH / "gram_positive").glob("*.png")))
    return g + p
