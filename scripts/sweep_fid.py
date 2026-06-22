#!/usr/bin/env python
"""Score every checkpoint in a run with StyleGAN2-ADA's official FID and report
the best one.

This computes the exact same thing as the canonical command:

    python calc_metrics.py --metrics=fid50k_full --network=<snapshot.pkl> --data=<real.zip>

i.e. `fid50k_full`: 50,000 generated images vs the ENTIRE real dataset, with
NO truncation and NO fixed seed (both are the metric's defaults). The generator
is sampled internally by the metric — nothing is written to disk, and no images
are generated up front.

Streams JSON lines on stdout for the orchestration server:
  {"type":"progress","pct":..,"step":".."}
  {"type":"fid","class":"gn","x":<kimg>,"y":<fid>}
  {"type":"result","key":"sweep","value":{"curve":[...],"best":{kimg,fid,pkl}}}
"""
import argparse
import json
import re
import sys
from pathlib import Path


def emit(o):
    print(json.dumps(o), flush=True)


def kimg_of(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgdir", required=True, help="stylegan2-ada-pytorch checkout")
    ap.add_argument("--ckpt-dir", required=True, help="dir with network-snapshot-*.pkl")
    ap.add_argument("--data", required=True, help="real dataset (zip or dir) for this class")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", default="")
    a = ap.parse_args()

    sys.path.insert(0, a.sgdir)
    import torch
    import dnnlib
    import legacy
    from metrics import metric_main          # the same entry point calc_metrics.py uses

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = sorted(Path(a.ckpt_dir).glob("network-snapshot-*.pkl"), key=lambda p: kimg_of(p.name))
    if ckpts:  # skip truncated checkpoints (a crashed final snapshot is much smaller)
        mx = max(p.stat().st_size for p in ckpts)
        ckpts = [p for p in ckpts if p.stat().st_size >= mx * 0.9]
    if not ckpts:
        emit({"type": "error", "message": f"no checkpoints in {a.ckpt_dir}"})
        sys.exit(1)

    curve, best = [], None
    for i, pkl in enumerate(ckpts):
        k = kimg_of(pkl.name)
        emit({"type": "progress", "pct": i / len(ckpts),
              "step": f"fid50k_full {a.cls} · checkpoint {i + 1}/{len(ckpts)} · kimg {k}"})
        with dnnlib.util.open_url(str(pkl)) as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device)
        dk = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", path=a.data)
        dk.resolution = G.img_resolution
        dk.use_labels = (G.c_dim != 0)
        try:
            # calc_metric runs the registered fid50k_full metric: max_real=None
            # (full real set, cached) + 50k generated, no truncation, no seed.
            res = metric_main.calc_metric(metric="fid50k_full", G=G, dataset_kwargs=dk,
                                          num_gpus=1, rank=0, device=device)
            fid = round(float(res["results"]["fid50k_full"]), 3)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "resolution" in msg.lower():
                msg = (f"real images must be {G.img_resolution}x{G.img_resolution} to match this model "
                       f"(the dataset you gave is a different size)")
            emit({"type": "error", "message": msg})
            sys.exit(1)
        curve.append({"x": k, "y": fid})
        emit({"type": "fid", "class": a.cls, "x": k, "y": fid})
        if best is None or fid < best["fid"]:
            best = {"kimg": k, "fid": fid, "pkl": str(pkl)}
        del G
        if device.type == "cuda":
            torch.cuda.empty_cache()

    res = {"curve": curve, "best": best}
    Path(a.out).write_text(json.dumps(res, indent=2))
    emit({"type": "result", "key": a.cls or "sweep", "value": res})
    emit({"type": "progress", "pct": 1.0})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)
