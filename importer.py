"""Import already-trained StyleGAN2-ADA run directories as a completed run.

Shared by both the CLI (scripts/import_models.py) and the POST /api/import
endpoint, so models trained outside this machine can be brought in from the UI.
"""
import json
import re
import shutil
from pathlib import Path

from config import settings
from db import init_db, SessionLocal, upsert_run


def kimg_of(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def _latest(dir_path: Path, pattern: str):
    files = sorted(dir_path.glob(pattern), key=lambda p: kimg_of(p.name))
    return files[-1] if files else None


def _latest_pkl(dir_path: Path):
    """Latest checkpoint that isn't truncated (a crashed/interrupted final
    snapshot can be a fraction of the normal size — skip those)."""
    pkls = sorted(dir_path.glob("network-snapshot-*.pkl"), key=lambda p: kimg_of(p.name))
    if not pkls:
        return None
    mx = max(p.stat().st_size for p in pkls)
    valid = [p for p in pkls if p.stat().st_size >= mx * 0.9]
    return valid[-1] if valid else pkls[-1]


def default_cfg():
    return {
        "cfg": "auto", "res": "256", "ticks": "5000", "snap": "50", "batch": "16", "gpus": "1",
        "aug": "ada", "target": "0.6", "resume": "ffhq256", "augpipe": "bgc", "gamma": "",
        "metrics": "fid50k_full", "seed": "0", "mirror": False, "subset": "", "freezed": "0",
        "fp32": False, "tf32": False, "workers": "3",
    }


def _ingest_class(run_dir: Path, src_dir: Path, cls: str):
    """Copy every snapshot's sample grid + record the checkpoint path.
    Returns (best_pkl_kimg, latest_sample_url, per_checkpoint_samples, duration_kimg)."""
    out = run_dir / cls
    out.mkdir(parents=True, exist_ok=True)
    pkl = _latest_pkl(src_dir)
    tick = kimg_of(pkl.name) if pkl else None
    # Remember where ALL the snapshots live, so the FID sweep can evaluate them.
    (out / "SRC_DIR.txt").write_text(str(src_dir.resolve()))
    if pkl:
        (out / "PKL_PATH.txt").write_text(str(pkl.resolve()))
    # Copy EACH snapshot's fakes grid so the UI can show per-checkpoint samples
    # (not just the final one). fakes_init.png has no kimg and is skipped.
    sdir = out / "trainsamples"
    sdir.mkdir(exist_ok=True)
    samples = []
    for f in sorted(src_dir.glob("fakes*.png"), key=lambda p: kimg_of(p.name)):
        k = kimg_of(f.name)
        if k < 0:
            continue
        ext = "jpg" if _save_sample(f, sdir / f"{k}.jpg") else None
        if ext is None:
            try:
                shutil.copyfile(f, sdir / f"{k}.png"); ext = "png"
            except OSError:
                continue
        samples.append({"kimg": k, "url": f"/api/runs/{run_dir.name}/artifacts/{cls}/trainsamples/{k}.{ext}"})
    sample_url = samples[-1]["url"] if samples else None    # latest grid alias
    duration = samples[-1]["kimg"] if samples else (tick or 0)
    return tick, sample_url, samples, duration


def _save_sample(src_png: Path, dst_jpg: Path, max_px: int = 1536) -> bool:
    """Downscale a (huge) StyleGAN fakes grid to a light JPEG for the UI.
    Best-effort: returns False if Pillow is unavailable so the caller can fall back."""
    try:
        from PIL import Image
        im = Image.open(src_png).convert("RGB")
        w, h = im.size
        if max(w, h) > max_px:
            sc = max_px / max(w, h)
            im = im.resize((round(w * sc), round(h * sc)), Image.LANCZOS)
        im.save(dst_jpg, "JPEG", quality=85)
        return True
    except Exception:  # noqa: BLE001
        return False


def _clean_path(p):
    # Tolerate quoted paths (Windows "Copy as path") + stray whitespace.
    return str(p or "").strip().strip('"').strip("'").strip()


def do_import(gn_dir, gp_dir, run_id="r-imported", name="Imported · PLA generators"):
    """Ingest the two model dirs into a run and register it. Returns the run dict."""
    gn_dir, gp_dir = Path(_clean_path(gn_dir)), Path(_clean_path(gp_dir))
    # tolerate someone pasting a file (e.g. a .pkl) — use its containing folder
    if gn_dir.is_file():
        gn_dir = gn_dir.parent
    if gp_dir.is_file():
        gp_dir = gp_dir.parent
    for d in (gn_dir, gp_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"model directory not found: {d}  (give the run FOLDER, not a file)")

    cfg = default_cfg()
    opts = gn_dir / "training_options.json"
    if opts.is_file():
        try:
            o = json.loads(opts.read_text())
            cfg["batch"] = str(o.get("batch_size", cfg["batch"]))
            cfg["gpus"] = str(o.get("num_gpus", cfg["gpus"]))
            cfg["snap"] = str(o.get("image_snapshot_ticks", cfg["snap"]))
            cfg["res"] = str(o.get("training_set_kwargs", {}).get("resolution", cfg["res"]))
        except (ValueError, OSError):
            pass

    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    gn_tick, gn_sample, gn_samples, gn_dur = _ingest_class(run_dir, gn_dir, "gn")
    gp_tick, gp_sample, gp_samples, gp_dur = _ingest_class(run_dir, gp_dir, "gp")
    cfg["ticks"] = str(max(gn_dur, gp_dur) or 5000)   # FID chart x-extent only

    results = {
        "format": {"res": int(cfg["res"]), "cropsPos": None, "cropsNeg": None},
        "train": {
            "gn": {"bestFid": None, "bestTick": gn_tick, "durKimg": gn_dur, "sample": gn_sample},
            "gp": {"bestFid": None, "bestTick": gp_tick, "durKimg": gp_dur, "sample": gp_sample},
            "trainSamplesGN": gn_samples,
            "trainSamplesGP": gp_samples,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    pipe = {
        "stage": 2, "done": [True, True, False, False, False, False],
        "posFile": {"name": "gram_positive.zip", "mb": "—", "count": None},
        "negFile": {"name": "gram_negative.zip", "mb": "—", "count": None},
        "fmtDone": True, "cfg": cfg, "gnDone": True, "gpDone": True,
        "generated": False, "genN": "5000", "fidDone": False, "feasDone": False,
        "results": results,
    }
    dataset = f"Imported StyleGAN2-ADA · {cfg['res']}² · ffhq256 (gn {gn_tick} / gp {gp_tick} kimg)"

    init_db()
    s = SessionLocal()
    try:
        run = upsert_run(s, run_id, name=name, dataset=dataset, config=cfg, pipe=pipe)
        return run.to_dict()
    finally:
        s.close()
