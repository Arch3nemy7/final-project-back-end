"""Job runner + telemetry.

This module is intentionally dependency-free (standard library only) so the
parsers can be unit-tested without installing FastAPI/SQLAlchemy.

Two execution paths, chosen by ``settings.mock``:

* REAL  — launches the StyleGAN2-ADA scripts as subprocesses and tails the run
          directory (``stats.jsonl`` per tick, ``metric-*.jsonl`` per snapshot,
          ``fakes*.png`` samples), publishing structured events.
* MOCK  — simulates the whole pipeline in-process (no GPU, no scripts) using the
          same event protocol, so the front end can be demoed end-to-end.

Event protocol (one JSON object per SSE message, discriminated by ``type``):
  progress   {pct, step?, phase?}            # format / generate / fidelity / feasibility
  tick       {class, tick, total, eta}       # training, from stats.jsonl
  fid        {class, x, y}                    # training, from metric-*.jsonl
  sample     {class, kimg, url}              # training, a new fakes*.png appeared
  class_done {class}                          # one generator finished (GN then GP)
  done       {}                               # whole stage finished
  cancelled  {}
  error      {message}
"""
import asyncio
import json
import math
import os
import re
import signal
import time
import zipfile
from pathlib import Path

# Known F1 table used only for MOCK feasibility (the real numbers come from
# scripts/feasibility.py on the GPU).
MOCK_F1 = {
    "archs": ["ResNet-50", "DenseNet-121", "VGG-16", "MobileNetV3", "InceptionV3"],
    "rows": [
        [0.9425, 0.9245, 0.9150, 0.9008],
        [0.9466, 0.9431, 0.9320, 0.8958],
        [0.8497, 0.8547, 0.4064, 0.4064],
        [0.9464, 0.9335, 0.9105, 0.9115],
        [0.9323, 0.9247, 0.9202, 0.8980],
    ],
    "top": "DenseNet-121",
    "topF1": 0.9466,
}
# Synthetic per-epoch loss curves (8 epochs) so the UI loss charts render in mock
# mode: losses[arch][scenario] = [mean train loss per epoch], gently decaying.
MOCK_F1["losses"] = [
    [[round(0.72 * (0.62 ** e) + 0.05 + 0.015 * si, 4) for e in range(8)]
     for si in range(4)]
    for _ in range(len(MOCK_F1["archs"]))
]

# --------------------------------------------------------------------------- #
#  Parsers (pure functions — covered by tests/test_parsers.py)
# --------------------------------------------------------------------------- #

def _mean(d: dict, key: str):
    """StyleGAN2-ADA stats values are nested as {'num','mean','std'}."""
    v = d.get(key)
    return v.get("mean") if isinstance(v, dict) else v


def parse_stats_line(line: str):
    """One line of stats.jsonl -> normalised progress dict (or None)."""
    line = (line or "").strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except (ValueError, TypeError):
        return None
    tick, kimg = _mean(d, "Progress/tick"), _mean(d, "Progress/kimg")
    if tick is None and kimg is None:
        return None
    return {
        "tick": int(tick) if tick is not None else None,
        "kimg": float(kimg) if kimg is not None else None,
        "loss_g": _mean(d, "Loss/G/loss"),
        "loss_d": _mean(d, "Loss/D/loss"),
        "sec_per_kimg": _mean(d, "Timing/sec_per_kimg"),
        "sec_per_tick": _mean(d, "Timing/sec_per_tick"),
        "total_sec": _mean(d, "Timing/total_sec"),
        "maintenance": _mean(d, "Timing/maintenance_sec"),
        "cpu_mem": _mean(d, "Resources/cpu_mem_gb"),
        "gpu_mem": _mean(d, "Resources/peak_gpu_mem_gb"),
        "augment": _mean(d, "Progress/augment"),
    }


def kimg_from_name(name: str):
    """'network-snapshot-000200.pkl' / 'fakes000200.png' -> 200."""
    m = re.search(r"(\d+)", name or "")
    return int(m.group(1)) if m else None


def parse_metric_line(line: str):
    """One line of metric-*.jsonl -> {kimg, fid} (or None)."""
    try:
        d = json.loads((line or "").strip())
    except (ValueError, TypeError):
        return None
    results = d.get("results") or {}
    if not results:
        return None
    value = next(iter(results.values()))
    return {"kimg": kimg_from_name(d.get("snapshot_pkl") or ""), "fid": float(value)}


def build_train_args(cfg: dict, data_path, outdir) -> list[str]:
    """Map the front-end config dict to train.py CLI flags."""
    a = ["--outdir", str(outdir), "--data", str(data_path)]

    def add(flag, key):
        v = cfg.get(key)
        if v not in (None, ""):
            a.extend([flag, str(v)])

    add("--cfg", "cfg")
    if cfg.get("ticks"):                       # the UI's "ticks" maps to --kimg
        a.extend(["--kimg", str(int(cfg["ticks"]))])
    add("--snap", "snap")
    add("--batch", "batch")
    add("--gpus", "gpus")
    add("--aug", "aug")
    add("--target", "target")
    add("--resume", "resume")
    add("--augpipe", "augpipe")
    add("--metrics", "metrics")
    add("--seed", "seed")
    add("--freezed", "freezed")
    add("--workers", "workers")
    if cfg.get("gamma"):
        a.extend(["--gamma", str(cfg["gamma"])])
    if cfg.get("subset"):
        a.extend(["--subset", str(cfg["subset"])])
    a += ["--mirror", "true" if cfg.get("mirror") else "false"]
    a += ["--fp32", "true" if cfg.get("fp32") else "false"]
    a += ["--allow-tf32", "true" if cfg.get("tf32") else "false"]
    return a


def _downscale_grid(src_png, dst_jpg, max_px: int = 1536) -> bool:
    """Save a (large) fakes grid as a light JPEG for the UI. Lazy-imports PIL so
    this module stays import-light; returns False if PIL is unavailable."""
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


# --------------------------------------------------------------------------- #
#  SSE broker (one per run)
# --------------------------------------------------------------------------- #

class Broker:
    """Fan-out of a run's events to any number of SSE subscribers, with replay
    of the current job's history so a client that connects late still rebuilds
    the full FID curve."""

    def __init__(self):
        self.subscribers: set[asyncio.Queue] = set()
        self.history: list[dict] = []
        self.active = False

    def begin(self):
        # A new job clears the previous job's history. Clients open their event
        # stream only AFTER the start POST resolves (so begin() has already run),
        # which is what stops a previous job's events leaking into the new stream.
        self.history = []
        self.active = True

    def end(self):
        # Keep history: a stream that opens just after a FAST job finishes must
        # still receive its terminal (done/error/cancelled) event. It's cleared by
        # the next begin(), not here.
        self.active = False

    def publish(self, event: dict):
        if len(self.history) >= 6000:
            self.history = self.history[-3000:]
        self.history.append(event)
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        # Replay the current job's history so a late/re-attaching client rebuilds
        # the live chart AND catches a fast job's terminal event. (begin() cleared
        # any previous job's history before this stream was opened.)
        for e in self.history:
            q.put_nowait(e)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self.subscribers.discard(q)


# --------------------------------------------------------------------------- #
#  Job manager
# --------------------------------------------------------------------------- #

class JobManager:
    def __init__(self, settings):
        self.s = settings
        self.brokers: dict[str, Broker] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.kinds: dict[str, str] = {}     # run_id -> running job kind (for re-attach)
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.cancelled: set[str] = set()    # runs whose current job a user cancelled
        self._aux: set[asyncio.Task] = set()  # keep escalation tasks from being GC'd

    def broker(self, run_id: str) -> Broker:
        return self.brokers.setdefault(run_id, Broker())

    def _emit(self, run_id: str, **ev):
        self.broker(run_id).publish(ev)

    def running_kind(self, run_id: str):
        """The kind of job currently running for a run (format/train/…), or None."""
        t = self.tasks.get(run_id)
        return self.kinds.get(run_id) if (t and not t.done()) else None

    # ---- lifecycle -------------------------------------------------------- #
    def _launch(self, run_id: str, coro_factory, kind):
        old = self.tasks.get(run_id)
        if old and not old.done():
            # A job is already running for this run. Ignore the duplicate start
            # (e.g. a double-click) instead of cancelling the in-flight job —
            # cancelling would surface as the job "stopping itself".
            return
        self.broker(run_id).begin()
        self.kinds[run_id] = kind
        self.tasks[run_id] = asyncio.create_task(self._supervise(run_id, coro_factory))

    async def _supervise(self, run_id: str, coro_factory):
        try:
            await coro_factory()
        except asyncio.CancelledError:
            self._emit(run_id, type="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            # If the user cancelled, the subprocess dying shows up here as a
            # non-zero-exit error first; report it as a clean cancellation.
            if run_id in self.cancelled:
                self._emit(run_id, type="cancelled")
            else:
                self._emit(run_id, type="error", message=str(exc))
        finally:
            self.cancelled.discard(run_id)
            self.broker(run_id).end()
            self.kinds.pop(run_id, None)

    @staticmethod
    def _signal_tree(proc, sig):
        """Signal the whole process group (the job + any children it spawned,
        e.g. torch DataLoader workers), falling back to the lone process."""
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    async def cancel(self, run_id: str):
        # Mark first so _supervise reports a cancellation rather than an error.
        self.cancelled.add(run_id)
        proc = self.procs.get(run_id)
        if proc and proc.returncode is None:
            self._signal_tree(proc, signal.SIGTERM)

            async def _escalate(p):
                try:
                    await asyncio.wait_for(p.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._signal_tree(p, signal.SIGKILL)   # didn't stop -> force kill

            t = asyncio.create_task(_escalate(proc))
            self._aux.add(t)
            t.add_done_callback(self._aux.discard)
        task = self.tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        else:
            self._emit(run_id, type="cancelled")

    # ---- public start methods -------------------------------------------- #
    def start_format(self, run_id, pos, neg, res):
        self._launch(run_id, lambda: self._format(run_id, pos, neg, res), "format")

    def start_train(self, run_id, cfg):
        self._launch(run_id, lambda: self._train(run_id, cfg), "train")

    def start_generate(self, run_id, n):
        self._launch(run_id, lambda: self._generate(run_id, n), "generate")

    def start_fidelity(self, run_id, num=5000, data_override=None):
        self._launch(run_id, lambda: self._fidelity(run_id, num, data_override or {}), "fidelity")

    def start_feasibility(self, run_id, data_override=None):
        self._launch(run_id, lambda: self._feasibility(run_id, data_override or {}), "feasibility")

    # ---- subprocess helpers ---------------------------------------------- #
    async def _run(self, run_id, args, cwd):
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,   # own process group -> cancel can kill the whole tree
        )
        self.procs[run_id] = proc
        return proc

    async def _simple_proc(self, run_id, args, cwd, est_sec, steps=None, phase=None):
        """Run a process whose progress we can't read precisely: report a
        time-based estimate, then snap to 100% when it actually exits."""
        proc = await self._run(run_id, args, cwd)
        t0 = time.monotonic()
        while proc.returncode is None:
            pct = min(0.95, (time.monotonic() - t0) / est_sec)
            ev = {"type": "progress", "pct": pct}
            if phase:
                ev["phase"] = phase
            if steps:
                ev["step"] = steps[min(len(steps) - 1, int(pct * len(steps)))]
            self._emit(run_id, **ev)
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
        if proc.returncode not in (0, None):
            raise RuntimeError(f"process exited with code {proc.returncode}")

    @staticmethod
    def _newest_subdir(outdir: Path):
        if not outdir.exists():
            return None
        subs = sorted([d for d in outdir.iterdir() if d.is_dir() and re.match(r"^\d+", d.name)])
        return subs[-1] if subs else None

    @staticmethod
    def _read_new(path: Path, pos: int, on_line):
        if not path.exists():
            return pos
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(pos)
            for line in f:
                on_line(line)
            return f.tell()

    def _dataset_for(self, run_id, cls):
        # Produced by the format stage; falls back to a staged zip in DATA_DIR.
        rd = self.s.runs_dir / run_id
        candidate = rd / f"dataset_{cls}.zip"
        return candidate if candidate.exists() else self.s.data_dir / f"dataset_{cls}.zip"

    # ---- results (persisted + streamed) ----------------------------------- #
    def _save_result(self, run_id, key, value):
        """Merge one stage's result into the run's results.json AND stream it."""
        rd = self.s.runs_dir / run_id
        rd.mkdir(parents=True, exist_ok=True)
        f = rd / "results.json"
        data = {}
        if f.exists():
            try:
                data = json.loads(f.read_text())
            except (ValueError, OSError):
                data = {}
        data[key] = value
        f.write_text(json.dumps(data, indent=2))
        self._emit(run_id, type="result", key=key, value=value)

    def _best_fid(self, run_id, cls):
        """Scan a class's metric-*.jsonl: return (best_fid, best_kimg, curve)."""
        sub = self._newest_subdir(self.s.runs_dir / run_id / cls)
        best, best_kimg, curve = None, None, []
        if sub:
            for mf in sorted(sub.glob("metric-*.jsonl")):
                for line in mf.read_text(errors="ignore").splitlines():
                    m = parse_metric_line(line)
                    if m and m["fid"] is not None:
                        curve.append({"x": m["kimg"], "y": round(m["fid"], 2)})
                        if best is None or m["fid"] < best:
                            best, best_kimg = m["fid"], m["kimg"]
        return (round(best, 2) if best is not None else None), best_kimg, curve

    def _collect_train_samples(self, run_id, cls):
        """Persist this class's per-checkpoint fakes grids as light JPEGs so they
        survive a page refresh — the live 'sample' SSE events are in-memory only,
        so without this the checkpoint strip is empty after reload."""
        sub = self._newest_subdir(self.s.runs_dir / run_id / cls)
        if not sub:
            return []
        out = self.s.runs_dir / run_id / cls / "trainsamples"
        out.mkdir(parents=True, exist_ok=True)
        samples = []
        for png in sorted(sub.glob("fakes*.png"),
                          key=lambda p: kimg_from_name(p.name) if kimg_from_name(p.name) is not None else -1):
            k = kimg_from_name(png.name)
            if k is None:                      # skip fakes_init.png (no kimg)
                continue
            dst = out / f"{k}.jpg"
            if _downscale_grid(png, dst):
                url = f"/api/runs/{run_id}/artifacts/{cls}/trainsamples/{k}.jpg"
            else:                              # PIL missing -> point at the raw grid
                url = f"/api/runs/{run_id}/artifacts/{cls}/{sub.name}/{png.name}"
            samples.append({"kimg": k, "url": url})
        return samples

    @staticmethod
    def _zip_image_count(path: Path):
        if not path.exists():
            return 0
        try:
            with zipfile.ZipFile(path) as z:
                return sum(1 for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg")))
        except zipfile.BadZipFile:
            return 0

    # ---- REAL training (subprocess + tail) -------------------------------- #
    async def _train(self, run_id, cfg):
        if self.s.mock:
            return await self._mock_train(run_id, cfg)
        if self._is_demo(run_id):
            return await self._demo_train(run_id, cfg)
        total = int(cfg.get("ticks") or 1000)
        rd = self.s.runs_dir / run_id
        for cls in ("gn", "gp"):
            outdir = rd / cls
            outdir.mkdir(parents=True, exist_ok=True)
            args = self._train_command(cfg, self._dataset_for(run_id, cls), outdir)
            proc = await self._run(run_id, args, self.s.stylegan_dir)
            await self._tail_training(run_id, cls, outdir, total, proc)
            if proc.returncode not in (0, None):
                raise RuntimeError(f"train.py ({cls}) exited with {proc.returncode}")
            self._emit(run_id, type="class_done", **{"class": cls})
        gnB, gnK, gnCurve = self._best_fid(run_id, "gn")
        gpB, gpK, gpCurve = self._best_fid(run_id, "gp")
        gn_s = self._collect_train_samples(run_id, "gn")
        gp_s = self._collect_train_samples(run_id, "gp")
        self._save_result(run_id, "train", {
            "gn": {"bestFid": gnB, "bestTick": gnK,
                   "sample": gn_s[-1]["url"] if gn_s else None,
                   "durKimg": gn_s[-1]["kimg"] if gn_s else gnK},
            "gp": {"bestFid": gpB, "bestTick": gpK,
                   "sample": gp_s[-1]["url"] if gp_s else None,
                   "durKimg": gp_s[-1]["kimg"] if gp_s else gpK},
            "curveGN": gnCurve, "curveGP": gpCurve,
            "trainSamplesGN": gn_s, "trainSamplesGP": gp_s,
        })
        self._emit(run_id, type="done")

    def _train_command(self, cfg, data, outdir):
        script = "train.py"
        return [self.s.python_bin, script] + build_train_args(cfg, data, outdir)

    async def _tail_training(self, run_id, cls, outdir, total_kimg, proc):
        stats_pos, metric_pos = 0, 0
        seen = set()
        last_kimg = 0

        def on_stats(line):
            nonlocal last_kimg
            row = parse_stats_line(line)
            if not row or row["kimg"] is None:
                return
            last_kimg = row["kimg"]
            spk = row.get("sec_per_kimg") or 0
            eta = max(0.0, (total_kimg - row["kimg"]) * spk) if spk else None
            self._emit(run_id, type="tick", **{"class": cls},
                       tick=row.get("tick"), kimg=round(row["kimg"], 1), total=total_kimg, eta=eta,
                       augment=row.get("augment"), secPerTick=row.get("sec_per_tick"),
                       secPerKimg=row.get("sec_per_kimg"), totalSec=row.get("total_sec"),
                       gpumem=row.get("gpu_mem"), cpumem=row.get("cpu_mem"))

        def on_metric(line):
            m = parse_metric_line(line)
            if m and m["fid"] is not None:
                self._emit(run_id, type="fid", **{"class": cls},
                           x=m["kimg"] if m["kimg"] is not None else last_kimg, y=m["fid"])

        while True:
            sub = self._newest_subdir(outdir)
            if sub:
                stats_pos = self._read_new(sub / "stats.jsonl", stats_pos, on_stats)
                for mf in sorted(sub.glob("metric-*.jsonl")):
                    metric_pos = self._read_new(mf, metric_pos, on_metric)
                for png in sorted(sub.glob("fakes*.png")):
                    if png.name not in seen:
                        seen.add(png.name)
                        self._emit(run_id, type="sample", **{"class": cls},
                                   kimg=kimg_from_name(png.name),
                                   url=f"/api/runs/{run_id}/artifacts/{cls}/{sub.name}/{png.name}")
            if proc.returncode is not None:
                break
            await asyncio.sleep(self.s.poll_interval)

    # ---- REAL simple stages ---------------------------------------------- #
    async def _format(self, run_id, pos, neg, res):
        rd = self.s.runs_dir / run_id
        rd.mkdir(parents=True, exist_ok=True)
        if self.s.mock or self.s.mock_format:
            # Simulated formatting: show the progress steps but DON'T run
            # dataset_tool or write a multi-GB formatted copy. Instead link the
            # already-staged source archive as this run's dataset (zero disk), so
            # REAL training downstream still trains on real crops.
            await self._mock_simple(run_id, 3.0, steps=[
                "Validate archives & scan images",
                f"Resize crops -> {res}x{res} (bicubic)",
                "Build dataset.json + Gram labels",
                "Package training-ready dataset"])
            for cls, src in (("gp", pos), ("gn", neg)):
                if not src:
                    continue
                source = self.s.data_dir / src
                link = rd / f"dataset_{cls}.zip"
                if link.is_symlink() or link.exists():
                    link.unlink()
                if source.exists():
                    try:
                        link.symlink_to(source.resolve())
                    except OSError:
                        pass
            self._save_result(run_id, "format", {
                "res": res,
                "cropsPos": self._zip_image_count(rd / "dataset_gp.zip") or 6054,
                "cropsNeg": self._zip_image_count(rd / "dataset_gn.zip") or 13141,
            })
            return self._emit(run_id, type="done")
        for cls, src in (("gp", pos), ("gn", neg)):
            if not src:
                continue
            args = [self.s.python_bin, "dataset_tool.py",
                    "--source", str(self.s.data_dir / src),
                    "--dest", str(rd / f"dataset_{cls}.zip"),
                    "--width", str(res), "--height", str(res)]
            await self._simple_proc(run_id, args, self.s.stylegan_dir, est_sec=30,
                                    steps=[f"Formatting {cls} dataset"])
        self._save_result(run_id, "format", {
            "res": res,
            "cropsPos": self._zip_image_count(rd / "dataset_gp.zip"),
            "cropsNeg": self._zip_image_count(rd / "dataset_gn.zip"),
        })
        self._emit(run_id, type="progress", pct=1.0)
        self._emit(run_id, type="done")

    async def _generate(self, run_id, n):
        rd = self.s.runs_dir / run_id
        if self.s.mock:
            await self._mock_simple(run_id, 2.4, phase="Gram-positive")
            await self._mock_simple(run_id, 2.4, phase="Gram-negative")
            # Demo gallery references the front end's own static tiles.
            gallery = ([{"src": f"/figs/tiles/pos-{i}.png", "cls": "pos"} for i in range(16)]
                       + [{"src": f"/figs/tiles/neg-{i}.png", "cls": "neg"} for i in range(16)])
            self._save_result(run_id, "generate", {"gallery": gallery, "total": (int(n) or 0) * 2})
            return self._emit(run_id, type="done")
        if self._is_demo(run_id):
            await self._mock_simple(run_id, 2.4, phase="Gram-positive")
            await self._mock_simple(run_id, 2.4, phase="Gram-negative")
            gen = self._ref_results().get("generate")
            if gen is not None:
                self._save_result(run_id, "generate", gen)   # reveal the seeded real gallery
            return self._emit(run_id, type="done")
        for cls, phase in (("gp", "Gram-positive"), ("gn", "Gram-negative")):
            pkl = self._best_pkl(run_id, cls)
            args = [self.s.python_bin, "generate.py", "--network", str(pkl),
                    "--seeds", f"0-{max(0, int(n) - 1)}",
                    "--outdir", str(rd / "gen" / cls)]
            await self._simple_proc(run_id, args, self.s.stylegan_dir, est_sec=40, phase=phase)
        self._save_result(run_id, "generate", self._collect_gallery(run_id, int(n)))
        self._emit(run_id, type="progress", pct=1.0)
        self._emit(run_id, type="done")

    def _collect_gallery(self, run_id, n):
        gallery, total = [], 0
        for cls, kind in (("gp", "pos"), ("gn", "neg")):
            d = self.s.runs_dir / run_id / "gen" / cls
            if d.exists():
                pngs = sorted(d.glob("*.png"))
                total += len(pngs)
                for p in pngs[:32]:   # cap what the UI renders
                    gallery.append({"src": f"/api/runs/{run_id}/artifacts/gen/{cls}/{p.name}", "cls": kind})
        return {"gallery": gallery, "total": total or (n * 2)}

    async def _fidelity(self, run_id, num=5000, data_override=None):
        """Fidelity = per-checkpoint FID sweep: score every snapshot against the
        real crops, pick the lowest-FID one per class, and point the generator at
        it (so the later Generate stage uses the best checkpoint)."""
        if self.s.mock:
            collected = await self._mock_fid_sweep(run_id)
        elif self._is_demo(run_id):
            collected = await self._demo_fid_sweep(run_id)
        else:
            collected = await self._real_fid_sweep(run_id, num, data_override or {})
        self._persist_sweep(run_id, collected)   # FID curve + best per class (results.train)
        gn = (collected.get("gn") or {}).get("best") or {}
        gp = (collected.get("gp") or {}).get("best") or {}
        self._save_result(run_id, "fidelity", {
            "gn": gn.get("fid"), "gp": gp.get("fid"),
            "gnTick": gn.get("kimg"), "gpTick": gp.get("kimg"),
        })
        self._emit(run_id, type="done")

    @staticmethod
    def _clean_path(v):
        return str(v or "").strip().strip('"').strip("'").strip()

    def _feas_dataset(self, override, key, default, label):
        """Resolve a feasibility dataset dir from the UI override (falling back
        to the server default) and verify it has the two class subfolders."""
        v = self._clean_path(override.get(key))
        root = Path(v) if v else default
        if not root.is_dir() or not any((root / c).is_dir() for c in ("gram_negative", "gram_positive")):
            raise RuntimeError(
                f"{label} dataset not found at {root} — point it at a folder containing "
                f"gram_negative/ and gram_positive/ subfolders of crops")
        return root

    async def _feasibility(self, run_id, data_override=None):
        data_override = data_override or {}
        if self.s.mock:
            labels = MOCK_F1["archs"]
            steps = [f"Training {i + 1} / 20 · {labels[i // 4]} · Scenario {['I', 'II', 'III', 'IV'][i % 4]}"
                     for i in range(20)]
            await self._mock_simple(run_id, 5.4, steps=steps)
            self._save_result(run_id, "feasibility", MOCK_F1)
            return self._emit(run_id, type="done")
        if self._is_demo(run_id):
            ref = self._ref_results()
            feas = ref.get("feasibility") or MOCK_F1
            archs = feas.get("archs") or MOCK_F1["archs"]
            steps = [f"Training {i + 1} / 20 · {archs[i // 4]} · Scenario {['I', 'II', 'III', 'IV'][i % 4]}"
                     for i in range(20)]
            await self._mock_simple(run_id, 5.4, steps=steps)
            self._save_result(run_id, "feasibility", feas)   # reveal the seeded real F1 table
            gen = ref.get("generate")
            if gen is not None:
                self._save_result(run_id, "generate", gen)   # synth gallery (shown in Results)
            return self._emit(run_id, type="done")
        # Real path. Generate is folded into Feasibility: first synthesize crops
        # from the best checkpoints (selected by Fidelity), then run the F1 study.
        rd = self.s.runs_dir / run_id
        n_synth = 2000
        for cls, phase in (("gp", "Gram-positive"), ("gn", "Gram-negative")):
            out = rd / "gen" / cls
            have = len(list(out.glob("*.png"))) if out.exists() else 0
            if have >= n_synth:
                continue
            pkl = self._best_pkl(run_id, cls)
            if not Path(pkl).is_file():
                raise RuntimeError(f"{cls}: no generator checkpoint to synthesize from — run Train + Fidelity first")
            gargs = [self.s.python_bin, "generate.py", "--network", str(pkl),
                     "--seeds", f"0-{n_synth - 1}", "--outdir", str(out)]
            await self._simple_proc(run_id, gargs, self.s.stylegan_dir, est_sec=120,
                                    steps=[f"Synthesizing {phase} crops ({n_synth})…"])

        # Then the real 5-CNN x 4-scenario study via scripts/feasibility.py. It
        # streams JSON progress/result lines on stdout, which we forward verbatim.
        real = self._feas_dataset(data_override, "real", self.s.data_dir / "real", "real training")
        test = self._feas_dataset(data_override, "test", self.s.data_dir / "test", "real test")
        script = Path(__file__).parent / "scripts" / "feasibility.py"
        args = [self.s.python_bin, str(script),
                "--real", str(real),
                "--synth", str(rd / "gen"),
                "--test", str(test),
                "--out", str(rd / "feasibility.json")]
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(Path(__file__).parent),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True)
        self.procs[run_id] = proc
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue  # non-JSON log line from the script
            if obj.get("type") == "progress":
                self._emit(run_id, **obj)
            elif obj.get("type") == "result":
                self._save_result(run_id, obj["key"], obj["value"])
            elif obj.get("type") == "error":
                raise RuntimeError(obj.get("message", "feasibility failed"))
        rc = await proc.wait()
        if rc not in (0, None):
            raise RuntimeError(f"feasibility.py exited with {rc}")
        self._emit(run_id, type="done")

    def _load_results(self, run_id):
        f = self.s.runs_dir / run_id / "results.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except (ValueError, OSError):
                return {}
        return {}

    # ---- DEMO replay (seeded presentation runs) --------------------------- #
    def _is_demo(self, run_id):
        """Seeded presentation runs carry a `.demo` marker. Their stages replay
        the pre-seeded REAL results with a quick animation instead of touching
        the GPU — so a presentation never hits a missing-checkpoint error, while
        any run the user creates themselves still executes for real."""
        return (self.s.runs_dir / run_id / ".demo").exists()

    def _ref_results(self):
        """Canonical full real results (the completed run) used to fill in a demo
        run's stage when it is replayed."""
        return self._load_results("r-imported-378")

    async def _demo_train(self, run_id, cfg):
        """Animate training ticks for both generators, then reveal the run's
        pre-seeded real train result (best FID, curves, checkpoint samples)."""
        total = int(cfg.get("ticks") or 100)
        step = max(1, total // 30)
        for cls in ("gn", "gp"):
            t, tick = 0, 0
            while t < total:
                t = min(total, t + step)
                tick += 1
                self._emit(run_id, type="tick", **{"class": cls}, tick=tick, kimg=round(t, 1),
                           total=total, eta=(total - t) * 0.05,
                           augment=round(0.6 * min(1.0, t / total), 3),
                           secPerTick=10.0, secPerKimg=2.5, totalSec=tick * 10,
                           gpumem=4.0, cpumem=2.5)
                await asyncio.sleep(0.05)
            self._emit(run_id, type="class_done", **{"class": cls})
        ref = self._ref_results().get("train")
        if ref is not None:
            # A real metrics=none train measures no FID — strip best/curve so the
            # best FID only appears AFTER the fidelity test runs.
            tr = json.loads(json.dumps(ref))
            for cls in ("gn", "gp"):
                if cls in tr:
                    tr[cls] = {**tr[cls], "bestFid": None, "bestTick": None}
            for k in ("curveGN", "curveGP", "fidSamplesGN", "fidSamplesGP"):
                tr.pop(k, None)
            self._save_result(run_id, "train", tr)
        self._emit(run_id, type="done")

    async def _demo_fid_sweep(self, run_id):
        """Stream the run's pre-seeded real FID-vs-checkpoint curve, then return
        the real best per class — a fast stand-in for the GPU sweep."""
        ref = self._ref_results()
        train = ref.get("train", {})
        fid = ref.get("fidelity", {})
        collected = {}
        for ci, cls in enumerate(("gn", "gp")):
            curve = train.get("curveGN" if cls == "gn" else "curveGP", []) or []
            n = len(curve) or 1
            label = "Gram-negative" if cls == "gn" else "Gram-positive"
            for i, pt in enumerate(curve):
                self._emit(run_id, type="progress", pct=(ci * n + i + 1) / (2 * n),
                           step=f"FID {label} · checkpoint {i + 1}/{n} · kimg {pt['x']}")
                self._emit(run_id, type="fid", **{"class": cls}, x=pt["x"], y=pt["y"])
                await asyncio.sleep(0.03)
            collected[cls] = {"curve": curve, "best": {"fid": fid.get(cls), "kimg": fid.get(f"{cls}Tick")}}
        return collected

    # ---- checkpoint FID sweep (find the best snapshot) -------------------- #
    def _ckpt_dir(self, run_id, cls):
        base = self.s.runs_dir / run_id / cls
        src = base / "SRC_DIR.txt"            # imported runs: the original dir with all snapshots
        if src.is_file():
            p = Path(src.read_text().strip())
            if p.is_dir():
                return p
        return self._newest_subdir(base)      # trained-here runs

    def _real_data_for(self, run_id, cls):
        rd = self.s.runs_dir / run_id
        for c in (rd / f"dataset_{cls}.zip", self.s.data_dir / f"dataset_{cls}.zip"):
            if c.exists():
                return c
        return self.s.data_dir / "real" / ("gram_negative" if cls == "gn" else "gram_positive")

    async def _real_fid_sweep(self, run_id, num, data_override):
        rd = self.s.runs_dir / run_id
        script = Path(__file__).parent / "scripts" / "sweep_fid.py"
        collected = {}
        for cls in ("gn", "gp"):
            ckptdir = self._ckpt_dir(run_id, cls)
            override = (data_override.get(cls) or "").strip().strip('"').strip("'").strip()
            data = Path(override) if override else self._real_data_for(run_id, cls)
            if not ckptdir or not Path(ckptdir).is_dir():
                raise RuntimeError(f"{cls}: no checkpoint directory found to evaluate")
            if not Path(data).exists():
                raise RuntimeError(f"{cls}: real dataset not found: {data} — give the {('gram-negative' if cls=='gn' else 'gram-positive')} crops at the model's resolution")
            args = [self.s.python_bin, str(script), "--sgdir", str(self.s.stylegan_dir),
                    "--ckpt-dir", str(ckptdir), "--data", str(data),
                    "--out", str(rd / f"fidsweep_{cls}.json"), "--cls", cls]
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=str(self.s.stylegan_dir),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True)
            self.procs[run_id] = proc
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                t = obj.get("type")
                if t in ("progress", "fid"):
                    self._emit(run_id, **obj)
                elif t == "sample":
                    self._emit(run_id, type="sample", **{"class": obj["class"]},
                               kimg=obj["kimg"], url=f"/api/runs/{run_id}/artifacts/{obj['rel']}")
                elif t == "result":
                    collected[cls] = obj["value"]
                elif t == "error":
                    raise RuntimeError(obj.get("message", "fid sweep failed"))
            rc = await proc.wait()
            if rc not in (0, None):
                raise RuntimeError(f"sweep_fid ({cls}) exited with {rc}")
            best = collected.get(cls, {}).get("best")
            if best and best.get("pkl"):
                (rd / cls / "PKL_PATH.txt").write_text(best["pkl"])  # point generator at the best
        return collected

    def _persist_sweep(self, run_id, collected):
        existing = self._load_results(run_id).get("train", {})
        train = dict(existing)
        for cls in ("gn", "gp"):
            v = collected.get(cls)
            if not v:
                continue
            if v.get("best"):
                train[cls] = {**existing.get(cls, {}), "bestFid": v["best"]["fid"], "bestTick": v["best"]["kimg"]}
            if v.get("curve"):
                train["curveGN" if cls == "gn" else "curveGP"] = v["curve"]
            if v.get("samples"):
                train["fidSamplesGN" if cls == "gn" else "fidSamplesGP"] = [
                    {"kimg": s["kimg"], "url": f"/api/runs/{run_id}/artifacts/{s['rel']}"} for s in v["samples"]]
        self._save_result(run_id, "train", train)

    async def _mock_fid_sweep(self, run_id):
        collected = {}
        ticks = list(range(200, 5001, 200))
        for ci, cls in enumerate(("gn", "gp")):
            base = 7.35 if cls == "gn" else 7.74
            curve, best = [], None
            for i, k in enumerate(ticks):
                self._emit(run_id, type="progress",
                           pct=(ci * len(ticks) + i) / (2 * len(ticks)),
                           step=f"FID {cls} · checkpoint {i + 1}/{len(ticks)} · kimg {k}")
                fid = round(base + 80 * math.exp(-k / 500) + 0.4 * math.sin(k / 300), 2)
                curve.append({"x": k, "y": fid})
                self._emit(run_id, type="fid", **{"class": cls}, x=k, y=fid)
                if best is None or fid < best["fid"]:
                    best = {"kimg": k, "fid": fid}
                await asyncio.sleep(0.04)
            collected[cls] = {"curve": curve, "best": best}
        return collected

    def _best_pkl(self, run_id, cls):
        # An imported run records its checkpoint path here (avoids copying GBs).
        override = self.s.runs_dir / run_id / cls / "PKL_PATH.txt"
        if override.is_file():
            p = Path(override.read_text().strip())
            if p.is_file():
                return p
        sub = self._newest_subdir(self.s.runs_dir / run_id / cls)
        if sub:
            pkls = sorted(sub.glob("network-snapshot-*.pkl"))
            if pkls:
                return pkls[-1]
        return self.s.runs_dir / run_id / cls / "best.pkl"

    # ---- MOCK (no GPU) ---------------------------------------------------- #
    async def _mock_simple(self, run_id, dur, steps=None, phase=None, done=False):
        n = 40
        for i in range(1, n + 1):
            pct = i / n
            ev = {"type": "progress", "pct": pct}
            if phase:
                ev["phase"] = phase
            if steps:
                ev["step"] = steps[min(len(steps) - 1, int(pct * len(steps)))]
            self._emit(run_id, **ev)
            await asyncio.sleep(dur / n)
        if done:
            self._emit(run_id, type="done")

    async def _mock_train(self, run_id, cfg):
        import random
        total = int(cfg.get("ticks") or 1000)
        step = max(1, total // 40)
        target = float(cfg.get("target") or 0.6)
        for cls in ("gn", "gp"):
            t, tick = 0, 0
            while t < total:
                t = min(total, t + step); tick += 1
                # ADA augment probability ramps toward the target as training proceeds
                aug = max(0.0, round(target * min(1.0, (t / total) * 1.3) + random.uniform(-0.02, 0.02), 3))
                self._emit(run_id, type="tick", **{"class": cls}, tick=tick, kimg=round(t, 1), total=total,
                           eta=(total - t) * 0.05, augment=aug,
                           secPerTick=round(random.uniform(8, 12), 1), secPerKimg=round(random.uniform(2, 3), 2),
                           totalSec=tick * 10, gpumem=round(random.uniform(3, 4), 2), cpumem=round(random.uniform(2, 3), 2))
                await asyncio.sleep(0.1)
            self._emit(run_id, type="class_done", **{"class": cls})
        # Training no longer reports FID — the Fidelity step measures it per checkpoint.
        self._emit(run_id, type="done")
