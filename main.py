"""GramSynth orchestration API.

Run with:  uvicorn main:app --reload --port 8000   (from the server/ directory)
"""
import asyncio
import json

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from config import settings
from db import init_db, SessionLocal, Run, upsert_run
from schemas import RunCreate, FormatStart, TrainStart, GenerateStart, FidelityStart, ImportModels
from runner import JobManager
from importer import do_import

app = FastAPI(title="GramSynth Pipeline API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = JobManager(settings)


@app.on_event("startup")
def _startup():
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
#  Run CRUD
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"ok": True, "mock": settings.mock}


@app.get("/api/gpu")
def gpu():
    """Detect the GPU shown in the run header. Tries `nvidia-smi` first (no
    Python deps needed), then torch, then falls back to CPU."""
    import os
    import subprocess

    name = os.environ.get("GPU_NAME")
    if not name:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8,
            )
            line = (out.stdout or "").strip().splitlines()[0].strip()
            if line:
                parts = [p.strip() for p in line.split(",")]
                name = parts[0] + (f" · {parts[1]}" if len(parts) > 1 else "")
        except Exception:
            name = None
    if not name:
        try:
            import torch
            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no CUDA)"
        except Exception:
            name = "CPU (no CUDA)"
    return {"name": name}


@app.get("/api/runs")
def list_runs(db=Depends(get_db)):
    return [r.to_dict() for r in db.query(Run).order_by(Run.updated_at.desc()).all()]


@app.post("/api/runs")
def create_run(body: RunCreate, db=Depends(get_db)):
    run = upsert_run(db, body.id, name=body.name, dataset=body.dataset, config=body.config, pipe=body.pipe)
    return run.to_dict()


@app.post("/api/import")
def import_models(body: ImportModels):
    """Import two already-trained run directories as a ready-to-generate run."""
    rid = body.id or ("r-imported-" + str(abs(hash(body.gn + body.gp)) % 10000))
    try:
        return do_import(body.gn, body.gp, rid, body.name)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, db=Depends(get_db)):
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run.to_dict()


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str, db=Depends(get_db)):
    run = db.get(Run, run_id)
    if run:
        db.delete(run)
        db.commit()
    return {"ok": True}


# --------------------------------------------------------------------------- #
#  Stage actions
# --------------------------------------------------------------------------- #
def _ensure(db, run_id: str):
    if not db.get(Run, run_id):
        upsert_run(db, run_id)


# NOTE: these are `async def` on purpose — the job manager schedules work with
# asyncio.create_task(), which needs the running event loop. A sync endpoint
# would execute in a worker thread with no loop and raise at runtime.
@app.post("/api/runs/{run_id}/format/start")
async def start_format(run_id: str, body: FormatStart, db=Depends(get_db)):
    _ensure(db, run_id)
    manager.start_format(run_id, body.pos, body.neg, body.res)
    return {"ok": True}


@app.post("/api/runs/{run_id}/train/start")
async def start_train(run_id: str, body: TrainStart, db=Depends(get_db)):
    _ensure(db, run_id)
    if body.config:
        upsert_run(db, run_id, config=body.config)
    manager.start_train(run_id, body.config or {})
    return {"ok": True}


@app.post("/api/runs/{run_id}/generate/start")
async def start_generate(run_id: str, body: GenerateStart, db=Depends(get_db)):
    _ensure(db, run_id)
    manager.start_generate(run_id, body.n)
    return {"ok": True}


@app.post("/api/runs/{run_id}/fidelity/start")
async def start_fidelity(run_id: str, body: FidelityStart, db=Depends(get_db)):
    _ensure(db, run_id)
    manager.start_fidelity(run_id, body.num, {"gn": body.gn, "gp": body.gp})
    return {"ok": True}


@app.post("/api/runs/{run_id}/feasibility/start")
async def start_feasibility(run_id: str, db=Depends(get_db)):
    _ensure(db, run_id)
    manager.start_feasibility(run_id)
    return {"ok": True}


@app.post("/api/runs/{run_id}/cancel")
async def cancel(run_id: str):
    await manager.cancel(run_id)
    return {"ok": True}


# --------------------------------------------------------------------------- #
#  Real-time event stream (Server-Sent Events)
# --------------------------------------------------------------------------- #
@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, request: Request):
    broker = manager.broker(run_id)
    q = await broker.subscribe()

    async def gen():
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # keep the connection alive through proxies
        finally:
            broker.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
#  Artifacts (sample grids, etc.) — served from inside the run directory
# --------------------------------------------------------------------------- #
@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str):
    """Whether a job is currently running for this run, and which kind — lets the
    UI re-attach to a live job after navigating away and back."""
    kind = manager.running_kind(run_id)
    return {"running": kind is not None, "kind": kind}


@app.get("/api/runs/{run_id}/results")
def results(run_id: str):
    """The accumulated real results for a run (best FID, F1 table, gallery, …).
    Lets the front end show real numbers when reopening a completed run."""
    f = settings.runs_dir / run_id / "results.json"
    if f.is_file():
        try:
            return json.loads(f.read_text())
        except (ValueError, OSError):
            return {}
    return {}


@app.get("/api/runs/{run_id}/artifacts/{path:path}")
def artifact(run_id: str, path: str):
    base = (settings.runs_dir / run_id).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(target)
