# GramSynth — Backend API (orchestration + real-time telemetry)

A small **FastAPI** server that sits between the React front end and the
**StyleGAN2-ADA** training code. It launches each pipeline stage as a job and
streams **real-time** progress to the browser over **Server-Sent Events (SSE)**.

```
Browser (React)  ──HTTP──▶  FastAPI (this server)  ──subprocess──▶  train.py / generate.py / …
      ▲                          │                                        │
      └────────SSE───────────────┴────────tails the run directory◀────────┘
                          (stats.jsonl · metric-*.jsonl · fakes*.png)
```

---

## Do I need a database? (No PostgreSQL required)

**No.** This server stores only lightweight *run metadata* (name, dataset,
config, current stage). The heavy data — checkpoints, FID logs, sample images —
already lives on disk in each run's folder, which is the real source of truth.

So by default it uses **SQLite**, a single file (`gramsynth.db`) created
automatically. Nothing to install.

You mentioned you already have **PostgreSQL** — you *can* use it, but you don't
need to. To switch, set one environment variable (and install the driver):

```bash
pip install "psycopg[binary]"
# in server/.env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/gramsynth
```

The front end itself uses the browser's localStorage, so it needs no database
at all.

---

## Install

From this `server/` folder, in the Python environment that has PyTorch/CUDA
(the same one that runs StyleGAN2-ADA):

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
```

Copy the env template and edit if needed:

```bash
cp .env.example .env
```

---

## Run

### A) Demo without a GPU (try the real-time UI today)

```bash
# Windows PowerShell:  $env:MOCK=1; uvicorn main:app --port 8000
# Linux/macOS:         MOCK=1 uvicorn main:app --port 8000
```

`MOCK=1` simulates the whole pipeline in-process (no GPU, no StyleGAN scripts),
emitting the same events as the real thing — perfect for verifying the
front-end wiring end to end.

### B) Real training (on the GPU box)

```bash
uvicorn main:app --port 8000
```

Make sure these point at the right places (defaults in `config.py`, override in
`.env`):

- `STYLEGAN_DIR` → the `stylegan2-ada-pytorch` checkout (default `../stylegan2-ada-pytorch`)
- `DATA_DIR` → where the cropped class `.zip` archives are staged
- `PYTHON_BIN` → the interpreter with PyTorch/CUDA
- `RUNS_DIR` → where per-run output is written

Then connect the front end by setting `VITE_API_BASE=http://localhost:8000`
(see the [web app README](../webapp/README.md)).

---

## Running next to a remote GPU (e.g. vast.ai)

Run this server **on the GPU instance** (it needs the StyleGAN code + CUDA).
From your laptop, forward the port over SSH:

```bash
ssh -L 8000:localhost:8000 user@your-vast-instance
```

Now `http://localhost:8000` on your laptop reaches the remote API, so the front
end's `VITE_API_BASE=http://localhost:8000` works unchanged.

---

## How the real-time data works

StyleGAN2-ADA already writes telemetry to each run directory while it trains.
The server tails these files and turns them into events:

| File it watches            | Becomes the event       | Shown in the UI as            |
| -------------------------- | ----------------------- | ----------------------------- |
| `stats.jsonl` (per tick)   | `tick` (kimg, ETA)      | tick counter, progress, ETA   |
| `metric-fid50k_full.jsonl` | `fid` (kimg, value)     | the live FID convergence chart|
| `fakes*.png` (per snapshot) | `sample`               | checkpoint sample previews    |

**Granularity:** FID and samples update *per snapshot* (the `--snap` interval),
not every iteration — that's simply how StyleGAN2-ADA reports. Tick progress is
more frequent.

---

## API endpoints

| Method & path                            | Purpose                              |
| ---------------------------------------- | ------------------------------------ |
| `GET  /api/health`                       | liveness + whether mock mode is on   |
| `GET  /api/runs`                         | list runs                            |
| `POST /api/runs`                         | create/upsert a run                  |
| `GET/DELETE /api/runs/{id}`              | fetch / delete a run                 |
| `POST /api/runs/{id}/format/start`       | start data formatting                |
| `POST /api/runs/{id}/train/start`        | start training (GN then GP)          |
| `POST /api/runs/{id}/generate/start`     | start synthesis                      |
| `POST /api/runs/{id}/fidelity/start`     | start FID test                       |
| `POST /api/runs/{id}/feasibility/start`  | start 5-CNN feasibility test         |
| `POST /api/runs/{id}/cancel`             | cancel the running job               |
| `GET  /api/runs/{id}/events`             | **SSE** real-time event stream       |
| `GET  /api/runs/{id}/results`            | accumulated real results (FID, F1, gallery) |
| `GET  /api/runs/{id}/artifacts/{path}`   | serve a file from the run directory  |

Interactive docs are available at `http://localhost:8000/docs` once running.

---

## Every stage now produces real results

Each stage computes a result that is **persisted** to `runs/<id>/results.json`,
**streamed** to the UI as a `result` event, and re-served by `GET .../results`.
The front end shows these real numbers instead of the demo constants.

- **Format** → real crop counts (read from the produced dataset zips) + resolution.
- **Train** → best FID + tick per generator and the full FID curve (from the
  metric logs); samples stream as `fakes*.png` grids.
- **Generate** → the real generated PNGs (served as artifacts) fill the gallery.
- **Fidelity** → each generator's measured best FID (Scenario A).
- **Feasibility** → the real macro-F1 table from `scripts/feasibility.py`
  (ResNet-50 / DenseNet-121 / VGG-16 / MobileNetV3 / InceptionV3 × 4 scenarios),
  trained on real + synthetic and evaluated on a held-out real test split.

### Data you must stage (under `DATA_DIR`)

```
data/
├─ gram_positive.zip   gram_negative.zip   # cropped class archives (Format input)
├─ real/{gram_positive,gram_negative}/*.png  # real training crops (Feasibility)
└─ test/{gram_positive,gram_negative}/*.png  # isolated real test split (Feasibility)
```

The synthetic crops for feasibility come from the Generate stage automatically
(`runs/<id>/gen/{gp,gn}`). Feasibility needs `torch`, `torchvision`, and
`scikit-learn` in the `PYTHON_BIN` environment. Tune epochs/per-class/batch via
the args in `JobManager._feasibility` (`runner.py`).

---

## Tests

The telemetry parsers are pure functions with no dependencies:

```bash
python tests/test_parsers.py
```

---

## Files

```
server/
├─ main.py          # FastAPI app: routes + SSE + artifact serving
├─ runner.py        # job manager, run-dir tailing, parsers (stdlib only)
├─ db.py            # SQLAlchemy model + session (SQLite default)
├─ schemas.py       # request/response models
├─ config.py        # settings (env-overridable)
├─ requirements.txt
├─ .env.example
├─ scripts/feasibility.py   # real 5-CNN × 4-scenario macro-F1 study
└─ tests/test_parsers.py
```
