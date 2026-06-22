#!/usr/bin/env python
"""CLI to import already-trained StyleGAN2-ADA run directories into the backend.

(The same logic is exposed in the UI via POST /api/import.)

Usage (from the server/ directory; defaults point at the user's dirs):
    python scripts/import_models.py
    python scripts/import_models.py --gn <dir> --gp <dir> --name "PLA combined"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from importer import do_import  # noqa: E402

DEFAULT_VAST = Path(__file__).resolve().parents[3]  # ...\Proyek Akhir\vast
DEFAULT_GN = DEFAULT_VAST / "00001-gram_negative-auto1-resumeffhq256"
DEFAULT_GP = DEFAULT_VAST / "00001-gram_positive-auto1-resumeffhq256"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gn", default=str(DEFAULT_GN))
    ap.add_argument("--gp", default=str(DEFAULT_GP))
    ap.add_argument("--id", default="r-imported")
    ap.add_argument("--name", default="Imported · PLA generators")
    args = ap.parse_args()

    try:
        run = do_import(args.gn, args.gp, args.id, args.name)
    except FileNotFoundError as e:
        print("ERROR:", e)
        sys.exit(1)

    t = run["pipe"]["results"]["train"]
    print(f"Imported run '{run['id']}' ({run['name']})")
    print(f"  gn: checkpoint @ {t['gn']['bestTick']} kimg")
    print(f"  gp: checkpoint @ {t['gp']['bestTick']} kimg")
    print("Open the front end (live mode) — it will adopt this run at the Generate stage.")


if __name__ == "__main__":
    main()
