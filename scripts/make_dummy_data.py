"""Generate a tiny dummy dataset so the pipeline can be exercised locally
(Format/Train/Fidelity/Feasibility) without the real clinical archives.
These are random images — only for verifying the workflow runs end to end."""
import sys
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["gram_positive", "gram_negative"]


def rnd_img(size, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 200, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def make_zip(zip_path, n, size, seed0):
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(n):
            p = zip_path.parent / f"_tmp_{i}.png"
            rnd_img(size, seed0 + i).save(p)
            z.write(p, f"img{i:04d}.png")
            p.unlink()


def make_folder(folder, n, size, seed0):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rnd_img(size, seed0 + i).save(folder / f"img{i:04d}.png")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    # Format inputs (zips of raw crops)
    make_zip(DATA / "gram_positive.zip", 40, 256, 1000)
    make_zip(DATA / "gram_negative.zip", 40, 256, 2000)
    # Feasibility data (real train + isolated test split), ImageFolder layout
    for ci, cls in enumerate(CLASSES):
        make_folder(DATA / "real" / cls, 30, 128, 3000 + ci * 100)
        make_folder(DATA / "test" / cls, 12, 128, 5000 + ci * 100)
    print("dummy data written to", DATA)


if __name__ == "__main__":
    main()
