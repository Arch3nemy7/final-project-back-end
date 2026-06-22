#!/usr/bin/env python
"""Feasibility test: do the synthetic crops actually help a classifier?

Trains 5 CNN architectures under 4 fixed-total real/synthetic compositions and
reports macro-F1 on a held-out REAL test split. This is the project-specific
"Test 2" from the thesis — it is not part of the NVlabs repo.

It streams machine-readable progress to stdout (one JSON object per line) so the
orchestration server can forward it to the UI in real time:

    {"type": "progress", "pct": 0.15, "step": "Training 3 / 20 · VGG-16 · Scenario III"}
    {"type": "result", "key": "feasibility", "value": {... F1 table ...}}

Expected data layout (ImageFolder style, two classes each):
    --real  DIR/{gram_positive,gram_negative}/*.png   # real training crops
    --synth DIR/{gram_positive,gram_negative}/*.png   # GAN-generated crops (Generate stage)
    --test  DIR/{gram_positive,gram_negative}/*.png   # isolated real test split

Scenarios (real% / synthetic%) at a fixed total per class:
    I 100/0 (baseline) · II 75/25 · III 50/50 · IV 25/75
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import f1_score

CLASSES = ["gram_negative", "gram_positive"]          # label 0, 1
ARCHS = ["ResNet-50", "DenseNet-121", "VGG-16", "MobileNetV3", "InceptionV3"]
SCENARIOS = [(1.0, 0.0), (0.75, 0.25), (0.50, 0.50), (0.25, 0.75)]
SCENARIO_LABELS = ["I", "II", "III", "IV"]


def emit(obj):
    """Print one JSON line and flush so the server sees it immediately."""
    print(json.dumps(obj), flush=True)


def list_images(root: Path, cls: str):
    d = root / cls
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


class CropDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


def build_model(arch: str):
    """Pretrained backbone with a fresh 2-class head."""
    if arch == "ResNet-50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "DenseNet-121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif arch == "VGG-16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
    elif arch == "MobileNetV3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif arch == "InceptionV3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, 2)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


def compose(real_root, synth_root, real_frac, synth_frac, per_class, rng):
    """Build a fixed-total training list mixing real + synthetic per class."""
    items = []
    for label, cls in enumerate(CLASSES):
        real = list_images(real_root, cls)
        synth = list_images(synth_root, cls) if synth_root else []
        rng.shuffle(real)
        rng.shuffle(synth)
        n_real = int(round(per_class * real_frac))
        n_synth = int(round(per_class * synth_frac))
        chosen = real[:n_real] + synth[:n_synth]
        items += [(p, label) for p in chosen]
    rng.shuffle(items)
    return items


def evaluate(model, loader, device, arch):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            if arch == "InceptionV3" and isinstance(out, tuple):
                out = out[0]
            y_pred += out.argmax(1).cpu().tolist()
            y_true += y.tolist()
    if not y_true:
        return 0.0
    return f1_score(y_true, y_pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", default=None)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    real_root, synth_root, test_root = Path(args.real), (Path(args.synth) if args.synth else None), Path(args.test)
    size = 299 if False else 224  # default size; Inception is resized below

    test_items = [(p, lbl) for lbl, cls in enumerate(CLASSES) for p in list_images(test_root, cls)]

    rows = []
    losses = []                 # losses[arch][scenario] = [mean train loss per epoch]
    total_jobs = len(ARCHS) * len(SCENARIOS)
    job = 0
    for ai, arch in enumerate(ARCHS):
        in_size = 299 if arch == "InceptionV3" else 224
        train_tf = transforms.Compose([
            transforms.Resize((in_size, in_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_loader = DataLoader(CropDataset(test_items, eval_tf), batch_size=args.batch)
        row = []
        loss_row = []
        for si, (rf, sf) in enumerate(SCENARIOS):
            job += 1
            emit({"type": "progress", "pct": (job - 1) / total_jobs,
                  "step": f"Training {job} / {total_jobs} · {arch} · Scenario {SCENARIO_LABELS[si]}"})
            items = compose(real_root, synth_root, rf, sf, args.per_class, rng)
            loader = DataLoader(CropDataset(items, train_tf), batch_size=args.batch, shuffle=True, drop_last=True)

            model = build_model(arch).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = nn.CrossEntropyLoss()
            model.train()
            epoch_losses = []
            for _ in range(args.epochs):
                run_loss, n_batches = 0.0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    out = model(x)
                    if arch == "InceptionV3" and isinstance(out, tuple):
                        out, aux = out
                        loss = loss_fn(out, y) + 0.4 * loss_fn(aux, y)
                    else:
                        loss = loss_fn(out, y)
                    loss.backward()
                    opt.step()
                    run_loss += float(loss.detach()); n_batches += 1
                epoch_losses.append(round(run_loss / n_batches, 4) if n_batches else None)
            f1 = round(evaluate(model, test_loader, device, arch), 4)
            row.append(f1)
            loss_row.append(epoch_losses)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        losses.append(loss_row)

    # top classifier = best baseline (scenario I) F1
    baselines = [r[0] for r in rows]
    top_i = max(range(len(ARCHS)), key=lambda i: baselines[i])
    value = {"archs": ARCHS, "rows": rows, "losses": losses,
             "top": ARCHS[top_i], "topF1": baselines[top_i]}

    Path(args.out).write_text(json.dumps(value, indent=2))
    emit({"type": "progress", "pct": 1.0})
    emit({"type": "result", "key": "feasibility", "value": value})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface failures as a parseable line
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)
