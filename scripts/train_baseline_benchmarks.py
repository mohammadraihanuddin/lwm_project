#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from task1.train_mcs_models import load_all_samples


CLASS_RE = re.compile(r"^CLASS_(\d+)$", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpectrogramDataset(Dataset):
    def __init__(self, samples: Sequence[np.ndarray], labels: Sequence[int]) -> None:
        self.samples = [np.asarray(x, dtype=np.float32) for x in samples]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.samples[idx]
        if x.ndim != 2:
            x = np.squeeze(x)
        # Per-sample normalization for scratch baselines.
        x = (x - x.mean()) / max(x.std(), 1e-6)
        x = torch.from_numpy(x).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    # True scratch training: all torchvision backbones use weights=None.
    if model_name == "resnet18":
        m = models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        first = m.features[0][0]
        m.features[0][0] = nn.Conv2d(1, first.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    if model_name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    if model_name == "cnn":
        return SmallCNN(num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def discover_samples(city_dir: Path) -> Dict[int, List[np.ndarray]]:
    by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    for class_dir in city_dir.rglob("*"):
        if not class_dir.is_dir():
            continue
        m = CLASS_RE.match(class_dir.name)
        if not m:
            continue
        label = int(m.group(1))
        for sample_file in class_dir.glob("*"):
            if not sample_file.is_file():
                continue
            try:
                arr = load_all_samples(sample_file)
                if arr.ndim == 2:
                    by_class[label].append(arr)
                elif arr.ndim >= 3:
                    for i in range(arr.shape[0]):
                        by_class[label].append(arr[i])
            except Exception:
                continue
    return by_class


def split_per_class(
    by_class: Dict[int, List[np.ndarray]],
    shots: int,
    val_per_class: int,
    test_per_class: int,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    x_tr: List[np.ndarray] = []
    y_tr: List[int] = []
    x_val: List[np.ndarray] = []
    y_val: List[int] = []
    x_te: List[np.ndarray] = []
    y_te: List[int] = []
    for label, samples in sorted(by_class.items()):
        pool = samples.copy()
        random.shuffle(pool)
        tr = pool[:shots]
        rem = pool[shots:]
        va = rem[:val_per_class]
        te = rem[val_per_class : val_per_class + test_per_class]
        x_tr.extend(tr)
        y_tr.extend([label] * len(tr))
        x_val.extend(va)
        y_val.extend([label] * len(va))
        x_te.extend(te)
        y_te.extend([label] * len(te))
    return x_tr, y_tr, x_val, y_val, x_te, y_te


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.tolist())
    if not y_true:
        return 0.0, 0.0
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return acc, f1


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> nn.Module:
    model.to(device)
    if len(train_loader.dataset) == 0:
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        val_acc, _ = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train conventional baseline benchmarks from scratch.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--city", type=str, required=True)
    p.add_argument("--shots", nargs="+", type=int, default=[0, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--models", nargs="+", default=["resnet18", "efficientnet_b0", "mobilenet_v3_small", "cnn"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-per-class", type=int, default=100)
    p.add_argument("--test-per-class", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_benchmarks"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    city_dir = args.data_root / args.city
    if not city_dir.is_dir():
        raise FileNotFoundError(f"City folder not found: {city_dir}")
    by_class = discover_samples(city_dir)
    if not by_class:
        raise RuntimeError(f"No CLASS_* samples found under {city_dir}")
    n_classes = len(by_class)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for shot in args.shots:
        (
            x_tr,
            y_tr,
            x_val,
            y_val,
            x_te,
            y_te,
        ) = split_per_class(by_class, shot, args.val_per_class, args.test_per_class)
        train_ds = SpectrogramDataset(x_tr, y_tr)
        val_ds = SpectrogramDataset(x_val, y_val)
        test_ds = SpectrogramDataset(x_te, y_te)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        shot_key = str(shot)
        summary[shot_key] = {}
        for model_name in args.models:
            model = build_model(model_name, n_classes)
            model = train_one(model, train_loader, val_loader, device, args.epochs, args.lr)
            acc, f1 = evaluate(model, test_loader, device)
            summary[shot_key][model_name] = {"accuracy": acc, "f1": f1}
            print(f"[shot={shot}] {model_name}: acc={acc:.4f} f1={f1:.4f}")

    out_path = args.output_dir / "baseline_benchmarks_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
