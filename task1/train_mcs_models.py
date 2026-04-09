"""Minimal train_mcs_models stubs for MoE/task1 imports (Res1DCNNHead, load_all_samples, MODULATION_LABELS)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Modulation label mapping (name -> int) for task metadata
MODULATION_LABELS: dict[str, int] = {
    "BPSK": 0,
    "QPSK": 1,
    "8PSK": 2,
    "16QAM": 3,
    "32QAM": 4,
    "64QAM": 5,
    "128QAM": 6,
    "256QAM": 7,
    "LTE": 0,
    "WIFI": 1,
    "5G": 2,
    "WIFI30": 1,
    "WIFI50": 1,
    "LTE30": 0,
    "LTE50": 0,
    "5G-NR": 2,
    "CLASS_0": 0,
    "CLASS_1": 1,
    "CLASS_2": 2,
    "CLASS_3": 3,
    "CLASS_4": 4,
    "CLASS_5": 5,
    "CLASS_6": 6,
    "SPECTROGRAMS": 0,
}


def load_all_samples(path: str | Path) -> np.ndarray:
    """Load all spectrogram samples from a pickle or .npy file. Returns array of shape (N, H, W)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, dict):
            for key in ("spectrograms", "X", "data", "specs"):
                if key in data:
                    arr = data[key]
                    if isinstance(arr, np.ndarray):
                        return arr.astype(np.float32, copy=False)
        return np.array(data, dtype=np.float32)
    except (ValueError, OSError, TypeError):
        pass
    try:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, dict):
            for key in ("spectrograms", "X", "data", "specs"):
                if key in data:
                    arr = data[key]
                    if isinstance(arr, np.ndarray):
                        return arr.astype(np.float32, copy=False)
        return np.array(data, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}") from e


class Res1DCNNHead(nn.Module):
    """1D residual-style head: input (B, in_dim) -> (B, num_classes)."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, in_dim)
        self.fc_out = nn.Linear(in_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        h = torch.relu(h + x)
        return self.fc_out(self.dropout(h))


def apply_normalization(specs: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    """Apply normalization to spectrograms using dataset stats. Stub for plot scripts."""
    if stats is None:
        return specs.astype(np.float32, copy=False)
    norm = (stats.get("normalization") or "per_sample").lower()
    if norm == "per_sample":
        mean = specs.mean(axis=(1, 2), keepdims=True)
        std = np.maximum(specs.std(axis=(1, 2), keepdims=True), 1e-6)
        return ((specs - mean) / std).astype(np.float32, copy=False)
    mean = float(stats.get("mean", 0.0))
    std = max(float(stats.get("std", 1.0)), 1e-6)
    return ((specs - mean) / std).astype(np.float32, copy=False)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
