#!/usr/bin/env python3
"""
Export Real-World IQ HDF5 subsets to spectrogram containers for the MoE pipeline.

Input:
    HDF5 files with:
      - X: IQ samples, shape (N, 1024, 2)
      - y_mod: modulation labels (int)
      - y_chan: channel type (0 = clean, 1 = multipath)
      - y_snr: SNR in dB

Output layout (matching MoE data discovery expectations):

    <output>/
      <city>/LTE/CLASS_<k>/rate1/SNR10dB/realworld_train_CLASS_<k>.npy

Each .npy file stores an array of shape (Nk, 128, 128) with magnitude spectrograms
derived from the 1024-length IQ sequences via STFT, zero-padded/cropped to 128x128.

Notes:
  - We currently export only subset_train.h5 (train split); MoE will create its
    own train/val/test splits on top of these containers.
  - We group by y_mod only and treat everything as a single comm="LTE". This is
    sufficient for experimenting with MoE + prompts on Real-World IQ, but does
    not reproduce the exact LTE/WiFi/5G mapping used in the prepared .pt export.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    import h5py  # type: ignore
except ImportError as e:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "h5py is required for this script. Install with:\n\n"
        "  pip install h5py\n"
    ) from e


def make_spectrogram(iq: np.ndarray, n_fft: int = 128, hop_length: int = 8) -> np.ndarray:
    """Convert one IQ sequence (1024, 2) to a 128x128 magnitude spectrogram.

    We use torch.stft on the complex IQ, then zero-pad/crop magnitude to (128, 128).
    """
    if iq.shape != (1024, 2):
        raise ValueError(f"Expected IQ shape (1024, 2), got {iq.shape}")

    x = torch.from_numpy(iq).float()  # (1024, 2)
    z = torch.complex(x[:, 0], x[:, 1])  # (1024,)

    spec = torch.stft(
        z,
        n_fft=n_fft,
        hop_length=hop_length,
        center=False,
        return_complex=True,
    )  # (freq, frames)
    mag = spec.abs().numpy().astype(np.float32)

    H, W = mag.shape
    out = np.zeros((128, 128), dtype=np.float32)
    h = min(128, H)
    w = min(128, W)
    out[:h, :w] = mag[:h, :w]
    return out


def export_split(
    h5_path: Path,
    output_root: Path,
    city: str,
    max_per_mod: int | None = None,
) -> None:
    """Export one HDF5 split (e.g. subset_train.h5) into spectrogram containers."""
    print(f"[EXPORT] Loading {h5_path}")
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]  # (N, 1024, 2)
        y_mod = f["y_mod"][:]  # (N,)

    if X.ndim != 3 or X.shape[1:] != (1024, 2):
        raise RuntimeError(f"Unexpected X shape in {h5_path}: {X.shape}")

    N = X.shape[0]
    mods = np.unique(y_mod)
    print(f"[EXPORT] Found {N} samples, {len(mods)} modulation labels: {mods.tolist()}")

    # Collect specs per modulation label
    specs_by_mod: Dict[int, List[np.ndarray]] = {int(m): [] for m in mods}

    for idx in range(N):
        m = int(y_mod[idx])
        if max_per_mod is not None and len(specs_by_mod[m]) >= max_per_mod:
            continue
        spec = make_spectrogram(X[idx])
        specs_by_mod[m].append(spec)
        if (idx + 1) % 10000 == 0:
            print(f"[EXPORT] Processed {idx+1}/{N} samples...")

    # Save per-modulation .npy containers under LTE/CLASS_k/rate1/SNR10dB/
    for m, arr_list in specs_by_mod.items():
        if not arr_list:
            continue
        arr = np.stack(arr_list, axis=0)  # (Nm, 128, 128)
        class_dir = output_root / city / "LTE" / f"CLASS_{m}" / "rate1" / "SNR10dB"
        class_dir.mkdir(parents=True, exist_ok=True)

        out_path = class_dir / "realworld_train_CLASS_{}.npy".format(m)
        print(f"[EXPORT] Saving CLASS_{m}: {arr.shape} -> {out_path}")
        np.save(out_path, arr)

    print("[EXPORT] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Real-World IQ HDF5 subsets to spectrogram npy containers.")
    parser.add_argument(
        "--input-root",
        type=str,
        default="datasets/realworld_iq_raw/Real-World IQ Dataset for Automatic Radio Modulati/dataset",
        help="Directory containing subset_train.h5 (and optionally subset_val.h5, subset_test.h5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/spectrograms_realworld_iq",
        help="Output root for spectrogram containers.",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="city_realworld_iq",
        help="City folder name under the spectrogram root.",
    )
    parser.add_argument(
        "--max-per-mod",
        type=int,
        default=None,
        help="Optional cap on number of samples per modulation label (for faster export).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output)
    city = args.city

    train_path = input_root / "subset_train.h5"
    if not train_path.is_file():
        raise SystemExit(f"subset_train.h5 not found at: {train_path}")

    print(f"[CONFIG] input_root={input_root}")
    print(f"[CONFIG] output_root={output_root}")
    print(f"[CONFIG] city={city}")
    if args.max_per_mod is not None:
        print(f"[CONFIG] max_per_mod={args.max_per_mod}")

    export_split(train_path, output_root, city, max_per_mod=args.max_per_mod)


if __name__ == "__main__":
    main()

