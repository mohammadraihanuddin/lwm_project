#!/usr/bin/env python3
"""
Export IEEE Dataport prepared .pt to the directory layout expected by
task1/train_mcs_models.py and task2/train_joint_snr_mobility.py.

Expects a .pt file in demo_data format: list of dicts with 'data' (spectrogram),
'mod' or 'modulation', and optionally 'tech' (LTE/WiFi/5G). If 'tech' is present,
samples are exported under that comm folder so MoE sees LTE, WiFi, 5G. Uses
placeholder SNR/mobility so paths are valid.

Layout: data_root/city/comm/SNR/mobility/modulation/rate1/512FFT/spectrograms/*.pkl

Usage:
  python scripts/export_ieee_to_spectrograms_dir.py --input datasets/prepared/ieee_dataport.pt --output datasets/spectrograms_ieee
  python scripts/export_ieee_to_spectrograms_dir.py --input datasets/prepared/ieee_dataport.pt --output datasets/spectrograms_ieee --city city_ieee
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=REPO_ROOT / "datasets" / "prepared" / "ieee_dataport.pt", help="IEEE .pt path (demo_data format)")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "datasets" / "spectrograms_ieee", help="Output root")
    p.add_argument("--samples-per-pkl", type=int, default=500, help="Samples per pkl file")
    p.add_argument("--city", type=str, default="city_ieee_dataport", help="City folder name")
    p.add_argument("--comm", type=str, default=None, help="If set, use single comm folder for all (legacy); else use per-sample 'tech' from .pt")
    p.add_argument("--snr", type=str, default="SNR10dB", help="SNR folder name (placeholder)")
    p.add_argument("--mobility", type=str, default="static", help="Mobility folder name (placeholder)")
    p.add_argument("--fft-folder", type=str, default="512FFT")
    p.add_argument("--classes", nargs="*", default=None, help="Only export these modulation names (e.g. QPSK 16QAM 64QAM). Default: all.")
    args = p.parse_args()

    path = Path(args.input).resolve()
    if not path.exists():
        raise SystemExit(f"Input not found: {path}. Run scripts/prepare_ieee_dataport_for_moe.py first.")

    try:
        data = torch.load(path, weights_only=False)
    except TypeError:
        # PyTorch < 1.13 has no weights_only
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, list) or not data or "data" not in data[0]:
        raise SystemExit("Expected list of dicts with 'data' (ieee_dataport.pt / demo_data format)")

    use_tech = args.comm is None
    if use_tech:
        # Group by (comm, mod) so we get LTE/, WiFi/, 5G/ folders
        groups = defaultdict(lambda: defaultdict(list))  # comm -> mod -> [specs]
        for item in data:
            spec = item["data"]
            if hasattr(spec, "numpy"):
                spec = spec.numpy().squeeze()
            else:
                spec = np.asarray(spec).squeeze()
            if spec.ndim == 2:
                spec = spec[None, ...]
            mod = str(item.get("mod") or item.get("modulation", "CLASS_0")).strip()
            comm = str(item.get("tech", "LTE")).strip()
            groups[comm][mod].append(spec.astype(np.float32))
    else:
        # Legacy: single comm folder, group by mod only
        groups = defaultdict(list)
        for item in data:
            spec = item["data"]
            if hasattr(spec, "numpy"):
                spec = spec.numpy().squeeze()
            else:
                spec = np.asarray(spec).squeeze()
            if spec.ndim == 2:
                spec = spec[None, ...]
            mod = str(item.get("mod") or item.get("modulation", "CLASS_0")).strip()
            groups[mod].append(spec.astype(np.float32))

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    n_files = 0

    if use_tech:
        # Global (comm, mod) -> CLASS_i so task labels are consistent across LTE/WiFi/5G
        if args.classes:
            class_set = set(args.classes)
            groups = {c: {m: s for m, s in list(v.items()) if m in class_set} for c, v in groups.items()}
        all_comm_mod = sorted((c, m) for c in groups for m in groups[c])
        comm_mod_to_global = {(c, m): i for i, (c, m) in enumerate(all_comm_mod)}
        comm_order = sorted(groups.keys())
        for comm in comm_order:
            for mod, specs_list in sorted(groups[comm].items()):
                global_idx = comm_mod_to_global[(comm, mod)]
                specs = np.concatenate(specs_list, axis=0)
                folder_name = f"CLASS_{global_idx}"
                dir_path = out_root / args.city / comm / args.snr / args.mobility / folder_name / "rate1" / args.fft_folder / "spectrograms"
                dir_path.mkdir(parents=True, exist_ok=True)
                n_per_file = args.samples_per_pkl
                for start in range(0, len(specs), n_per_file):
                    chunk = specs[start : start + n_per_file]
                    pkl_path = dir_path / f"specs_{n_files:04d}.pkl"
                    with open(pkl_path, "wb") as f:
                        pickle.dump({"spectrograms": chunk}, f)
                    n_files += 1
        total_samples = sum(len(np.concatenate(specs)) for c in groups for specs in groups[c].values())
        print(f"[SAVE] {n_files} pkl files under {out_root}")
        print(f"  Layout: {args.city}/<comm>/{args.snr}/{args.mobility}/<CLASS_*>/rate1/{args.fft_folder}/spectrograms/*.pkl")
        print(f"  Comms: {comm_order}, {len(comm_mod_to_global)} global classes ({total_samples} total samples)")
    else:
        if args.classes:
            class_set = set(args.classes)
            groups = {k: v for k, v in groups.items() if k in class_set}
            missing = class_set - set(groups.keys())
            if missing:
                print(f"[WARN] Requested classes not found in data: {missing}")
            if not groups:
                raise SystemExit("No data left after --classes filter.")
        sorted_mods = sorted(groups.keys())
        for idx, mod in enumerate(sorted_mods):
            specs = np.concatenate(groups[mod], axis=0)
            folder_name = f"CLASS_{idx}"
            dir_path = out_root / args.city / args.comm / args.snr / args.mobility / folder_name / "rate1" / args.fft_folder / "spectrograms"
            dir_path.mkdir(parents=True, exist_ok=True)
            n_per_file = args.samples_per_pkl
            for start in range(0, len(specs), n_per_file):
                chunk = specs[start : start + n_per_file]
                pkl_path = dir_path / f"specs_{n_files:04d}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump({"spectrograms": chunk}, f)
                n_files += 1
        print(f"[SAVE] {n_files} pkl files under {out_root}")
        print(f"  Layout: {args.city}/{args.comm}/{args.snr}/{args.mobility}/<mod>/rate1/{args.fft_folder}/spectrograms/*.pkl")
        print(f"  Classes: {sorted_mods} -> CLASS_0..CLASS_{len(sorted_mods)-1} ({sum(len(s) for s in groups.values())} total samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
