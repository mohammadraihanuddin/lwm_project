#!/usr/bin/env python3
"""Train a router that selects top-k LWM backbones and feeds a shared classifier.

This variant differs from ``train_top1_router.py`` by:
  * loading each expert checkpoint only for its backbone (classifier discarded)
  * extracting 128-d embeddings with the standard mean-pooled LWM features
  * selecting the top-k experts per sample (default k=2) via a router network
  * feeding each selected embedding through a shared Res1DCNN head
  * weighting the per-embedding logits using the router probabilities

The script auto-discovers six experts by default:
  1. Latest ``lwm_epoch*.pth`` from ``models/LTE_models`` (base LTE backbone)
  2. Latest ``lwm_epoch*.pth`` from ``models/WiFi_models`` (base WiFi backbone)
  3. Latest ``lwm_epoch*.pth`` from ``models/5G_models`` (base 5G backbone)
  4. Most recent epoch checkpoint from ``task2/mobility_benchmark/lte`` (LTE mobility expert)
  5. Most recent epoch checkpoint from ``task2/mobility_benchmark/wifi`` (WiFi mobility expert)
  6. Most recent epoch checkpoint from ``task2/mobility_benchmark/5g`` (5G mobility expert)

Each expert must expose dataset statistics (mean/std or per-sample flag).  The router
is first warmed up on communication labels, then fine-tuned jointly with the shared
classifier using both task loss and an auxiliary communication loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERT_ROOT = REPO_ROOT / "MoE" / "experts"

sys.path.append(str(REPO_ROOT))

from task1.train_mcs_models import Res1DCNNHead, load_all_samples  # type: ignore
from task2.mobility_utils import prepare_model  # type: ignore
from MoE.train_top1_router import (
    SampleMetadata,
    _collect_candidate_files,
    load_dataset_stats,
    snr_sort_key,
)  # type: ignore


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


COMM_CANONICAL = {"lte": "LTE", "wifi": "WiFi", "5g": "5G"}
ExpertIndex = int


@dataclass(slots=True)
class SampleEntry:
    """Describe a single spectrogram sample stored inside a pickled tensor file."""

    path: Path
    index: int
    metadata: SampleMetadata


def canonical_comm_name(name: str) -> str:
    lower = name.strip().lower()
    if lower in COMM_CANONICAL:
        return COMM_CANONICAL[lower]
    for canonical in COMM_CANONICAL.values():
        if canonical.lower() == lower:
            return canonical
    raise ValueError(f"Unknown communication type: {name}")


def discover_latest_base_checkpoint(comm: str) -> Path:
    # Normalize communication type name (lte -> LTE)
    comm_upper = canonical_comm_name(comm)
    
    # Try multiple possible locations
    possible_folders = [
        REPO_ROOT / "models" / "experts" / "baseline" / f"{comm_upper}_models",
        REPO_ROOT / "models" / f"{comm_upper}_models",
        REPO_ROOT / "models" / f"{comm.capitalize()}_models",
    ]
    
    folder = None
    for candidate in possible_folders:
        if candidate.exists():
            folder = candidate
            break
    
    if folder is None:
        raise FileNotFoundError(
            f"Base model directory not found for {comm}. Tried: {[str(p) for p in possible_folders]}"
        )
    
    preferred_names = {
        "LTE": "lteExpert.pth",
        "WiFi": "wifiExpert.pth",
        "5G": "5gExpert.pth",
    }
    preferred = preferred_names.get(comm_upper)
    if preferred:
        preferred_path = folder / preferred
        if preferred_path.exists():
            return preferred_path

    legacy_candidates = sorted(folder.glob("lwm_epoch*_val*.pth"))
    if legacy_candidates:
        def key(path: Path) -> Tuple[float, float]:
            match = re.search(r"epoch(\d+)_val([\d.]+)\.pth$", path.name)
            if match:
                epoch = float(match.group(1))
                val = float(match.group(2))
                return val, -epoch
            return float("inf"), 0.0

        return min(legacy_candidates, key=key)

    expert_candidates = sorted(folder.glob("*Expert.pth"))
    if expert_candidates:
        return expert_candidates[0]

    raise FileNotFoundError(f"No base checkpoints found under {folder}")


def discover_latest_mobility_checkpoint(comm: str) -> Path:
    # Normalize communication type name (lte -> LTE)
    comm_upper = canonical_comm_name(comm)
    
    # Try multiple possible locations
    possible_locations = [
        # First try the experts/task2 folder
        REPO_ROOT / "models" / "experts" / "task2" / f"{comm_upper}_models",
        # Then try task2/mobility_benchmark
        REPO_ROOT / "task2" / "mobility_benchmark" / comm.lower(),
    ]
    
    for base_dir in possible_locations:
        if not base_dir.exists():
            continue
        
        # If it's a models directory, look for checkpoint files directly
        if "_models" in base_dir.name:
            candidates = sorted(base_dir.glob("*.pth"))
            if candidates:
                return candidates[-1]
        
        # Otherwise, it's a run directory - search for checkpoints
        run_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
        if run_dirs:
            for run_dir in reversed(run_dirs):
                epoch_dir = run_dir / "epoch_checkpoints"
                if epoch_dir.exists():
                    epochs = sorted(epoch_dir.glob("epoch_*.pth"))
                    if epochs:
                        return epochs[-1]
                # Fallback: search recursively for *.pth within run
                candidates = sorted(run_dir.rglob("*.pth"))
                if candidates:
                    return candidates[-1]
    
    raise FileNotFoundError(
        f"No mobility checkpoints found for {comm}. Tried: {[str(p) for p in possible_locations]}"
    )


@dataclass(slots=True)
class ExpertSpec:
    name: str
    comm: str
    checkpoint: Path
    stats_path: Optional[Path] = None


def infer_comm_from_path(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    stem = path.stem.lower()
    filename = path.name.lower()
    for part in parts:
        for key, canonical in COMM_CANONICAL.items():
            canonical_lower = canonical.lower()
            if key in part or canonical_lower in part:
                return canonical
    for key, canonical in COMM_CANONICAL.items():
        canonical_lower = canonical.lower()
        if (
            stem.startswith(f"{key}_")
            or stem.startswith(f"{canonical_lower}_")
            or key in filename
            or canonical_lower in filename
        ):
            return canonical
    return None


def discover_experts_from_directory(base_dir: Path) -> List[ExpertSpec]:
    specs: List[ExpertSpec] = []
    if not base_dir.exists():
        return specs
    for ckpt in sorted(base_dir.rglob("*.pth")):
        comm = infer_comm_from_path(ckpt)
        if comm is None:
            print(f"[WARN] Unable to infer communication type for expert at {ckpt}; skipping")
            continue
        name = ckpt.stem
        # Stats path is optional - we use per-sample normalization
        stats_candidates = [
            ckpt.with_suffix(".json"),
            ckpt.parent / "dataset_stats.json",
            REPO_ROOT / "models" / f"{comm}_models" / "dataset_stats.json",
        ]
        stats_path: Optional[Path] = None
        for candidate in stats_candidates:
            if candidate.exists():
                stats_path = candidate
                break
        
        specs.append(
            ExpertSpec(
                name=name,
                comm=comm,
                checkpoint=ckpt.resolve(),
                stats_path=stats_path.resolve() if stats_path else None,
            )
        )
    return specs


def discover_default_experts() -> List[ExpertSpec]:
    directory_specs = discover_experts_from_directory(DEFAULT_EXPERT_ROOT)
    if directory_specs:
        print(f"[INFO] Discovered {len(directory_specs)} expert(s) under {DEFAULT_EXPERT_ROOT}")
        return directory_specs

    specs: List[ExpertSpec] = []
    for comm in ("lte", "wifi", "5g"):
        pretty = canonical_comm_name(comm)
        base_ckpt = discover_latest_base_checkpoint(comm)
        # Stats path is optional - we use per-sample normalization
        stats_path = base_ckpt.parent / "dataset_stats.json"
        specs.append(
            ExpertSpec(
                name=f"{pretty}_base",
                comm=pretty,
                checkpoint=base_ckpt,
                stats_path=stats_path if stats_path.exists() else None,
            )
        )
    for comm in ("lte", "wifi", "5g"):
        pretty = canonical_comm_name(comm)
        ckpt = discover_latest_mobility_checkpoint(comm)
        # Stats path is optional - we use per-sample normalization
        stats_candidates = [
            ckpt.parent / "dataset_stats.json",
            (REPO_ROOT / "models" / f"{pretty}_models" / "dataset_stats.json"),
        ]
        stats_path = next((p for p in stats_candidates if p.exists()), None)
        specs.append(
            ExpertSpec(
                name=f"{pretty}_mobility",
                comm=pretty,
                checkpoint=ckpt,
                stats_path=stats_path,
            )
        )
    return specs


def collect_sample_entries_for_comm(
    *,
    data_root: Path,
    cities: Sequence[str],
    comm: str,
    snrs: Optional[Sequence[str]],
    mobilities: Optional[Sequence[str]],
    modulations: Optional[Sequence[str]],
    fft_folders: Optional[Sequence[str]],
    max_samples: int,
    max_per_combo: Optional[int],
    target_per_combo: Optional[int],
    rng: np.random.Generator,
) -> List[SampleEntry]:
    """Gather per-sample references without materialising full tensors."""
    candidates = _collect_candidate_files(
        data_root=data_root,
        cities=cities,
        comm=comm,
        snr_filters=snrs,
        mobility_filters=mobilities,
        modulation_filters=modulations,
        fft_filters=fft_folders,
    )
    if not candidates:
        raise RuntimeError(f"No spectrogram files matched filters for {comm}")

    # Resolve all paths upfront to avoid repeated resolve() calls
    candidates = [(path.resolve(), meta) for path, meta in candidates]
    rng.shuffle(candidates)
    combo_counts = defaultdict(int)
    entries: List[SampleEntry] = []
    remaining = max_samples if max_samples > 0 else None
    per_combo_limit = max_per_combo if (max_per_combo is not None and max_per_combo > 0) else None
    combos_available: Set[Tuple[str, str, str]] = {
        (meta.modulation, meta.snr, meta.mobility) for _, meta in candidates
    }
    combo_targets: Optional[Dict[Tuple[str, str, str], int]] = None
    satisfied_combos: Set[Tuple[str, str, str]] = set()
    warned_combo_limit = False
    if target_per_combo is not None:
        combo_targets = {}
        for combo in combos_available:
            target = target_per_combo
            if per_combo_limit is not None:
                effective = min(target, per_combo_limit)
                if effective < target and not warned_combo_limit:
                    print(
                        f"[WARN] {comm}: per-combo limit ({per_combo_limit}) is below requested total ({target}); "
                        "consider relaxing --max-per-combo or per-class caps."
                    )
                    warned_combo_limit = True
                target = effective
            combo_targets[combo] = target
            if target <= 0:
                satisfied_combos.add(combo)

    files_processed = 0
    for file_idx, (path, meta) in enumerate(candidates, start=1):
        files_processed = file_idx
        if remaining is not None and remaining <= 0:
            break
        combo_key = (meta.modulation, meta.snr, meta.mobility)
        already = combo_counts[combo_key]
        if per_combo_limit is not None and already >= per_combo_limit:
            continue

        path_str = str(path)
        try:
            # Fast metadata-only read to get sample count
            num_samples = get_sample_count_fast(path_str)
        except Exception as exc:  # pragma: no cover - guard against corrupted files
            print(f"[WARN] Failed to load {path_str}: {exc}")
            continue

        if num_samples == 0:
            continue

        remaining_for_combo = (
            per_combo_limit - already if per_combo_limit is not None else num_samples
        )
        allowed = min(num_samples, remaining_for_combo)
        if remaining is not None:
            allowed = min(allowed, remaining)
        if allowed <= 0:
            continue

        if allowed == num_samples:
            chosen_indices = np.arange(num_samples)
        else:
            chosen_indices = rng.choice(num_samples, size=allowed, replace=False)

        # Reuse metadata for all samples from same file
        entry_meta = SampleMetadata(
            path=path,
            comm=meta.comm,
            modulation=meta.modulation,
            snr=meta.snr,
            mobility=meta.mobility,
            rate=meta.rate,
            source=path_str,
        )
        
        # Batch create and extend entries (faster than repeated append)
        batch_entries = [
            SampleEntry(path=path, index=int(idx), metadata=entry_meta)
            for idx in chosen_indices.tolist()
        ]
        entries.extend(batch_entries)

        combo_counts[combo_key] += int(len(chosen_indices))
        if remaining is not None:
            remaining -= int(len(chosen_indices))
        # Only log at major milestones to reduce overhead
        if file_idx == 1 or file_idx % 50 == 0:
            print(f"[DATA] {comm}: gathered {len(entries):,} samples after {file_idx} files", flush=True)
        if combo_targets is not None:
            target = combo_targets.get(combo_key)
            if target is not None and combo_counts[combo_key] >= target:
                satisfied_combos.add(combo_key)
            if len(satisfied_combos) == len(combo_targets):
                break

    if not entries:
        raise RuntimeError(f"Unable to collect samples for {comm} after applying limits")
    if combo_targets is not None:
        unmet = [
            combo for combo, target in combo_targets.items() if combo_counts[combo] < target
        ]
        if unmet:
            print(
                f"[WARN] {comm}: target not met for {len(unmet)} combo(s); "
                "consider lowering per-class requirements.",
                flush=True,
            )
    print(
        f"[DATA] {comm}: gathered {len(entries)} samples after scanning {files_processed} files",
        flush=True,
    )
    return entries


def iterate_batches(loader: Iterable, desc: str, *, log_every: Optional[int] = None) -> Iterable:
    """Yield from a loader while emitting progress information."""
    if tqdm is not None:
        for item in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            yield item
        return

    try:
        total = len(loader)  # type: ignore[arg-type]
    except Exception:
        total = None
    if log_every is None:
        if total:
            log_every = max(1, total // 10)
        else:
            log_every = 50
    for idx, batch in enumerate(loader, start=1):
        if idx == 1 or idx % log_every == 0 or (total is not None and idx == total):
            if total is not None:
                print(f"[Progress] {desc}: {idx}/{total}", flush=True)
            else:
                print(f"[Progress] {desc}: batch {idx}", flush=True)
        yield batch


def _get_cache_capacity(default: int = 32) -> int:
    """Derive cache size from env while guarding against invalid values."""
    override = os.environ.get("LWM_FILE_CACHE_SIZE")
    if not override:
        return default
    try:
        value = int(override)
        return max(1, value)
    except ValueError:
        print(
            f"[WARN] Ignoring invalid LWM_FILE_CACHE_SIZE={override!r}; using default {default}",
            flush=True,
        )
        return default


_FILE_CACHE_SIZE = _get_cache_capacity()


def _resolve_preload_dtype(default: str = "float16") -> torch.dtype:
    override = os.environ.get("LWM_PRELOAD_DTYPE", default)
    alias = override.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "single": torch.float32,
    }
    dtype = mapping.get(alias)
    if dtype is None:
        print(
            f"[WARN] Unknown LWM_PRELOAD_DTYPE={override!r}; defaulting to {default}",
            flush=True,
        )
        dtype = mapping[default]
    return dtype


def _parse_float_env(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        print(f"[WARN] Ignoring invalid {name}={value!r}", flush=True)
        return None


def _available_ram_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # Value reported in kB
                        return int(float(parts[1]) * 1024)
    except Exception:
        pass
    return None


@lru_cache(maxsize=_FILE_CACHE_SIZE)
def _load_file_spectrograms(path_str: str) -> np.ndarray:
    """Load all spectrograms for a path with a bounded LRU cache."""
    arr = load_all_samples(path_str)
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16, copy=False)
    return arr


def get_sample_count_fast(path: str) -> int:
    """Get number of samples in a file. Loads array to determine shape."""
    try:
        arr = load_all_samples(path)
        return int(arr.shape[0])
    except Exception:
        return 0


def load_spec_tensor(entry: SampleEntry) -> torch.Tensor:
    """Materialise a single spectrogram as a float32 tensor."""
    path_str = str(entry.path)
    specs = _load_file_spectrograms(path_str)
    if entry.index < 0 or entry.index >= specs.shape[0]:
        raise IndexError(f"Sample index {entry.index} out of range for {path_str}")
    sample = specs[entry.index]
    if sample.ndim != 2:
        raise ValueError(f"Expected 2-D spectrogram, got shape {sample.shape}")
    # Clone to decouple the tensor lifetime from the cached numpy array.
    # Explicitly ensure float32 dtype for GPU efficiency
    return torch.from_numpy(sample).clone().float()


class EmbeddingRouterDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[SampleEntry],
        comm_labels: np.ndarray,
        task_labels: np.ndarray,
        preload: bool = True,
    ) -> None:
        if not (len(entries) == len(comm_labels) == len(task_labels)):
            raise ValueError("Dataset inputs must share the same length")
        self.entries = list(entries)
        self.comm_labels = torch.from_numpy(comm_labels.astype(np.int64, copy=False))
        self.task_labels = torch.from_numpy(task_labels.astype(np.int64, copy=False))
        self.metadata = [entry.metadata for entry in self.entries]
        
        # Preload all spectrograms into memory for faster training
        self.preload = preload
        self.spectrograms = None
        self.preload_dtype = _resolve_preload_dtype()
        if self.preload:
            element_size = torch.tensor([], dtype=self.preload_dtype).element_size()
            required_bytes = len(entries) * 128 * 128 * element_size
            required_gb = required_bytes / 1e9
            max_gb = _parse_float_env("LWM_PRELOAD_MAX_GB")
            available_bytes = _available_ram_bytes()
            allow_preload = True
            if max_gb is not None and required_gb > max_gb:
                print(
                    f"[WARN] Requested preload requires {required_gb:.2f} GB, "
                    f"exceeding LWM_PRELOAD_MAX_GB={max_gb:.2f}; falling back to streaming.",
                    flush=True,
                )
                allow_preload = False
            elif available_bytes is not None and required_bytes > available_bytes * 0.8:
                available_gb = available_bytes / 1e9
                print(
                    f"[WARN] Requested preload requires {required_gb:.2f} GB but only "
                    f"{available_gb:.2f} GB appears available; falling back to streaming.",
                    flush=True,
                )
                allow_preload = False
            if not allow_preload:
                self.preload = False
        
        file_groups: Optional[Dict[str, List[Tuple[int, int]]]] = None
        if self.preload:
            print(f"[INFO] Preloading {len(entries):,} spectrograms into RAM...")
            # Group entries by file path for efficient loading
            from collections import defaultdict
            groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
            for idx, entry in enumerate(entries):
                groups[str(entry.path)].append((idx, entry.index))
            file_groups = groups
            
            # Preallocate tensor using the configured dtype to control memory footprint
            try:
                self.spectrograms = torch.empty(
                    (len(entries), 128, 128), dtype=self.preload_dtype
                )
            except RuntimeError as exc:
                print(
                    f"[WARN] Failed to allocate preload buffer ({exc}); falling back to streaming.",
                    flush=True,
                )
                self.preload = False
                self.spectrograms = None
        
        if self.preload and self.spectrograms is not None:
            if file_groups is None:
                from collections import defaultdict

                file_groups = defaultdict(list)
                for idx, entry in enumerate(entries):
                    file_groups[str(entry.path)].append((idx, entry.index))
            
            # Load files in batch
            if tqdm is not None:
                iter_files = tqdm(
                    file_groups.items(), desc="Loading files", leave=False, total=len(file_groups)
                )
            else:
                iter_files = file_groups.items()
            
            for path_str, indices_list in iter_files:
                # Load file once
                file_data = load_all_samples(path_str)
                # Extract all needed samples from this file
                for sample_idx, file_offset in indices_list:
                    tensor = torch.from_numpy(file_data[file_offset]).to(dtype=self.preload_dtype)
                    self.spectrograms[sample_idx] = tensor
            
            print(
                f"[INFO] Preloaded {self.spectrograms.shape[0]:,} spectrograms "
                f"({self.spectrograms.element_size() * self.spectrograms.nelement() / 1e9:.2f} GB)"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        if self.preload and self.spectrograms is not None:
            spec = self.spectrograms[idx].to(dtype=torch.float32)
        else:
            entry = self.entries[idx]
            spec = load_spec_tensor(entry)
        return spec, int(self.comm_labels[idx]), int(self.task_labels[idx])


def modulation_labels_from_metadata(metadata: Sequence[SampleMetadata]) -> np.ndarray:
    from task1.train_mcs_models import MODULATION_LABELS  # type: ignore

    labels: List[int] = []
    for meta in metadata:
        label = MODULATION_LABELS.get(meta.modulation.upper())
        if label is None:
            raise ValueError(f"Unknown modulation label in metadata: {meta.modulation}")
        labels.append(label)
    return np.array(labels, dtype=np.int64)


def snr_mobility_labels_from_metadata(
    metadata: Sequence[SampleMetadata],
    *,
    snr_order: Sequence[str],
    mobility_order: Sequence[str],
) -> Tuple[np.ndarray, Dict[int, Tuple[str, str]]]:
    combos: List[Tuple[str, str]] = []
    for snr in snr_order:
        for mobility in mobility_order:
            combos.append((snr, mobility))
    combo_to_idx = {combo: idx for idx, combo in enumerate(combos)}

    labels: List[int] = []
    for meta in metadata:
        combo = (meta.snr, meta.mobility)
        if combo not in combo_to_idx:
            raise ValueError(f"Sample combo {combo} not present in configured (snr, mobility) grid")
        labels.append(combo_to_idx[combo])
    mapping = {idx: combo for combo, idx in combo_to_idx.items()}
    return np.array(labels, dtype=np.int64), mapping


def prepare_dataset(
    *,
    data_root: Path,
    cities: Sequence[str],
    comm_types: Sequence[str],
    snrs: Optional[Sequence[str]],
    mobilities: Optional[Sequence[str]],
    modulations: Optional[Sequence[str]],
    fft_folders: Optional[Sequence[str]],
    max_samples_per_comm: int,
    max_per_combo: Optional[int],
    max_samples_per_class: int,
    val_samples_per_class: int,
    test_samples_per_class: int,
    task: str,
    seed: int,
    preload: bool = True,
) -> Tuple[EmbeddingRouterDataset, Dict[str, int], Optional[Dict[int, Tuple[str, str]]]]:
    rng = np.random.default_rng(seed)
    entries: List[SampleEntry] = []
    comm_labels_list: List[int] = []
    comm_to_idx: Dict[str, int] = {}
    total_required = 0
    have_requirements = False
    if max_samples_per_class > 0:
        total_required += max_samples_per_class
        have_requirements = True
    if val_samples_per_class > 0:
        total_required += val_samples_per_class
        have_requirements = True
    if test_samples_per_class > 0:
        total_required += test_samples_per_class
        have_requirements = True
    target_per_combo = total_required if have_requirements else None

    for comm in comm_types:
        try:
            comm_entries = collect_sample_entries_for_comm(
                data_root=data_root,
                cities=cities,
                comm=comm,
                snrs=snrs,
                mobilities=mobilities,
                modulations=modulations,
                fft_folders=fft_folders,
                max_samples=max_samples_per_comm,
                max_per_combo=max_per_combo,
                target_per_combo=target_per_combo,
                rng=rng,
            )
        except RuntimeError as exc:
            print(f"[WARN] {exc}; skipping {comm}")
            continue
        if comm not in comm_to_idx:
            comm_to_idx[comm] = len(comm_to_idx)
        comm_idx = comm_to_idx[comm]
        entries.extend(comm_entries)
        comm_labels_list.extend([comm_idx] * len(comm_entries))

    if not entries:
        raise RuntimeError("No spectrogram data collected for any communication type")

    comm_labels = np.array(comm_labels_list, dtype=np.int64)
    order = rng.permutation(len(entries))
    entries = [entries[idx] for idx in order]
    comm_labels = comm_labels[order]
    metadata = [entry.metadata for entry in entries]

    if task == "modulation":
        task_labels = modulation_labels_from_metadata(metadata)
        mapping = None
    else:
        if snrs is None:
            snr_order = sorted({meta.snr for meta in metadata}, key=snr_sort_key)
        else:
            snr_order = [snr for snr in snrs if any(meta.snr == snr for meta in metadata)]
        if mobilities is None:
            mobility_order = sorted({meta.mobility for meta in metadata})
        else:
            mobility_order = [mob for mob in mobilities if any(meta.mobility == mob for meta in metadata)]
        task_labels, mapping = snr_mobility_labels_from_metadata(
            metadata,
            snr_order=snr_order,
            mobility_order=mobility_order,
        )

    dataset = EmbeddingRouterDataset(entries, comm_labels, task_labels, preload=preload)

    # Print data statistics
    print(f"\n[DATA] Collected {len(dataset)} total samples:")
    for comm_name, comm_idx in sorted(comm_to_idx.items(), key=lambda x: x[1]):
        count = int((comm_labels == comm_idx).sum())
        print(f"  {comm_name}: {count:,} samples")
    print()
    
    return dataset, comm_to_idx, mapping


def stratified_split(
    labels: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    max_train_per_class: int = 0,
    val_samples_per_class: int = 0,
    test_samples_per_class: int = 0,
    zero_shot_eval: bool = False,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0, 1)")
    if (
        val_samples_per_class <= 0
        and test_samples_per_class <= 0
        and train_ratio + val_ratio >= 1.0
    ):
        raise ValueError("train_ratio + val_ratio must be < 1.0 when using ratios for all splits")

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    base_test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_total = idx.size
        n_val = (
            min(val_samples_per_class, n_total)
            if val_samples_per_class > 0
            else int(math.floor(val_ratio * n_total))
        )
        if val_samples_per_class > 0 and n_val < val_samples_per_class:
            print(
                f"[WARN] Class {label}: requested {val_samples_per_class} validation samples but only {n_total} available"
            )
        remaining_after_val = max(n_total - n_val, 0)
        n_test = (
            min(test_samples_per_class, remaining_after_val)
            if test_samples_per_class > 0
            else int(math.floor(base_test_ratio * n_total))
        )
        if test_samples_per_class > 0 and n_test < test_samples_per_class:
            print(
                f"[WARN] Class {label}: requested {test_samples_per_class} test samples but only {remaining_after_val} available after validation"
            )
        n_test = min(n_test, remaining_after_val)
        base_train = remaining_after_val - n_test
        if base_train < 0:
            n_test = max(0, remaining_after_val)
            base_train = remaining_after_val - n_test
        if zero_shot_eval:
            n_train = 0
        elif max_train_per_class > 0 and base_train > max_train_per_class:
            overflow = base_train - max_train_per_class
            n_train = max_train_per_class
            n_test = min(n_test + overflow, remaining_after_val)
        else:
            n_train = base_train

        used = n_val + n_test + n_train
        if used < n_total:
            # Prefer allocating leftovers to the test split for more evaluation coverage.
            extra = min(n_total - used, remaining_after_val - n_test)
            n_test += extra
            used = n_val + n_test + n_train
        if used > n_total:
            overflow = used - n_total
            reduction = min(overflow, n_test)
            n_test -= reduction
            overflow -= reduction
            if overflow > 0:
                n_val = max(0, n_val - overflow)
                overflow = 0
            used = n_val + n_test + n_train

        if n_val < 0 or n_test < 0 or n_train < 0:
            raise RuntimeError(
                f"Negative split size encountered for class {label}: "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

        start_train = n_val + n_test
        train_indices.extend(idx[start_train:start_train + n_train])
        val_indices.extend(idx[:n_val])
        test_indices.extend(idx[n_val:n_val + n_test])

    return (
        np.sort(np.array(train_indices, dtype=np.int64)),
        np.sort(np.array(val_indices, dtype=np.int64)),
        np.sort(np.array(test_indices, dtype=np.int64)),
    )


class RouterNet(nn.Module):
    """Lightweight CNN router."""

    def __init__(self, num_experts: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        head_layers: List[nn.Module] = [nn.Flatten()]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(128, num_experts))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        x = specs
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected specs rank 3 or 4, got shape {tuple(specs.shape)}")
        features = self.features(x)
        logits = self.classifier(features)
        return logits


class TaskClassifier(nn.Module):
    """Shared Res1DCNN head operating on 128-d embeddings."""

    def __init__(self, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            Res1DCNNHead(128, num_classes, dropout=dropout),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)


class EmbeddingExpert(nn.Module):
    def __init__(
        self,
        spec: ExpertSpec,
        device: torch.device,
        *,
        trainable: bool = False,
        use_prompts: bool = False,
        num_prompts: int = 16,
        pool_size: int = 10,
        selection_size: int = 5,
        prompt_style: str = "deep",
        prompt_hybrid: bool = False,
    ) -> None:
        super().__init__()
        self._trainable = bool(trainable)
        self._prompt_hybrid = bool(prompt_hybrid)
        # Use per-sample normalization by default if no stats are provided.
        # load_dataset_stats returns (stats_dict, aux) so we unpack the first element.
        if spec.stats_path is not None and spec.stats_path.exists():
            loaded_stats, _ = load_dataset_stats(spec.stats_path)
            if isinstance(loaded_stats, dict):
                stats_dict = loaded_stats
                self.stats = {
                    "normalization": str(stats_dict.get("normalization", "per_sample")).lower(),
                    "mean": float(stats_dict.get("mean", 0.0)),
                    "std": float(stats_dict.get("std", 1.0)),
                }
            else:
                # Fallback if stats file could not be parsed
                self.stats = {
                    "normalization": "per_sample",
                    "mean": 0.0,
                    "std": 1.0,
                }
        else:
            # Default to per-sample normalization when no stats file is available
            self.stats = {
                "normalization": "per_sample",
                "mean": 0.0,
                "std": 1.0,
            }
        # Prepare normalization stats for prepare_model
        # If using per-sample normalization, we don't need to pass dataset stats
        normalization_stats = None
        if self.stats["normalization"] != "per_sample":
            normalization_stats = {
                "normalization": self.stats["normalization"],
                "mean": self.stats["mean"],
                "std": self.stats["std"],
            }
        
        model = prepare_model(
            checkpoint=spec.checkpoint,
            num_classes=2,
            classifier_dim=128,
            dropout=0.0,
            trainable_layers=2 if (trainable and ((use_prompts and self._prompt_hybrid) or not use_prompts)) else 0,
            projection_dim=0,
            append_input_stats=False,
            normalization_stats=normalization_stats,
            head_type="mlp",
            use_prompts=use_prompts,
            num_prompts=num_prompts,
            pool_size=pool_size,
            selection_size=selection_size,
            prompt_style=prompt_style,
        )
        # prepare_model already sets requires_grad correctly based on trainable_layers
        model.train(self._trainable)
        self.model = model.to(device)

    @property
    def trainable(self) -> bool:
        return self._trainable

    def set_trainable(self, trainable: bool) -> None:
        self._trainable = bool(trainable)
        # Freeze all backbone parameters first
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        # If trainable and not using prompt-only variants, enable last 2 layers of backbone.
        # Prompt variants are detected via either a pooled prompt attribute (L2P) or
        # per-layer prompts (DeepPromptWrapper). In those cases we keep the backbone frozen.
        uses_prompt_wrapper = bool(
            getattr(self.model, "pool", None) is not None  # L2P-style pool
            or hasattr(self.model, "layer_prompts")        # Deep prompt wrapper
        )
        should_unfreeze_backbone_tail = self._trainable and (not uses_prompt_wrapper or self._prompt_hybrid)
        if should_unfreeze_backbone_tail:
            layers = getattr(self.model.backbone, "layers", None)
            if layers is not None and len(layers) >= 2:
                for layer in layers[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        self.model.train(self._trainable)
        super().train(self._trainable)

    def train(self, mode: bool = True) -> "EmbeddingExpert":
        effective_mode = bool(mode and self._trainable)
        self.model.train(effective_mode)
        return super().train(mode)

    def eval(self) -> "EmbeddingExpert":
        self.model.eval()
        return super().eval()

    def forward(self, specs: torch.Tensor, *, allow_grad: Optional[bool] = None) -> torch.Tensor:
        use_grad = self._determine_grad(allow_grad)
        x = self._normalize(specs)
        if use_grad:
            return self.model.forward_features(x)
        with torch.no_grad():
            return self.model.forward_features(x)
    
    def forward_prenormalized(
        self,
        specs: torch.Tensor,
        *,
        allow_grad: Optional[bool] = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with pre-normalized spectrograms (skip normalization).
        When return_tokens=True and the model supports it, returns (embeddings, token_features).
        token_features is (B, T, D) or None."""
        use_grad = self._determine_grad(allow_grad)
        if return_tokens:
            if use_grad:
                out = self.model.forward_features(specs, return_tokens=True)
            else:
                with torch.no_grad():
                    out = self.model.forward_features(specs, return_tokens=True)
            if isinstance(out, tuple):
                return out
            return out, None
        if use_grad:
            return self.model.forward_features(specs)
        with torch.no_grad():
            return self.model.forward_features(specs)

    def _normalize(self, specs: torch.Tensor) -> torch.Tensor:
        mode = self.stats["normalization"]
        mean = self.stats["mean"]
        std = max(abs(self.stats["std"]), 1e-6)
        if mode == "dataset":
            return (specs - mean) / std
        mean_tensor = specs.mean(dim=(1, 2), keepdim=True)
        std_tensor = specs.std(dim=(1, 2), keepdim=True, unbiased=False)
        std_tensor = torch.clamp(std_tensor, min=1e-6)
        return (specs - mean_tensor) / std_tensor

    def _determine_grad(self, allow_grad: Optional[bool]) -> bool:
        if allow_grad is None:
            return self._trainable
        return bool(allow_grad)


def normalize_per_sample_tensor(specs: torch.Tensor) -> torch.Tensor:
    mean = specs.mean(dim=(1, 2), keepdim=True)
    std = specs.std(dim=(1, 2), keepdim=True, unbiased=False)
    std = torch.clamp(std, min=1e-6)
    return (specs - mean) / std


def build_dataloaders(
    dataset: EmbeddingRouterDataset,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def _subset(indices: np.ndarray) -> torch.utils.data.Subset:
        return torch.utils.data.Subset(dataset, indices.tolist())

    # Optimize DataLoader configuration based on whether we're using workers
    use_cuda = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None  # Increased prefetch for better pipelining
    
    # Use larger batch size effectively with proper pin_memory and non_blocking transfers
    # NOTE: In zero-shot eval mode, train_idx can be empty. RandomSampler (shuffle=True)
    # requires num_samples > 0, so fall back to a deterministic loader.
    train_shuffle = len(train_idx) > 0
    train_drop_last = len(train_idx) > 0
    train_loader = DataLoader(
        _subset(train_idx),
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=train_drop_last,  # Drop incomplete batches only when training
    )
    val_loader = DataLoader(
        _subset(val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        _subset(test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return train_loader, val_loader, test_loader


def aggregate_comm_probs(
    probs: torch.Tensor,
    group_map: Mapping[int, List[int]],
) -> torch.Tensor:
    num_comm = len(group_map)
    agg = torch.zeros(probs.size(0), num_comm, device=probs.device, dtype=probs.dtype)
    for comm_idx, expert_indices in group_map.items():
        if not expert_indices:
            continue
        agg[:, comm_idx] = probs[:, expert_indices].sum(dim=1)
    return agg


def router_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_map: Mapping[int, List[int]],
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    agg = aggregate_comm_probs(probs, group_map)
    agg = torch.clamp(agg, min=1e-12)
    return F.nll_loss(agg.log(), targets)


def build_group_map(experts: Sequence[ExpertSpec], comm_to_idx: Mapping[str, int]) -> Dict[int, List[int]]:
    grouping: Dict[int, List[int]] = {idx: [] for idx in comm_to_idx.values()}
    for expert_idx, spec in enumerate(experts):
        comm_idx = comm_to_idx[spec.comm]
        grouping[comm_idx].append(expert_idx)
    return grouping


def train_router(
    router: RouterNet,
    *,
    experts: Sequence[ExpertSpec],
    comm_to_idx: Mapping[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, List[float]]:
    group_map = build_group_map(experts, comm_to_idx)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # HPU support
    use_hpu = device.type == "hpu"
    if use_hpu:
        try:
            import habana_frameworks.torch.core as htcore
        except ImportError:
            use_hpu = False
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        router.train()
        running_loss = 0.0
        correct = 0
        total = 0
        desc = f"Router train {epoch:02d}"
        for specs, comm_labels, _ in iterate_batches(train_loader, desc):
            specs = specs.to(device, non_blocking=True)
            comm_labels = comm_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            norm_specs = normalize_per_sample_tensor(specs)
            context = autocast(device_type=device.type, enabled=scaler.is_enabled())
            with context:
                logits = router(norm_specs)
                loss = router_cross_entropy(logits, comm_labels, group_map)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # HPU memory management
            if use_hpu:
                # In eager mode, mark_step doesn't work but we can try synchronization
                try:
                    htcore.mark_step()
                except:
                    pass
                # Force synchronization and memory cleanup
                torch.hpu.synchronize()
                import gc
                gc.collect()
            
            running_loss += loss.item() * specs.size(0)
            probs = torch.softmax(logits.detach(), dim=1)
            agg = aggregate_comm_probs(probs, group_map)
            preds = agg.argmax(dim=1)
            correct += (preds == comm_labels).sum().item()
            total += specs.size(0)
            
            # Clear unused tensors to free memory
            del specs, comm_labels, logits, loss, norm_specs, probs, agg, preds
            
            # Additional memory cleanup for HPU
            if use_hpu and total % 100 == 0:  # Every 100 samples
                torch.hpu.synchronize()
                import gc
                gc.collect()
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate_router(router, val_loader, group_map, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"[Router] Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
    return history


@torch.no_grad()
def evaluate_router(
    router: RouterNet,
    loader: DataLoader,
    group_map: Mapping[int, List[int]],
    device: torch.device,
) -> Tuple[float, float]:
    router.eval()
    
    # HPU support
    use_hpu = device.type == "hpu"
    if use_hpu:
        try:
            import habana_frameworks.torch.core as htcore
        except ImportError:
            use_hpu = False
    
    total_loss = 0.0
    correct = 0
    total = 0
    for specs, comm_labels, _ in iterate_batches(loader, "Router eval"):
        specs = specs.to(device, non_blocking=True)
        comm_labels = comm_labels.to(device, non_blocking=True)
        logits = router(normalize_per_sample_tensor(specs))
        loss = router_cross_entropy(logits, comm_labels, group_map)
        total_loss += loss.item() * specs.size(0)
        probs = torch.softmax(logits, dim=1)
        agg = aggregate_comm_probs(probs, group_map)
        preds = agg.argmax(dim=1)
        correct += (preds == comm_labels).sum().item()
        total += specs.size(0)
        
        # HPU memory management
        if use_hpu:
            htcore.mark_step()
        
        # Clear unused tensors to free memory
        del specs, comm_labels, logits, loss, probs, agg, preds
    
    return total_loss / max(total, 1), correct / max(total, 1)


def stack_expert_embeddings(
    experts: Sequence[EmbeddingExpert],
    specs: torch.Tensor,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    for expert in experts:
        emb = expert(specs)
        embeddings.append(emb.unsqueeze(1))
    return torch.cat(embeddings, dim=1)


def compute_selected_expert_embeddings(
    experts: Sequence[EmbeddingExpert],
    specs_normalized: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    allow_grad: bool,
    return_tokens: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute embeddings only for selected experts (optimized GPU version).
    
    Args:
        experts: List of expert models
        specs_normalized: Pre-normalized input spectrograms [batch_size, H, W]
        topk_indices: Selected expert indices [batch_size, k]
        return_tokens: If True, also return [batch_size, T, embed_dim] tokens from top-1 expert (for LoCoOp).
    
    Returns:
        Selected embeddings [batch_size, k, embed_dim], or (embeddings, tokens) when return_tokens=True.
    """
    batch_size, k = topk_indices.shape
    device = specs_normalized.device
    unique_experts = torch.unique(topk_indices)
    
    output: Optional[torch.Tensor] = None
    output_tokens: Optional[torch.Tensor] = None
    for expert_idx in unique_experts.tolist():
        expert_idx_int = int(expert_idx)
        sample_mask = (topk_indices == expert_idx_int).any(dim=1)
        if not torch.any(sample_mask):
            continue
        sample_indices = sample_mask.nonzero(as_tuple=False).squeeze(1)
        specs_subset = specs_normalized.index_select(0, sample_indices)
        
        expert_model = experts[expert_idx_int]
        if isinstance(expert_model, nn.DataParallel):
            expert_model = expert_model.module
        
        if return_tokens:
            result = expert_model.forward_prenormalized(
                specs_subset,
                allow_grad=allow_grad and expert_model.trainable,
                return_tokens=True,
            )
            embeddings_subset = result[0]
            tokens_subset = result[1]  # (B_sub, T, D) or None
        else:
            embeddings_subset = expert_model.forward_prenormalized(
                specs_subset,
                allow_grad=allow_grad and expert_model.trainable,
            )
            tokens_subset = None
        
        if output is None:
            embed_dim = embeddings_subset.shape[-1]
            output = torch.empty(
                batch_size,
                k,
                embed_dim,
                device=device,
                dtype=embeddings_subset.dtype,
            )
            if return_tokens and tokens_subset is not None:
                T = tokens_subset.shape[1]
                output_tokens = torch.empty(
                    batch_size, T, embed_dim,
                    device=device,
                    dtype=tokens_subset.dtype,
                )
        
        for pos in range(k):
            pos_mask = topk_indices[sample_indices, pos] == expert_idx_int
            if pos_mask.any():
                output[sample_indices[pos_mask], pos] = embeddings_subset[pos_mask]
                if pos == 0 and output_tokens is not None and tokens_subset is not None:
                    output_tokens[sample_indices[pos_mask], :, :] = tokens_subset[pos_mask]
    
    if output is None:
        raise RuntimeError("No experts selected for current batch.")
    
    if return_tokens:
        return output, output_tokens
    return output


def gather_topk_embeddings(
    embeddings: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    batch, k = topk_indices.shape
    feature_dim = embeddings.size(-1)
    expanded_indices = topk_indices.unsqueeze(-1).expand(batch, k, feature_dim)
    return embeddings.gather(dim=1, index=expanded_indices)


def _classifier_class_weights(classifier: nn.Module) -> Optional[torch.Tensor]:
    """Extract (num_classes, D) weight matrix from the last Linear of the classifier."""
    last_linear = None
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        return None
    return last_linear.weight  # (num_classes, in_features); keep grad for LoCoOp


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for imbalanced classification. Down-weights easy examples."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
    pt = (probs * targets_one_hot).sum(dim=-1)
    focal_weight = (1.0 - pt) ** gamma
    ce = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = focal_weight * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def compute_class_weights_from_indices(
    task_labels: torch.Tensor,
    indices: np.ndarray,
    num_classes: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Inverse frequency class weights from train split. Shape (num_classes,)."""
    labels = task_labels.numpy()
    train_labels = labels[indices]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64) + smooth
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights.astype(np.float32))


def compute_class_weights_from_loader(
    train_loader: DataLoader,
    num_classes: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Inverse frequency class weights from one pass over train_loader. Shape (num_classes,)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _batch in train_loader:
        if len(_batch) >= 3:
            _task = _batch[2]
        else:
            _task = _batch[1]
        if isinstance(_task, torch.Tensor):
            _task = _task.numpy()
        counts += np.bincount(_task, minlength=num_classes).astype(np.float64)
    counts = counts + smooth
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights.astype(np.float32))


def locoop_nuisance_loss(
    tokens: torch.Tensor,
    class_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    nuisance_frac: float = 0.5,
    margin: float = 0.0,
) -> torch.Tensor:
    """LoCoOp-style loss: push low-similarity (nuisance) tokens away from correct class.
    tokens: (B, T, D), class_weights: (C, D), labels: (B,) long.
    """
    B, T, D = tokens.shape
    W = F.normalize(class_weights, dim=1)  # (C, D)
    tokens_norm = F.normalize(tokens, dim=2)  # (B, T, D)
    sim = torch.einsum("btd,cd->btc", tokens_norm, W)  # (B, T, C)
    sim_correct = sim.gather(2, labels.view(B, 1, 1).expand(B, T, 1)).squeeze(2)  # (B, T)
    k_nuis = max(1, int(T * nuisance_frac))
    loss_list: List[torch.Tensor] = []
    for b in range(B):
        s = sim_correct[b]
        _, idx = s.sort()
        nuisance_idx = idx[:k_nuis]
        loss_list.append(F.relu(s[nuisance_idx] - margin).mean())
    return torch.stack(loss_list).mean()


def train_task_model(
    *,
    router: RouterNet,
    experts: Sequence[EmbeddingExpert],
    expert_specs: Sequence[ExpertSpec],
    comm_to_idx: Mapping[str, int],
    classifier: TaskClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    topk: int,
    router_lr: float,
    classifier_lr: float,
    expert_lr: float,
    weight_decay: float,
    router_loss_weight: float,
    load_balance_weight: float,
    gating_noise_std: float,
    gating_noise_epochs: int,
    patience: int = 10,
    eval_interval: int = 1,
    early_delta: float = 0.0,
    selection_metric: str = "val_f1",
    log_magnitude: bool = False,
    locoop_lambda: float = 0.0,
    locoop_margin: float = 0.0,
    locoop_nuisance_frac: float = 0.5,
    num_classes: Optional[int] = None,
    task_loss_type: str = "ce",
    focal_gamma: float = 2.0,
    checkpoint_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, List[Any]]:
    expert_requires_grad = expert_lr > 0
    param_groups: List[Dict[str, object]] = []
    class_weights_tensor: Optional[torch.Tensor] = None
    if task_loss_type == "weighted_ce" and num_classes is not None:
        class_weights_tensor = compute_class_weights_from_loader(train_loader, num_classes)
        class_weights_tensor = class_weights_tensor.to(device=device)

    classifier_params = [p for p in classifier.parameters() if p.requires_grad]
    if classifier_lr > 0 and classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr})

    router_params = [p for p in router.parameters() if p.requires_grad]
    if router_lr > 0 and router_params:
        param_groups.append({"params": router_params, "lr": router_lr})

    expert_params: List[torch.Tensor] = []
    if expert_requires_grad:
        for expert in experts:
            expert_params.extend([p for p in expert.parameters() if p.requires_grad])
        if expert_params:
            param_groups.append({"params": expert_params, "lr": expert_lr})

    if not param_groups:
        raise ValueError(
            "No parameters selected for optimisation. Ensure at least one learning rate is > 0."
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    metric_name = "val_f1" if selection_metric == "val_f1" else "val_loss"
    scheduler_mode = "max" if metric_name == "val_f1" else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=0.5, patience=3, min_lr=1e-6
    )

    scaler = GradScaler(enabled=torch.cuda.is_available())
    group_map = build_group_map(expert_specs, comm_to_idx) if router_loss_weight > 0 else None

    history: Dict[str, List[Any]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "train_balance": [],
        "val_balance": [],
        "train_router_aux": [],
        "val_router_aux": [],
        "train_entropy": [],
        "val_entropy": [],
        "train_usage": [],
        "val_usage": [],
        "gating_noise": [],
    }

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience_counter = 0
    best_router_state: Optional[Dict[str, torch.Tensor]] = None
    best_classifier_state: Optional[Dict[str, torch.Tensor]] = None
    best_expert_states: Optional[List[Dict[str, torch.Tensor]]] = None
    eval_interval = max(1, int(eval_interval))
    early_delta = float(max(0.0, early_delta))

    for epoch in range(1, epochs + 1):
        router.train()
        classifier.train()
        for expert in experts:
            expert.train(expert_requires_grad)

        running_loss = 0.0
        running_balance = 0.0
        running_router_aux = 0.0
        correct = 0
        total = 0
        # Keep usage_sum on CPU to prevent memory accumulation on GPU/HPU
        usage_sum = torch.zeros(len(expert_specs), device='cpu')

        desc = f"Task train {epoch:02d}"
        if tqdm is not None:
            pbar = tqdm(train_loader, desc=desc, leave=False, dynamic_ncols=True)
        else:
            pbar = train_loader

        if gating_noise_std > 0 and gating_noise_epochs > 0:
            decay = max(0.0, 1.0 - (epoch - 1) / float(max(gating_noise_epochs, 1)))
            epoch_noise_std = gating_noise_std * decay
        else:
            epoch_noise_std = gating_noise_std

        for specs, comm_labels, task_labels in pbar:
            specs = specs.to(device, non_blocking=True)
            if log_magnitude:
                specs = torch.log1p(specs.clamp_min(0))
            comm_labels = comm_labels.to(device, non_blocking=True)
            task_labels = task_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            specs_norm = normalize_per_sample_tensor(specs)
            balance_penalty: Optional[torch.Tensor] = None
            router_aux_penalty: Optional[torch.Tensor] = None

            context = autocast(device_type=device.type, enabled=scaler.is_enabled())
            with context:
                router_logits = router(specs_norm)
                if epoch_noise_std > 0:
                    router_logits = router_logits + torch.randn_like(router_logits) * epoch_noise_std
                router_probs = torch.softmax(router_logits, dim=1)
                topk_vals, topk_idx = router_probs.topk(k=topk, dim=1)
                weights = topk_vals / torch.clamp(topk_vals.sum(dim=1, keepdim=True), min=1e-6)

                use_locoop = locoop_lambda > 0
                result = compute_selected_expert_embeddings(
                    experts,
                    specs_norm,
                    topk_idx,
                    allow_grad=expert_requires_grad,
                    return_tokens=use_locoop,
                )
                if use_locoop and isinstance(result, tuple):
                    selected_embeddings, selected_tokens = result
                else:
                    selected_embeddings = result
                    selected_tokens = None
                logits_each = classifier(selected_embeddings.view(-1, selected_embeddings.size(-1)))
                logits_each = logits_each.view(specs.size(0), topk, -1)
                weighted_logits = (weights.unsqueeze(-1) * logits_each).sum(dim=1)
                if task_loss_type == "focal":
                    task_loss = focal_loss(weighted_logits, task_labels, gamma=focal_gamma)
                elif task_loss_type == "weighted_ce" and class_weights_tensor is not None:
                    task_loss = F.cross_entropy(
                        weighted_logits, task_labels, weight=class_weights_tensor
                    )
                else:
                    task_loss = F.cross_entropy(weighted_logits, task_labels)

                loss = task_loss
                if use_locoop and selected_tokens is not None:
                    class_weights = _classifier_class_weights(classifier)
                    if class_weights is not None:
                        class_weights = class_weights.to(device=device, dtype=selected_tokens.dtype)
                        locoop_loss = locoop_nuisance_loss(
                            selected_tokens,
                            class_weights,
                            task_labels,
                            nuisance_frac=locoop_nuisance_frac,
                            margin=locoop_margin,
                        )
                        loss = loss + locoop_lambda * locoop_loss
                if router_loss_weight > 0 and group_map is not None:
                    router_aux_penalty = router_cross_entropy(router_logits, comm_labels, group_map)
                    loss = loss + router_loss_weight * router_aux_penalty
                if load_balance_weight > 0:
                    avg_probs = router_probs.mean(dim=0)
                    uniform = torch.full_like(avg_probs, 1.0 / max(avg_probs.numel(), 1))
                    balance_penalty = F.mse_loss(avg_probs, uniform)
                    loss = loss + load_balance_weight * balance_penalty

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            batch_size = specs.size(0)
            running_loss += loss.detach().item() * batch_size

            if balance_penalty is not None:
                running_balance += balance_penalty.detach().item() * batch_size
            if router_aux_penalty is not None:
                running_router_aux += router_aux_penalty.detach().item() * batch_size

            preds = weighted_logits.argmax(dim=1)
            correct += (preds == task_labels).sum().item()
            total += batch_size
            # Move to CPU to prevent GPU/HPU memory accumulation
            usage_sum = usage_sum + router_probs.detach().sum(dim=0).cpu()

            if tqdm is not None:
                current_loss = running_loss / max(total, 1)
                current_acc = correct / max(total, 1)
                postfix: Dict[str, str] = {
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.3f}",
                }
                if load_balance_weight > 0 and total > 0:
                    postfix["lb"] = f"{(running_balance / total):.4f}"
                pbar.set_postfix(postfix)

        denom = max(total, 1)
        train_loss = running_loss / denom
        train_acc = correct / denom
        train_balance = running_balance / denom
        train_router_aux = running_router_aux / denom
        train_usage_tensor = usage_sum / float(max(total, 1))
        train_usage_tensor = train_usage_tensor.clamp(min=0.0)
        train_entropy = float(
            -(train_usage_tensor * train_usage_tensor.clamp_min(1e-8).log()).sum().item()
        )
        train_usage_list = train_usage_tensor.detach().cpu().tolist()
        should_eval = (epoch % eval_interval == 0) or (epoch == epochs)

        if should_eval:
            val_metrics = evaluate_task_model(
                router=router,
                experts=experts,
                classifier=classifier,
                loader=val_loader,
                topk=topk,
                device=device,
                comm_to_idx=comm_to_idx,
                expert_specs=expert_specs,
                router_loss_weight=router_loss_weight,
                load_balance_weight=load_balance_weight,
            )
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            val_f1 = val_metrics["f1"]
            val_balance = val_metrics["balance"]
            val_router_aux = val_metrics["router_aux"]
            val_usage = val_metrics.get("usage")
            val_entropy = val_metrics.get("entropy", float("nan"))

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler_value = val_f1 if metric_name == "val_f1" else val_loss
            scheduler.step(scheduler_value)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(
                    f"[INFO] Learning rate reduced on {metric_name}: "
                    f"{old_lr:.2e} -> {new_lr:.2e}"
                )
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            val_f1 = float("nan")
            val_balance = float("nan")
            val_router_aux = float("nan")
            val_usage = None
            val_entropy = float("nan")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["train_balance"].append(train_balance)
        history["val_balance"].append(val_balance)
        history["train_router_aux"].append(train_router_aux)
        history["val_router_aux"].append(val_router_aux)
        history["train_entropy"].append(train_entropy)
        history["val_entropy"].append(val_entropy if not math.isnan(val_entropy) else None)
        history["train_usage"].append(train_usage_list)
        history["val_usage"].append(val_usage if val_usage is not None else None)
        history["gating_noise"].append(float(epoch_noise_std))

        msg = [
            f"[Task] Epoch {epoch:02d}:",
            f"train_loss={train_loss:.4f}",
            f"train_acc={train_acc:.3f}",
        ]
        if load_balance_weight > 0:
            msg.append(f"train_lb={train_balance:.4f}")
        if router_loss_weight > 0:
            msg.append(f"train_aux={train_router_aux:.4f}")
        if train_usage_list:
            train_usage_min = min(train_usage_list)
            train_usage_max = max(train_usage_list)
            msg.append(f"train_usage=({train_usage_min:.2f},{train_usage_max:.2f})")
            usage_list_str = ", ".join(f"{usage:.2f}" for usage in train_usage_list)
            msg.append(f"train_usage_all=[{usage_list_str}]")
        msg.append(f"train_H={train_entropy:.3f}")
        if epoch_noise_std > 0:
            msg.append(f"noise={epoch_noise_std:.3f}")

        if should_eval:
            msg.extend(
                [
                    f"val_loss={val_loss:.4f}",
                    f"val_acc={val_acc:.3f}",
                    f"val_f1={val_f1:.3f}",
                ]
            )
            if load_balance_weight > 0:
                msg.append(f"val_lb={val_balance:.4f}")
            if router_loss_weight > 0:
                msg.append(f"val_aux={val_router_aux:.4f}")
            if val_usage:
                val_usage_min = min(val_usage)
                val_usage_max = max(val_usage)
                msg.append(f"val_usage=({val_usage_min:.2f},{val_usage_max:.2f})")
                usage_list_str = ", ".join(f"{usage:.2f}" for usage in val_usage)
                msg.append(f"val_usage_all=[{usage_list_str}]")
            if not math.isnan(val_entropy):
                msg.append(f"val_H={val_entropy:.3f}")
        else:
            msg.append("validation skipped")
        print(" ".join(msg))

        if should_eval:
            if metric_name == "val_f1":
                improved = val_f1 > (best_val_f1 + early_delta)
            else:
                improved = (val_loss + early_delta) < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_val_f1 = val_f1
                patience_counter = 0
                # Free old state before saving new one to prevent memory accumulation
                if best_router_state is not None:
                    del best_router_state
                if best_classifier_state is not None:
                    del best_classifier_state
                if best_expert_states is not None:
                    del best_expert_states
                best_router_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}
                best_classifier_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
                if expert_requires_grad:
                    best_expert_states = [
                        {k: v.cpu().clone() for k, v in expert.state_dict().items()}
                        for expert in experts
                    ]
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"[INFO] Early stopping triggered after {epoch} epochs "
                        f"without {metric_name} improvement"
                    )
                    break

        if checkpoint_callback is not None:
            try:
                checkpoint_callback(epoch)
            except Exception as e:
                print(f"[WARN] Epoch {epoch:02d} checkpoint save failed: {e}")
        
        # Explicit memory cleanup at end of epoch
        if device.type == 'hpu':
            try:
                import habana_frameworks.torch.core as htcore
                htcore.mark_step()
            except (ImportError, AttributeError):
                pass
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_router_state is not None:
        router.load_state_dict({k: v.to(device) for k, v in best_router_state.items()})
    if best_classifier_state is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})
    if best_expert_states is not None:
        for expert, state in zip(experts, best_expert_states):
            expert.load_state_dict({k: v.to(device) for k, v in state.items()})
        print(f"[INFO] Restored best expert checkpoints (val_f1={best_val_f1:.3f})")
    elif best_router_state is not None and best_classifier_state is not None:
        print(f"[INFO] Restored best model with val_f1={best_val_f1:.3f}")

    return history


@torch.no_grad()
def evaluate_task_model(
    *,
    router: RouterNet,
    experts: Sequence[EmbeddingExpert],
    classifier: TaskClassifier,
    loader: DataLoader,
    topk: int,
    device: torch.device,
    comm_to_idx: Mapping[str, int],
    expert_specs: Sequence[ExpertSpec],
    router_loss_weight: float,
    load_balance_weight: float,
) -> Dict[str, Any]:
    """Evaluate task model and return aggregate metrics."""
    router.eval()
    classifier.eval()
    for expert in experts:
        expert.eval()

    group_map = build_group_map(expert_specs, comm_to_idx) if router_loss_weight > 0 else None

    total_loss = 0.0
    total_balance = 0.0
    total_router_aux = 0.0
    total_samples = 0
    usage_sum: Optional[torch.Tensor] = None
    all_preds: List[int] = []
    all_targets: List[int] = []

    for specs, comm_labels, task_labels in iterate_batches(loader, "Task eval"):
        specs = specs.to(device, non_blocking=True)
        comm_labels = comm_labels.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)
        batch_size = specs.size(0)

        specs_norm = normalize_per_sample_tensor(specs)
        router_logits = router(specs_norm)
        router_probs = torch.softmax(router_logits, dim=1)
        topk_vals, topk_idx = router_probs.topk(k=topk, dim=1)
        weights = topk_vals / torch.clamp(topk_vals.sum(dim=1, keepdim=True), min=1e-6)
        if usage_sum is None:
            usage_sum = torch.zeros(router_probs.size(1), device=device)
        usage_sum = usage_sum + router_probs.sum(dim=0)

        selected_embeddings = compute_selected_expert_embeddings(
            experts,
            specs_norm,
            topk_idx,
            allow_grad=False,
        )
        logits_each = classifier(selected_embeddings.view(-1, selected_embeddings.size(-1)))
        logits_each = logits_each.view(batch_size, topk, -1)
        weighted_logits = (weights.unsqueeze(-1) * logits_each).sum(dim=1)

        task_loss = F.cross_entropy(weighted_logits, task_labels)
        loss = task_loss

        if router_loss_weight > 0 and group_map is not None:
            router_aux = router_cross_entropy(router_logits, comm_labels, group_map)
            total_router_aux += router_aux.item() * batch_size
            loss = loss + router_loss_weight * router_aux

        if load_balance_weight > 0:
            avg_probs = router_probs.mean(dim=0)
            uniform = torch.full_like(avg_probs, 1.0 / max(avg_probs.numel(), 1))
            balance_penalty = F.mse_loss(avg_probs, uniform)
            total_balance += balance_penalty.item() * batch_size
            loss = loss + load_balance_weight * balance_penalty

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = weighted_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(task_labels.cpu().tolist())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    acc = float((all_preds_arr == all_targets_arr).mean()) if len(all_preds_arr) > 0 else 0.0

    f1 = 0.0
    if SKLEARN_AVAILABLE and len(all_preds_arr) > 0:
        try:
            f1 = float(f1_score(all_targets_arr, all_preds_arr, average="weighted", zero_division=0))
        except Exception:
            pass

    denom = max(total_samples, 1)
    if usage_sum is not None:
        avg_usage = usage_sum / float(denom)
        avg_usage = avg_usage.clamp(min=0.0)
        entropy = float(-(avg_usage * avg_usage.clamp_min(1e-8).log()).sum().item())
        usage_list: Optional[List[float]] = avg_usage.detach().cpu().tolist()
    else:
        usage_list = None
        entropy = float("nan")
    return {
        "loss": total_loss / denom,
        "acc": acc,
        "f1": f1,
        "balance": (total_balance / denom) if load_balance_weight > 0 else 0.0,
        "router_aux": (total_router_aux / denom) if router_loss_weight > 0 else 0.0,
        "usage": usage_list,
        "entropy": entropy,
    }


def train_oracle_baseline(
    *,
    experts: Sequence[EmbeddingExpert],
    expert_specs: Sequence[ExpertSpec],
    comm_to_idx: Mapping[str, int],
    classifier: TaskClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int = 10,
) -> Dict[str, List[float]]:
    """Train oracle baseline: use ground-truth communication labels to select experts."""
    # Build comm -> expert index mapping (only baseline experts)
    comm_to_expert_idx = build_baseline_expert_map(expert_specs, comm_to_idx)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    best_val_f1 = 0.0
    patience_counter = 0
    best_classifier_state = None
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss = 0.0
        total = 0
        desc = f"Oracle train {epoch:02d}"
        
        for specs, comm_labels, task_labels in iterate_batches(train_loader, desc):
            specs = specs.to(device, non_blocking=True)
            comm_labels = comm_labels.to(device, non_blocking=True)
            task_labels = task_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            context = autocast(device_type=device.type, enabled=scaler.is_enabled())
            with context:
                # Use ground-truth comm labels to select experts
                with torch.no_grad():
                    embeddings = stack_expert_embeddings(experts, specs)
                
                # Select expert based on ground-truth comm label
                batch_size = specs.size(0)
                selected_embeddings = []
                for i in range(batch_size):
                    comm_idx = int(comm_labels[i].item())
                    expert_idx = comm_to_expert_idx[comm_idx]  # Will raise KeyError if missing
                    selected_embeddings.append(embeddings[i, expert_idx])
                selected_embeddings = torch.stack(selected_embeddings, dim=0)
                
                logits = classifier(selected_embeddings)
                loss = F.cross_entropy(logits, task_labels)
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * specs.size(0)
            total += specs.size(0)
        
        train_loss = running_loss / max(total, 1)
        val_loss, val_acc, val_f1 = evaluate_oracle_baseline(
            experts=experts,
            expert_specs=expert_specs,
            comm_to_idx=comm_to_idx,
            comm_to_expert_idx=comm_to_expert_idx,
            classifier=classifier,
            loader=val_loader,
            device=device,
        )
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[INFO] Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        print(
            f"[Oracle] Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_classifier_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered after {epoch} epochs")
                break
    
    if best_classifier_state is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})
        print(f"[INFO] Restored best model with val_f1={best_val_f1:.3f}")
    
    return history


@torch.no_grad()
def evaluate_oracle_baseline(
    *,
    experts: Sequence[EmbeddingExpert],
    expert_specs: Sequence[ExpertSpec],
    comm_to_idx: Mapping[str, int],
    comm_to_expert_idx: Mapping[int, int],
    classifier: TaskClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate oracle baseline."""
    classifier.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    
    for specs, comm_labels, task_labels in iterate_batches(loader, "Oracle eval"):
        specs = specs.to(device, non_blocking=True)
        comm_labels = comm_labels.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)
        
        embeddings = stack_expert_embeddings(experts, specs)
        
        batch_size = specs.size(0)
        selected_embeddings = []
        for i in range(batch_size):
            comm_idx = int(comm_labels[i].item())
            expert_idx = comm_to_expert_idx[comm_idx]  # Will raise KeyError if missing
            selected_embeddings.append(embeddings[i, expert_idx])
        selected_embeddings = torch.stack(selected_embeddings, dim=0)
        
        logits = classifier(selected_embeddings)
        loss = F.cross_entropy(logits, task_labels)
        total_loss += loss.item() * specs.size(0)
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(task_labels.cpu().tolist())
    
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    acc = float((all_preds_arr == all_targets_arr).mean())
    
    f1 = 0.0
    if SKLEARN_AVAILABLE and len(all_preds_arr) > 0:
        try:
            f1 = float(f1_score(all_targets_arr, all_preds_arr, average='weighted', zero_division=0))
        except Exception:
            pass
    
    total = len(all_preds_arr)
    return total_loss / max(total, 1), acc, f1


class SingleModelBackbone(nn.Module):
    """Simple CNN backbone for single model baseline."""
    
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        if dropout > 0:
            self.features.add_module('dropout', nn.Dropout(dropout))
    
    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        x = specs
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected specs rank 3 or 4, got shape {tuple(specs.shape)}")
        return self.features(x)


class ImageNetBackbone(nn.Module):
    """ImageNet pretrained backbone (ResNet18) for baseline."""
    
    def __init__(self, dropout: float = 0.1, freeze_backbone: bool = False) -> None:
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required for ImageNet backbone. Install with: pip install torchvision")
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Convert grayscale to RGB by replicating first conv layer
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize by averaging RGB weights
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in param.parameters():
                    p.requires_grad = False
        
        # Project from 512-d (ResNet18 output) to 128-d
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        if dropout > 0:
            self.projection.add_module('dropout', nn.Dropout(dropout))
    
    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        x = specs
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected specs rank 3 or 4, got shape {tuple(specs.shape)}")
        
        # ResNet forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.projection(x)
        
        return x


def train_single_model(
    *,
    backbone: SingleModelBackbone,
    classifier: TaskClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int = 10,
    eval_interval: int = 1,
) -> Dict[str, List[float]]:
    """Train single model baseline: one model for all communication types."""
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    best_val_f1 = 0.0
    patience_counter = 0
    best_backbone_state = None
    best_classifier_state = None
    eval_interval = max(1, int(eval_interval))
    
    for epoch in range(1, epochs + 1):
        backbone.train()
        classifier.train()
        running_loss = 0.0
        total = 0
        desc = f"Single train {epoch:02d}"
        
        for specs, _, task_labels in iterate_batches(train_loader, desc):
            specs = specs.to(device, non_blocking=True)
            task_labels = task_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Normalize per-sample
            specs_norm = normalize_per_sample_tensor(specs)
            
            context = autocast(device_type=device.type, enabled=scaler.is_enabled())
            with context:
                embeddings = backbone(specs_norm)
                logits = classifier(embeddings)
                loss = F.cross_entropy(logits, task_labels)
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * specs.size(0)
            total += specs.size(0)
        
        train_loss = running_loss / max(total, 1)
        should_eval = (epoch % eval_interval == 0) or (epoch == epochs)
        if should_eval:
            val_loss, val_acc, val_f1 = evaluate_single_model(
                backbone=backbone,
                classifier=classifier,
                loader=val_loader,
                device=device,
            )
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[INFO] Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            val_f1 = float("nan")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        if should_eval:
            print(
                f"[Single] Epoch {epoch:02d}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
            )
        else:
            print(f"[Single] Epoch {epoch:02d}: train_loss={train_loss:.4f} (validation skipped)")
        
        if should_eval:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_backbone_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
                best_classifier_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping triggered after {epoch} epochs")
                    break
    
    if best_backbone_state is not None and best_classifier_state is not None:
        backbone.load_state_dict({k: v.to(device) for k, v in best_backbone_state.items()})
        classifier.load_state_dict({k: v.to(device) for k, v in best_classifier_state.items()})
        print(f"[INFO] Restored best model with val_f1={best_val_f1:.3f}")
    
    return history


@torch.no_grad()
def evaluate_single_model(
    *,
    backbone: SingleModelBackbone,
    classifier: TaskClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate single model baseline."""
    backbone.eval()
    classifier.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    
    for specs, _, task_labels in iterate_batches(loader, "Single eval"):
        specs = specs.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)
        
        specs_norm = normalize_per_sample_tensor(specs)
        embeddings = backbone(specs_norm)
        logits = classifier(embeddings)
        loss = F.cross_entropy(logits, task_labels)
        total_loss += loss.item() * specs.size(0)
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(task_labels.cpu().tolist())
    
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    acc = float((all_preds_arr == all_targets_arr).mean())
    
    f1 = 0.0
    if SKLEARN_AVAILABLE and len(all_preds_arr) > 0:
        try:
            f1 = float(f1_score(all_targets_arr, all_preds_arr, average='weighted', zero_division=0))
        except Exception:
            pass
    
    total = len(all_preds_arr)
    return total_loss / max(total, 1), acc, f1


@torch.no_grad()
def evaluate_test_metrics(
    *,
    router: RouterNet,
    experts: Sequence[EmbeddingExpert],
    classifier: TaskClassifier,
    loader: DataLoader,
    topk: int,
    device: torch.device,
) -> Dict[str, object]:
    router.eval()
    classifier.eval()
    for expert in experts:
        expert.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    coverage = torch.zeros(len(experts), dtype=torch.float64)
    for specs, _, task_labels in iterate_batches(loader, "Task test"):
        specs = specs.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)
        # Normalize once and reuse
        specs_norm = normalize_per_sample_tensor(specs)
        router_probs = torch.softmax(router(specs_norm), dim=1)
        topk_vals, topk_idx = router_probs.topk(k=topk, dim=1)
        weights = topk_vals / torch.clamp(topk_vals.sum(dim=1, keepdim=True), min=1e-6)
        # Run only selected experts with pre-normalized specs
        selected_embeddings = compute_selected_expert_embeddings(
            experts,
            specs_norm,
            topk_idx,
            allow_grad=False,
        )
        logits_each = classifier(selected_embeddings.view(-1, selected_embeddings.size(-1)))
        logits_each = logits_each.view(specs.size(0), topk, -1)
        weighted_logits = (weights.unsqueeze(-1) * logits_each).sum(dim=1)
        preds = weighted_logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(task_labels.detach().cpu().tolist())
        for b in range(specs.size(0)):
            for rank in range(topk):
                coverage[topk_idx[b, rank].item()] += float(weights[b, rank].item())
    all_preds_arr = np.array(all_preds, dtype=np.int64)
    all_targets_arr = np.array(all_targets, dtype=np.int64)
    acc = float((all_preds_arr == all_targets_arr).mean())
    
    # Compute F1 score (weighted)
    f1 = 0.0
    conf = None
    
    if SKLEARN_AVAILABLE and len(all_preds_arr) > 0:
        try:
            from sklearn.metrics import confusion_matrix
            conf = confusion_matrix(all_targets_arr, all_preds_arr).tolist()
            f1 = float(f1_score(all_targets_arr, all_preds_arr, average='weighted', zero_division=0))
        except Exception as e:
            print(f"[WARN] Could not compute sklearn metrics: {e}")
    
    coverage_dict = {
        idx: float(coverage[idx].item()) for idx in range(len(experts))
    }
    return {
        "test_accuracy": acc,
        "test_f1": f1,
        "confusion_matrix": conf,
        "coverage": coverage_dict,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("spectrograms"), help="Root directory with spectrogram data")
    parser.add_argument("--cities", nargs="*", default=["city_1_losangeles"], help="City folders to include")
    parser.add_argument("--comm-types", nargs="*", default=["LTE", "WiFi", "5G"], help="Communication standards to model")
    parser.add_argument("--snrs", nargs="*", default=None, help="SNR folders to include")
    parser.add_argument(
        "--mobilities",
        nargs="*",
        default=["pedestrian", "vehicular"],
        help="Mobility folders to include (default: pedestrian vehicular)",
    )
    parser.add_argument("--modulations", nargs="*", default=None, help="Modulation classes to include (default: all)")
    parser.add_argument("--fft-folders", nargs="*", default=None, help="Specific FFT/window folders to include")
    parser.add_argument("--task", choices=("modulation", "snr_mobility"), default="snr_mobility", help="Downstream task label")
    parser.add_argument("--max-samples-per-comm", type=int, default=0, help="Maximum samples per communication profile (0=use all available data)")
    parser.add_argument("--max-per-combo", type=int, default=0, help="Cap per (modulation,SNR,mobility) combo (0=unbounded, use all available)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of data for training")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--max-samples-per-class", type=int, default=0, help="Maximum training samples per task class (0=no cap)")
    parser.add_argument("--val-samples-per-class", type=int, default=0, help="Validation samples per task class (0=use fraction)")
    parser.add_argument("--test-samples-per-class", type=int, default=0, help="Test samples per task class (0=use remaining)")
    parser.add_argument(
        "--zero-shot-eval",
        action="store_true",
        help="Evaluate without using any training samples. Requires --resume-checkpoint because router/classifier are not trained from scratch in true zero-shot mode.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (optimized for speed and memory)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Accumulate gradients over N steps (effective batch = batch_size * N)")
    parser.add_argument(
        "--router-epochs",
        type=int,
        default=2,
        help="Warm-up epochs for router pre-training (default: 2; set to 0 to skip)",
    )
    parser.add_argument("--task-epochs", type=int, default=25, help="Joint training epochs for classifier and router")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="val_f1",
        choices=("val_f1", "val_loss"),
        help="Validation metric used for checkpoint selection and early stopping (default: val_f1).",
    )
    parser.add_argument(
        "--log-magnitude",
        action="store_true",
        help="Apply log(1+x) to input spectrogram magnitudes before normalization.",
    )
    parser.add_argument("--router-lr", type=float, default=5e-4, help="Learning rate for router during joint training (increased for faster convergence)")
    parser.add_argument("--router-warmup-lr", type=float, default=3e-4, help="Learning rate during router warm-up")
    parser.add_argument("--classifier-lr", type=float, default=2e-3, help="Learning rate for task classifier (increased for faster convergence)")
    parser.add_argument("--expert-lr", type=float, default=5e-5, help="Learning rate for expert fine-tuning (0 keeps experts frozen)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizers")
    parser.add_argument("--router-loss-weight", type=float, default=0.05, help="Weight for communication auxiliary loss")
    parser.add_argument("--load-balance-weight", type=float, default=0.05, help="Weight for expert load-balancing regulariser (0 disables)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for router and classifier heads")
    parser.add_argument("--routing-topk", type=int, default=2, help="Number of experts to keep per sample")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (4 recommended for optimal performance)")
    parser.add_argument("--preload-data", action="store_true", default=True, help="Preload all data into RAM for faster training")
    parser.add_argument("--no-preload-data", dest="preload_data", action="store_false", help="Disable data preloading")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument(
        "--expert",
        action="append",
        default=None,
        help="Optional manual expert definition NAME=COMM:checkpoint[:stats_path]",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("MoE/runs/embedding_router"), help="Directory for outputs")
    parser.add_argument("--save-router", action="store_true", help="Save trained router state_dict")
    parser.add_argument("--save-classifier", action="store_true", help="Save trained classifier state_dict")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume MoE training from an existing checkpoint (router/classifier fine-tuning)",
    )
    parser.add_argument(
        "--resume-router-warmup",
        action="store_true",
        help="When resuming, run the router warm-up stage before joint training",
    )
    parser.add_argument(
        "--baseline",
        choices=["oracle", "single", "imagenet"],
        default=None,
        help="Baseline mode: 'oracle' uses ground-truth comm labels with baseline experts, 'single' trains a single CNN model, 'imagenet' uses pretrained ResNet18",
    )
    parser.add_argument(
        "--gating-noise-std",
        type=float,
        default=0.1,
        help="Stddev of Gaussian noise added to router logits during early training (0 disables)",
    )
    parser.add_argument(
        "--gating-noise-epochs",
        type=int,
        default=5,
        help="Number of epochs over which gating noise decays to zero (0 keeps constant std while enabled)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ImageNet backbone weights (only train projection and classifier)",
    )
    parser.add_argument(
        "--expert-use-prompts",
        action="store_true",
        help="Enable expert prompt tuning (backbone frozen by default)",
    )
    parser.add_argument(
        "--expert-num-prompts",
        type=int,
        default=16,
        help="Prompt length / effective number of prompt tokens (default 16)",
    )
    parser.add_argument(
        "--expert-prompt-style",
        type=str,
        default="deep",
        choices=("deep", "l2p", "soft", "rfprompt"),
        help=(
            "Prompt style: 'deep' (VPT-deep), 'l2p' (pool with instance selection), "
            "'soft' (FiLM-style conditioning), or 'rfprompt' (physics-aware RFPrompt). "
            "Default: deep."
        ),
    )
    parser.add_argument(
        "--expert-pool-size",
        type=int,
        default=10,
        help="L2P prompt pool size (only used when --expert-prompt-style=l2p; default 10)",
    )
    parser.add_argument(
        "--expert-selection-size",
        type=int,
        default=5,
        help="L2P prompts to select per instance (only used when --expert-prompt-style=l2p; default 5)",
    )
    parser.add_argument(
        "--expert-hybrid-prompts",
        action="store_true",
        help="When prompts are enabled, also unfreeze the last 2 backbone layers for hybrid prompt + FT training.",
    )
    parser.add_argument(
        "--locoop-lambda",
        type=float,
        default=0.0,
        help="Weight for LoCoOp nuisance loss (push local tokens away from class anchor). 0 disables.",
    )
    parser.add_argument(
        "--locoop-margin",
        type=float,
        default=0.0,
        help="Margin in LoCoOp nuisance loss (default 0).",
    )
    parser.add_argument(
        "--locoop-nuisance-frac",
        type=float,
        default=0.5,
        help="Fraction of tokens treated as nuisance (bottom by similarity to correct class). Default 0.5.",
    )
    parser.add_argument(
        "--task-loss",
        type=str,
        default="ce",
        choices=("ce", "focal", "weighted_ce"),
        help="Task loss: ce (cross-entropy), focal (focal loss for imbalanced), weighted_ce (inverse-freq class weights). Default: ce.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss (default 2.0). Only used when --task-loss=focal.",
    )
    return parser.parse_args()


def parse_manual_expert(entry: str) -> ExpertSpec:
    if "=" not in entry:
        raise ValueError(f"Expert definition must use NAME=COMM:path syntax (got: {entry})")
    name_part, _, payload = entry.partition("=")
    if ":" not in payload:
        raise ValueError(f"Expert definition missing COMM:path separator (got: {entry})")
    comm_part, _, remainder = payload.partition(":")
    comm = canonical_comm_name(comm_part)
    if ":" in remainder:
        checkpoint_str, stats_str = remainder.split(":", 1)
        stats_path = Path(stats_str).expanduser().resolve()
        if not stats_path.exists():
            print(f"[WARN] Stats file not found: {stats_path}; will use per-sample normalization")
            stats_path = None
    else:
        checkpoint_str = remainder
        # Stats path is optional - we use per-sample normalization
        stats_path_candidate = Path(REPO_ROOT / "models" / f"{comm}_models" / "dataset_stats.json").resolve()
        stats_path = stats_path_candidate if stats_path_candidate.exists() else None
    checkpoint = Path(checkpoint_str).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expert checkpoint does not exist: {checkpoint}")
    return ExpertSpec(name=name_part.strip(), comm=comm, checkpoint=checkpoint, stats_path=stats_path)


def load_experts(
    specs: Sequence[ExpertSpec],
    device: torch.device,
    *,
    trainable: bool = False,
    use_prompts: bool = False,
    num_prompts: int = 16,
    pool_size: int = 10,
    selection_size: int = 5,
    prompt_style: str = "deep",
    prompt_hybrid: bool = False,
) -> List[EmbeddingExpert]:
    embeddings: List[EmbeddingExpert] = []
    for spec in specs:
        desc = f"[INFO] Loading expert '{spec.name}' ({spec.comm}) from {spec.checkpoint}"
        if trainable:
            desc += " [trainable]"
        if use_prompts:
            desc += f" [prompts={prompt_style}]"
        print(desc)
        embeddings.append(
            EmbeddingExpert(
                spec,
                device,
                trainable=trainable,
                use_prompts=use_prompts,
                num_prompts=num_prompts,
                pool_size=pool_size,
                selection_size=selection_size,
                prompt_style=prompt_style,
                prompt_hybrid=prompt_hybrid,
            )
        )
    return embeddings


def build_baseline_expert_map(
    expert_specs: Sequence[ExpertSpec],
    comm_to_idx: Mapping[str, int],
) -> Dict[int, int]:
    """Build mapping from communication type to baseline expert index."""
    comm_to_expert_idx: Dict[int, int] = {}
    for expert_idx, spec in enumerate(expert_specs):
        # Select baseline experts: either has "baseline" in path or not in "task2" folder
        is_baseline = "baseline" in str(spec.checkpoint).lower() or "task2" not in str(spec.checkpoint)
        if is_baseline:
            comm_idx = comm_to_idx[spec.comm]
            if comm_idx not in comm_to_expert_idx:
                comm_to_expert_idx[comm_idx] = expert_idx
    
    # Validate that we have experts for all communication types
    if len(comm_to_expert_idx) != len(comm_to_idx):
        missing_indices = set(comm_to_idx.keys()) - set(comm_to_expert_idx.keys())
        missing_names = [name for name, idx in comm_to_idx.items() if idx in missing_indices]
        raise RuntimeError(f"Missing baseline experts for communication types: {missing_names}")
    
    return comm_to_expert_idx


def sanitize_history_for_serialization(history: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Convert NaN/Inf values in history to None for safe serialization."""

    def _sanitize(value: Any) -> Any:
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [_sanitize(item) for item in value]
        return value

    return {key: [_sanitize(entry) for entry in values] for key, values in history.items()}


def write_training_metrics_csv(
    history: Dict[str, List[Any]],
    expert_specs: Sequence[ExpertSpec],
    csv_path: Path,
) -> None:
    """Write per-epoch training metrics to CSV with expert usage columns."""
    epochs = len(history.get("train_loss", []))
    if epochs == 0:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_f1",
        "train_balance",
        "val_balance",
        "train_router_aux",
        "val_router_aux",
        "train_entropy",
        "val_entropy",
        "gating_noise",
    ]
    usage_headers: List[str] = []
    for spec in expert_specs:
        usage_headers.append(f"train_usage[{spec.name}]")
        usage_headers.append(f"val_usage[{spec.name}]")
    fieldnames.extend(usage_headers)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for epoch_idx in range(epochs):
            row: Dict[str, Any] = {
                "epoch": epoch_idx + 1,
                "train_loss": history["train_loss"][epoch_idx],
                "train_acc": history["train_acc"][epoch_idx],
                "val_loss": history["val_loss"][epoch_idx],
                "val_acc": history["val_acc"][epoch_idx],
                "val_f1": history["val_f1"][epoch_idx],
                "train_balance": history["train_balance"][epoch_idx],
                "val_balance": history["val_balance"][epoch_idx],
                "train_router_aux": history["train_router_aux"][epoch_idx],
                "val_router_aux": history["val_router_aux"][epoch_idx],
                "train_entropy": history["train_entropy"][epoch_idx],
                "val_entropy": history["val_entropy"][epoch_idx],
                "gating_noise": history["gating_noise"][epoch_idx],
            }

            train_usage = history["train_usage"][epoch_idx] or [None] * len(expert_specs)
            if len(train_usage) < len(expert_specs):
                train_usage = list(train_usage) + [None] * (len(expert_specs) - len(train_usage))
            val_usage = history["val_usage"][epoch_idx]
            if val_usage is None:
                val_usage = [None] * len(expert_specs)
            elif len(val_usage) < len(expert_specs):
                val_usage = list(val_usage) + [None] * (len(expert_specs) - len(val_usage))

            for spec, usage_value in zip(expert_specs, train_usage):
                row[f"train_usage[{spec.name}]"] = usage_value
            for spec, usage_value in zip(expert_specs, val_usage):
                row[f"val_usage[{spec.name}]"] = usage_value

            writer.writerow(row)


def save_complete_checkpoint(
    *,
    router: Optional[RouterNet],
    classifier: TaskClassifier,
    expert_models: Optional[Sequence[EmbeddingExpert]],
    expert_specs: Sequence[ExpertSpec],
    comm_to_idx: Mapping[str, int],
    task_type: str,
    num_classes: int,
    topk: int,
    dropout: float,
    mapping: Optional[Dict[int, Tuple[str, str]]],
    output_path: Path,
    model_type: str = "embedding_router_moe",
    backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    backbone_meta: Optional[Dict[str, Any]] = None,
    expert_trainable: bool = False,
) -> None:
    """Save complete MoE checkpoint for inference."""
    checkpoint = {
        "model_type": model_type,
        "task": task_type,
        "num_classes": num_classes,
        "topk": topk,
        "dropout": dropout,
        "comm_to_idx": dict(comm_to_idx),
        "experts": [
            {
                "name": spec.name,
                "comm": spec.comm,
                "checkpoint": str(spec.checkpoint),
                "stats_path": str(spec.stats_path) if spec.stats_path else None,
            }
            for spec in expert_specs
        ],
        "classifier_state_dict": classifier.state_dict(),
        "mapping": {int(k): v for k, v in mapping.items()} if mapping else None,
        "expert_trainable": bool(expert_trainable),
    }
    if router is not None:
        checkpoint["router_state_dict"] = router.state_dict()
    if backbone_state_dict is not None:
        checkpoint["backbone_state_dict"] = backbone_state_dict
    if backbone_meta is not None:
        checkpoint["backbone_meta"] = backbone_meta
    if expert_models is not None:
        def _is_trainable(expert: nn.Module) -> bool:
            # Handle plain EmbeddingExpert or DataParallel-wrapped
            if hasattr(expert, "trainable"):
                return bool(getattr(expert, "trainable"))
            if hasattr(expert, "module") and hasattr(expert.module, "trainable"):  # type: ignore[attr-defined]
                return bool(expert.module.trainable)  # type: ignore[attr-defined]
            return False

        trainable_flags = [_is_trainable(expert) for expert in expert_models]
        if any(trainable_flags):
            checkpoint["expert_state_dicts"] = [
                {
                    "name": spec.name,
                    "state_dict": {k: v.cpu() for k, v in expert.state_dict().items()},
                }
                for spec, expert in zip(expert_specs, expert_models)
            ]
    
    torch.save(checkpoint, output_path)
    print(f"[INFO] Complete checkpoint saved to {output_path}")


def _expert_is_trainable(expert: nn.Module) -> bool:
    """Handle plain EmbeddingExpert and DataParallel-wrapped experts uniformly."""
    if hasattr(expert, "trainable"):
        return bool(getattr(expert, "trainable"))
    module = getattr(expert, "module", None)
    if module is not None and hasattr(module, "trainable"):
        return bool(getattr(module, "trainable"))
    return False


LEGACY_EXPERT_NAME_MAP = {
    "LTE_models": "lteExpert.pth",
    "WiFi_models": "wifiExpert.pth",
    "5G_models": "5gExpert.pth",
}


def _try_legacy_expert_name(path: Path) -> Optional[Path]:
    """Map legacy filenames to the new expert names if present."""
    mapped = LEGACY_EXPERT_NAME_MAP.get(path.parent.name)
    if mapped:
        candidate = path.with_name(mapped)
        if candidate.exists():
            return candidate
    return None


def _resolve_repo_path(path_str: str) -> Path:
    """Resolve paths saved inside checkpoints relative to the repository root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        legacy = _try_legacy_expert_name(path)
        if legacy:
            return legacy
        # Fallback for checkpoints saved with absolute training paths
        repo_name = REPO_ROOT.name
        if repo_name in path.parts:
            try:
                repo_idx = path.parts.index(repo_name)
                candidate = REPO_ROOT.joinpath(*path.parts[repo_idx + 1 :])
                if candidate.exists():
                    return candidate
                legacy_candidate = _try_legacy_expert_name(candidate)
                if legacy_candidate:
                    return legacy_candidate
            except ValueError:
                pass
        return path
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    legacy = _try_legacy_expert_name(candidate)
    if legacy:
        return legacy
    return candidate


def _checkpoint_to_expert_specs(checkpoint: Mapping[str, Any]) -> List[ExpertSpec]:
    specs: List[ExpertSpec] = []
    for expert in checkpoint.get("experts", []):
        checkpoint_path = _resolve_repo_path(expert["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Expert checkpoint referenced in resume file missing: {checkpoint_path}"
            )
        stats_path = None
        stats_path_str = expert.get("stats_path")
        if stats_path_str:
            stats_candidate = _resolve_repo_path(stats_path_str)
            if stats_candidate.exists():
                stats_path = stats_candidate
            else:
                print(
                    f"[WARN] Stats file referenced in checkpoint missing: {stats_candidate}; "
                    "defaulting to per-sample normalization"
                )
        specs.append(
            ExpertSpec(
                name=expert["name"],
                comm=canonical_comm_name(expert["comm"]),
                checkpoint=checkpoint_path,
                stats_path=stats_path,
            )
        )
    return specs


def _normalize_comm_mapping(comm_to_idx_raw: Mapping[str, Any]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for key, idx in comm_to_idx_raw.items():
        normalized[canonical_comm_name(str(key))] = int(idx)
    return normalized


def _normalize_label_mapping(mapping_raw: Optional[Mapping[Any, Any]]) -> Optional[Dict[int, Tuple[str, str]]]:
    if mapping_raw is None:
        return None
    mapping: Dict[int, Tuple[str, str]] = {}
    for key, value in mapping_raw.items():
        idx = int(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            mapping[idx] = (str(value[0]), str(value[1]))
        else:
            raise ValueError(f"Unexpected mapping entry for class {idx}: {value!r}")
    return mapping


def _build_checkpoint_components(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    *,
    train_mode: bool,
) -> Dict[str, Any]:
    dropout = float(checkpoint.get("dropout", 0.1))
    num_classes = int(checkpoint["num_classes"])
    expert_specs = _checkpoint_to_expert_specs(checkpoint)
    expert_trainable_flag = bool(checkpoint.get("expert_trainable", False))
    expert_state_dicts = checkpoint.get("expert_state_dicts")
    name_to_state: Dict[Optional[str], Any] = {}
    if expert_state_dicts:
        name_to_state = {
            entry.get("name"): entry.get("state_dict")
            for entry in expert_state_dicts
            if isinstance(entry, Mapping)
        }

    # Auto-detect prompt wrappers from saved expert state dicts so resume/eval works
    # for prompt-tuned experts even if the checkpoint doesn't store prompt hyperparams.
    use_prompts = False
    prompt_style = "deep"
    num_prompts = 16
    pool_size = 10
    selection_size = 4
    if name_to_state:
        # If any expert has an L2P pool, enable L2P wrapper for all experts.
        for _name, _state in name_to_state.items():
            if not isinstance(_state, Mapping):
                continue
            if "model.pool.prompts" in _state and "model.pool.keys" in _state:
                use_prompts = True
                prompt_style = "l2p"
                try:
                    p = _state["model.pool.prompts"]
                    pool_size = int(p.shape[0])
                    prompt_len = int(p.shape[1])
                    # Our L2P uses prompt_len derived from num_prompts; invert approximately.
                    num_prompts = max(16, prompt_len * 4)
                    selection_size = min(selection_size, pool_size)
                except Exception:
                    pass
                break

    experts = load_experts(
        expert_specs,
        device,
        trainable=train_mode and expert_trainable_flag,
        use_prompts=use_prompts,
        num_prompts=num_prompts,
        pool_size=pool_size,
        selection_size=selection_size,
        prompt_style=prompt_style,
    )

    if name_to_state:
        for spec, expert in zip(expert_specs, experts):
            state_dict = name_to_state.get(spec.name)
            if state_dict:
                expert.load_state_dict({k: v for k, v in state_dict.items()})
    if not train_mode:
        for expert_model in experts:
            expert_model.eval()

    router: Optional[RouterNet] = None
    router_state = checkpoint.get("router_state_dict")
    if router_state is not None:
        router = RouterNet(num_experts=len(expert_specs), dropout=dropout).to(device)
        router.load_state_dict(router_state)
        if train_mode:
            router.train()
        else:
            router.eval()

    classifier = TaskClassifier(num_classes=num_classes, dropout=dropout).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    if train_mode:
        classifier.train()
    else:
        classifier.eval()

    return {
        "router": router,
        "classifier": classifier,
        "experts": experts,
        "expert_specs": expert_specs,
        "comm_to_idx": _normalize_comm_mapping(checkpoint["comm_to_idx"]),
        "task": checkpoint["task"],
        "topk": int(checkpoint.get("topk", 1)),
        "num_classes": num_classes,
        "dropout": dropout,
        "mapping": _normalize_label_mapping(checkpoint.get("mapping")),
        "expert_trainable": expert_trainable_flag,
    }


def load_checkpoint_for_training(
    checkpoint_path: Path,
    device: torch.device,
    checkpoint_data: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load checkpoint components for continued training."""
    if checkpoint_data is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    else:
        checkpoint = dict(checkpoint_data)
    model_type = str(checkpoint.get("model_type", ""))
    if model_type != "embedding_router_moe":
        raise ValueError(
            f"Resume checkpoint expected 'embedding_router_moe' but found '{model_type}'"
        )
    return _build_checkpoint_components(checkpoint, device, train_mode=True)


def load_checkpoint_for_inference(checkpoint_path: Path, device: torch.device):
    """Load complete checkpoint and return all components."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return _build_checkpoint_components(checkpoint, device, train_mode=False)


class MoEPredictor:
    """Inference wrapper for trained MoE model."""
    
    def __init__(
        self,
        *,
        router: Optional[RouterNet],
        classifier: TaskClassifier,
        experts: Sequence[EmbeddingExpert],
        expert_specs: Sequence[ExpertSpec],
        comm_to_idx: Mapping[str, int],
        task_type: str,
        topk: int,
        mapping: Optional[Dict[int, Tuple[str, str]]],
        device: torch.device,
    ) -> None:
        self.router = router
        self.classifier = classifier
        self.experts = experts
        self.expert_specs = expert_specs
        self.comm_to_idx = comm_to_idx
        self.task_type = task_type
        self.topk = topk
        self.mapping = mapping
        self.device = device
        
        # Build reverse mapping for results
        if task_type == "modulation":
            from task1.train_mcs_models import MODULATION_LABELS
            self.idx_to_label = {v: k for k, v in MODULATION_LABELS.items()}
        elif mapping:
            self.idx_to_label = mapping
        else:
            self.idx_to_label = None
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, device: Optional[torch.device] = None):
        """Load predictor from checkpoint file."""
        if device is None:
            # Try HPU first, then CUDA, then CPU
            try:
                import habana_frameworks.torch.core as htcore
                device = torch.device("hpu")
            except (ImportError, RuntimeError):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        components = load_checkpoint_for_inference(checkpoint_path, device)
        
        return cls(
            router=components["router"],
            classifier=components["classifier"],
            experts=components["experts"],
            expert_specs=components["expert_specs"],
            comm_to_idx=components["comm_to_idx"],
            task_type=components["task"],
            topk=components["topk"],
            mapping=components["mapping"],
            device=device,
        )
    
    @torch.no_grad()
    def predict(
        self,
        spectrogram: torch.Tensor,
        return_probabilities: bool = False,
        return_routing: bool = False,
    ) -> Dict[str, object]:
        """Predict task label for a single spectrogram or batch.
        
        Args:
            spectrogram: Tensor of shape [H, W] or [B, H, W]
            return_probabilities: If True, return class probabilities
            return_routing: If True, return routing weights
        
        Returns:
            Dictionary with prediction results
        """
        # Handle single sample
        single_sample = spectrogram.dim() == 2
        if single_sample:
            spectrogram = spectrogram.unsqueeze(0)
        
        # Move to device and normalize
        specs = spectrogram.to(self.device)
        specs_norm = normalize_per_sample_tensor(specs)
        
        if self.router is not None:
            # Router-based prediction
            router_logits = self.router(specs_norm)
            router_probs = torch.softmax(router_logits, dim=1)
            topk_vals, topk_idx = router_probs.topk(k=self.topk, dim=1)
            weights = topk_vals / torch.clamp(topk_vals.sum(dim=1, keepdim=True), min=1e-6)
            
            # Get embeddings from selected experts
            selected_embeddings = compute_selected_expert_embeddings(
                self.experts,
                specs_norm,
                topk_idx,
                allow_grad=False,
            )
            logits_each = self.classifier(selected_embeddings.view(-1, selected_embeddings.size(-1)))
            logits_each = logits_each.view(specs.size(0), self.topk, -1)
            weighted_logits = (weights.unsqueeze(-1) * logits_each).sum(dim=1)
        else:
            # Oracle baseline: use all experts (fallback)
            embeddings = stack_expert_embeddings(self.experts, specs)
            # Average all expert embeddings
            avg_embedding = embeddings.mean(dim=1)
            weighted_logits = self.classifier(avg_embedding)
            topk_idx = None
            weights = None
        
        probs = torch.softmax(weighted_logits, dim=1)
        predicted_classes = weighted_logits.argmax(dim=1)
        
        # Build results
        results = {
            "predicted_class": int(predicted_classes[0].item()) if single_sample else predicted_classes.cpu().tolist(),
            "confidence": float(probs[0, predicted_classes[0]].item()) if single_sample else [float(probs[i, predicted_classes[i]].item()) for i in range(len(predicted_classes))],
        }
        
        # Add human-readable labels
        if self.idx_to_label:
            if single_sample:
                results["label"] = self.idx_to_label.get(int(predicted_classes[0].item()), "Unknown")
            else:
                results["labels"] = [self.idx_to_label.get(int(c), "Unknown") for c in predicted_classes.cpu().tolist()]
        
        if return_probabilities:
            results["probabilities"] = probs[0].cpu().tolist() if single_sample else probs.cpu().tolist()
        
        if return_routing and topk_idx is not None:
            routing_info = []
            for b in range(specs.size(0)):
                sample_routing = []
                for k in range(self.topk):
                    expert_idx = int(topk_idx[b, k].item())
                    sample_routing.append({
                        "expert": self.expert_specs[expert_idx].name,
                        "comm": self.expert_specs[expert_idx].comm,
                        "weight": float(weights[b, k].item()),
                    })
                routing_info.append(sample_routing)
            results["routing"] = routing_info[0] if single_sample else routing_info
        
        return results


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    resume_checkpoint_path: Optional[Path] = None
    resume_checkpoint_data: Optional[Dict[str, Any]] = None
    resume_comm_to_idx: Optional[Dict[str, int]] = None
    resume_topk: Optional[int] = None
    resume_dropout: Optional[float] = None
    resume_task: Optional[str] = None
    resume_num_classes: Optional[int] = None
    resume_mapping: Optional[Dict[int, Tuple[str, str]]] = None

    if args.resume_checkpoint is not None:
        if args.baseline is not None:
            raise ValueError("--resume-checkpoint is only supported for MoE training (omit --baseline).")
        if args.expert:
            raise ValueError("--resume-checkpoint cannot be combined with manual --expert definitions.")
        resume_checkpoint_path = args.resume_checkpoint.expanduser().resolve()
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_checkpoint_path}")
        resume_checkpoint_data = torch.load(resume_checkpoint_path, map_location="cpu")
        model_type = str(resume_checkpoint_data.get("model_type", ""))
        if model_type != "embedding_router_moe":
            raise ValueError(
                f"Cannot resume from checkpoint with model_type={model_type!r}; expected 'embedding_router_moe'"
            )
        resume_comm_to_idx = _normalize_comm_mapping(resume_checkpoint_data["comm_to_idx"])
        resume_topk = int(resume_checkpoint_data.get("topk", 1))
        resume_dropout = float(resume_checkpoint_data.get("dropout", args.dropout))
        resume_task = str(resume_checkpoint_data.get("task", args.task))
        resume_num_classes = int(resume_checkpoint_data["num_classes"])
        resume_mapping = _normalize_label_mapping(resume_checkpoint_data.get("mapping"))
        expert_specs = _checkpoint_to_expert_specs(resume_checkpoint_data)
        comm_types = [comm for comm, _ in sorted(resume_comm_to_idx.items(), key=lambda kv: kv[1])]
        requested_comm_types = [canonical_comm_name(comm) for comm in args.comm_types]
        if set(requested_comm_types) != set(comm_types):
            raise ValueError(
                "--comm-types must match the communications present in the resume checkpoint "
                f"{comm_types}; received {requested_comm_types}"
            )
        if args.task != resume_task:
            print(f"[WARN] Overriding task '{args.task}' -> '{resume_task}' to match resume checkpoint")
        args.task = resume_task
        if abs(args.dropout - resume_dropout) > 1e-6:
            print(f"[WARN] Overriding dropout {args.dropout} -> {resume_dropout} to match resume checkpoint")
        args.dropout = resume_dropout
        args.routing_topk = resume_topk
    else:
        if args.expert:
            expert_specs = [parse_manual_expert(entry) for entry in args.expert]
        else:
            expert_specs = discover_default_experts()
        comm_types = [canonical_comm_name(comm) for comm in args.comm_types]
        for spec in expert_specs:
            if spec.comm not in comm_types:
                comm_types.append(spec.comm)
        comm_types = list(dict.fromkeys(comm_types))

    dataset, comm_to_idx, mapping = prepare_dataset(
        data_root=args.data_root,
        cities=args.cities,
        comm_types=comm_types,
        snrs=args.snrs,
        mobilities=args.mobilities,
        modulations=args.modulations,
        fft_folders=args.fft_folders,
        max_samples_per_comm=args.max_samples_per_comm,
        max_per_combo=args.max_per_combo,
        max_samples_per_class=args.max_samples_per_class,
        val_samples_per_class=args.val_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        task=args.task,
        seed=args.seed,
        preload=args.preload_data,
    )

    available_comms = set(comm_to_idx.keys())
    filtered_specs: List[ExpertSpec] = []
    missing_specs: List[ExpertSpec] = []
    for spec in expert_specs:
        if spec.comm in available_comms:
            filtered_specs.append(spec)
        else:
            missing_specs.append(spec)
    if missing_specs:
        missing_comm = ", ".join(sorted({spec.comm for spec in missing_specs}))
        if resume_checkpoint_path is not None:
            raise RuntimeError(
                "Resume dataset is missing communication types required by the checkpoint: "
                f"{missing_comm}"
            )
        missing_names = ", ".join(sorted({f"{spec.name} ({spec.comm})" for spec in missing_specs}))
        print(f"[WARN] Skipping experts with no matching data: {missing_names}")
    expert_specs = filtered_specs
    if not expert_specs:
        raise RuntimeError("No experts remain after filtering by available communication types")

    if resume_comm_to_idx is not None and comm_to_idx != resume_comm_to_idx:
        raise RuntimeError(
            "Communication mapping inferred from data does not match resume checkpoint. "
            "Ensure the same communication types are present."
        )

    dataset_num_classes = int(dataset.task_labels.max()) + 1
    if resume_num_classes is not None and dataset_num_classes != resume_num_classes:
        raise RuntimeError(
            f"Dataset provides {dataset_num_classes} task classes but resume checkpoint expects {resume_num_classes}. "
            "Ensure the limited dataset still covers every class."
        )
    if resume_mapping is not None:
        if mapping is None:
            raise RuntimeError("Resume checkpoint includes task mapping but dataset could not infer one.")
        if mapping != resume_mapping:
            raise RuntimeError(
                "Task label mapping from limited dataset does not match resume checkpoint. "
                "Ensure all (SNR, mobility) combinations are present."
            )

    if args.zero_shot_eval and resume_checkpoint_path is None:
        raise ValueError(
            "True zero-shot evaluation requires --resume-checkpoint. "
            "The previous K=0 behavior meant 'use all data'; use that without --zero-shot-eval if you want the full-data regime."
        )
    if args.zero_shot_eval:
        if args.router_epochs > 0 or args.task_epochs > 0:
            print("[INFO] zero_shot_eval enabled: skipping router warm-up and task training.")
        args.router_epochs = 0
        args.task_epochs = 0

    train_idx, val_idx, test_idx = stratified_split(
        dataset.task_labels.numpy(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_train_per_class=args.max_samples_per_class,
        val_samples_per_class=args.val_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        zero_shot_eval=args.zero_shot_eval,
        seed=args.seed,
    )

    print(f"[SPLIT] Train: {len(train_idx):,} samples ({len(train_idx)/len(dataset)*100:.1f}%)")
    print(f"[SPLIT] Val:   {len(val_idx):,} samples ({len(val_idx)/len(dataset)*100:.1f}%)")
    print(f"[SPLIT] Test:  {len(test_idx):,} samples ({len(test_idx)/len(dataset)*100:.1f}%)\n")

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # Device selection with HPU support
    # Try HPU first, then CUDA, then CPU
    try:
        import habana_frameworks.torch.core as htcore
        device = torch.device("hpu")
        print("[INFO] HPU device detected and selected")
    except (ImportError, RuntimeError):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            num_gpus = torch.cuda.device_count()
            print(f"[INFO] CUDA device detected - using {num_gpus} GPU(s)")
        else:
            device = torch.device("cpu")
            num_gpus = 0
            print("[INFO] Using CPU device")
    num_classes = dataset_num_classes

    resume_state: Optional[Dict[str, Any]] = None
    if resume_checkpoint_path is not None:
        resume_state = load_checkpoint_for_training(
            resume_checkpoint_path,
            device,
            checkpoint_data=resume_checkpoint_data,
        )
        if resume_state["comm_to_idx"] != comm_to_idx:
            raise RuntimeError("Mismatch between resume checkpoint comm_to_idx and dataset mapping")
        args.routing_topk = resume_state["topk"]
        args.dropout = resume_state["dropout"]
        expert_specs = resume_state["expert_specs"]
        print(f"[INFO] Resuming MoE training from {resume_checkpoint_path}")
        if resume_state.get("expert_trainable") and args.expert_lr <= 0:
            print(
                "[WARN] Resume checkpoint includes fine-tuned experts but --expert-lr is 0; "
                "experts will remain frozen unless you provide a positive --expert-lr."
            )
    embedding_models: Optional[List[EmbeddingExpert]] = None
    training_history: Optional[Dict[str, Any]] = None

    effective_topk = max(1, min(args.routing_topk, len(expert_specs)))
    model_type = "embedding_router_moe"
    router: Optional[RouterNet] = None
    classifier: TaskClassifier
    backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None
    backbone_meta: Optional[Dict[str, Any]] = None
    use_data_parallel = torch.cuda.is_available() and torch.cuda.device_count() > 1

    if args.baseline == "single":
        print("[INFO] Training single model baseline...")
        backbone = SingleModelBackbone(dropout=args.dropout).to(device)
        classifier = TaskClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
        train_single_model(
            backbone=backbone,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.task_epochs,
            lr=args.classifier_lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        print("[INFO] Evaluating on test split...")
        test_loss, test_acc, test_f1 = evaluate_single_model(
            backbone=backbone,
            classifier=classifier,
            loader=test_loader,
            device=device,
        )
        test_metrics = {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss,
        }
        model_type = "baseline_single"
        backbone_state_dict = {k: v.cpu() for k, v in backbone.state_dict().items()}
        backbone_meta = {
            "baseline_mode": "single",
            "backbone_class": backbone.__class__.__name__,
        }
    elif args.baseline == "imagenet":
        print("[INFO] Training ImageNet pretrained baseline...")
        backbone = ImageNetBackbone(dropout=args.dropout, freeze_backbone=args.freeze_backbone).to(device)
        classifier = TaskClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
        train_single_model(
            backbone=backbone,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.task_epochs,
            lr=args.classifier_lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        print("[INFO] Evaluating on test split...")
        test_loss, test_acc, test_f1 = evaluate_single_model(
            backbone=backbone,
            classifier=classifier,
            loader=test_loader,
            device=device,
        )
        test_metrics = {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss,
        }
        model_type = "baseline_imagenet"
        backbone_state_dict = {k: v.cpu() for k, v in backbone.state_dict().items()}
        backbone_meta = {
            "baseline_mode": "imagenet",
            "backbone_class": backbone.__class__.__name__,
            "freeze_backbone": args.freeze_backbone,
        }
    elif args.baseline == "oracle":
        print("[INFO] Training oracle baseline (ground-truth comm labels)...")
        embedding_models = load_experts(expert_specs, device)
        classifier = TaskClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
        train_oracle_baseline(
            experts=embedding_models,
            expert_specs=expert_specs,
            comm_to_idx=comm_to_idx,
            classifier=classifier,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.task_epochs,
            lr=args.classifier_lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        print("[INFO] Evaluating on test split...")
        comm_to_expert_idx = build_baseline_expert_map(expert_specs, comm_to_idx)
        test_loss, test_acc, test_f1 = evaluate_oracle_baseline(
            experts=embedding_models,
            expert_specs=expert_specs,
            comm_to_idx=comm_to_idx,
            comm_to_expert_idx=comm_to_expert_idx,
            classifier=classifier,
            loader=test_loader,
            device=device,
        )
        test_metrics = {
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_loss": test_loss,
        }
        model_type = "baseline_oracle"
    else:
        expert_trainable = args.expert_lr > 0
        use_prompts = getattr(args, "expert_use_prompts", False)  # from --expert-use-prompts
        prompt_style = getattr(args, "expert_prompt_style", "deep")
        if resume_state is None:
            router = RouterNet(num_experts=len(expert_specs), dropout=args.dropout).to(device)
            if use_data_parallel:
                print(f"[INFO] Wrapping router with DataParallel")
                router = nn.DataParallel(router)
            if args.router_epochs > 0:
                print("[INFO] Pre-training router...")
                train_router(
                    router,
                    experts=expert_specs,
                    comm_to_idx=comm_to_idx,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=args.router_epochs,
                    lr=args.router_warmup_lr,
                    weight_decay=args.weight_decay,
                )
            else:
                print("[INFO] Skipping router warm-up (router_epochs=0).")
            embedding_models = load_experts(
                expert_specs,
                device,
                trainable=expert_trainable,
                use_prompts=use_prompts,
                num_prompts=getattr(args, "expert_num_prompts", 16),
                pool_size=getattr(args, "expert_pool_size", 10),
                selection_size=getattr(args, "expert_selection_size", 5),
                prompt_style=prompt_style,
                prompt_hybrid=getattr(args, "expert_hybrid_prompts", False),
            )
            if use_data_parallel:
                print(f"[INFO] Wrapping {len(embedding_models)} experts with DataParallel")
                embedding_models = [nn.DataParallel(m) if hasattr(m, 'forward') else m for m in embedding_models]
            classifier = TaskClassifier(num_classes=num_classes, dropout=args.dropout).to(device)
            if use_data_parallel:
                print(f"[INFO] Wrapping classifier with DataParallel")
                classifier = nn.DataParallel(classifier)
        else:
            router = resume_state["router"]
            classifier = resume_state["classifier"]
            embedding_models = resume_state["experts"]
            for expert in embedding_models:
                expert.set_trainable(expert_trainable)
            if args.resume_router_warmup and args.router_epochs > 0:
                print("[INFO] Running router warm-up on resume data...")
                train_router(
                    router,
                    experts=expert_specs,
                    comm_to_idx=comm_to_idx,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    epochs=args.router_epochs,
                    lr=args.router_warmup_lr,
                    weight_decay=args.weight_decay,
                )
            else:
                print("[INFO] Skipping router warm-up (resume).")
            # Ensure experts are moved to correct device/state after resume
            for expert in embedding_models:
                expert.to(device)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch_checkpoint_dir = output_dir / "epoch_checkpoints"
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def save_epoch_checkpoint(epoch_idx: int) -> None:
            checkpoint_path = epoch_checkpoint_dir / f"epoch_{epoch_idx:02d}.pth"
            expert_models_to_save = (
                embedding_models if (embedding_models is not None and args.baseline is None) else None
            )
            expert_trainable_flag = (
                bool(expert_models_to_save)
                and any(_expert_is_trainable(expert) for expert in expert_models_to_save)
            )
            save_complete_checkpoint(
                router=router if args.baseline is None else None,
                classifier=classifier,
                expert_models=expert_models_to_save,
                expert_specs=expert_specs,
                comm_to_idx=comm_to_idx,
                task_type=args.task,
                num_classes=num_classes,
                topk=effective_topk if args.baseline is None else 1,
                dropout=args.dropout,
                mapping=mapping,
                output_path=checkpoint_path,
                model_type=model_type,
                backbone_state_dict=backbone_state_dict if args.baseline in {"single", "imagenet"} else None,
                backbone_meta=backbone_meta if args.baseline in {"single", "imagenet"} else None,
                expert_trainable=expert_trainable_flag,
            )

        if args.task_epochs > 0:
            print("[INFO] Training classifier with router-guided embeddings...")
            training_history = train_task_model(
                router=router,
                experts=embedding_models,
                expert_specs=expert_specs,
                comm_to_idx=comm_to_idx,
                classifier=classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.task_epochs,
                topk=effective_topk,
                router_lr=args.router_lr,
                classifier_lr=args.classifier_lr,
                expert_lr=max(0.0, args.expert_lr),
                weight_decay=args.weight_decay,
                router_loss_weight=max(0.0, args.router_loss_weight),
                load_balance_weight=max(0.0, args.load_balance_weight),
                gating_noise_std=max(0.0, args.gating_noise_std),
                gating_noise_epochs=max(0, args.gating_noise_epochs),
                patience=args.patience,
                selection_metric=args.selection_metric,
                log_magnitude=args.log_magnitude,
                locoop_lambda=max(0.0, args.locoop_lambda),
                locoop_margin=args.locoop_margin,
                locoop_nuisance_frac=args.locoop_nuisance_frac,
                num_classes=num_classes,
                task_loss_type=args.task_loss,
                focal_gamma=args.focal_gamma,
                checkpoint_callback=save_epoch_checkpoint,
            )
        else:
            print("[INFO] Skipping task training (task_epochs=0).")
            training_history = None
        print("[INFO] Evaluating on test split...")
        test_metrics = evaluate_test_metrics(
            router=router,
            experts=embedding_models,
            classifier=classifier,
            loader=test_loader,
            topk=effective_topk,
            device=device,
        )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"F1 Score: {test_metrics['test_f1']:.4f}")
    print("=" * 60 + "\n")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(test_metrics, fh, indent=2)
    print("[RESULT] Test metrics saved to", metrics_path)

    if training_history is not None:
        sanitized_history = sanitize_history_for_serialization(training_history)
        history_path = output_dir / "training_history.json"
        with history_path.open("w", encoding="utf-8") as fh:
            json.dump(sanitized_history, fh, indent=2)
        metrics_csv_path = output_dir / "training_metrics.csv"
        write_training_metrics_csv(sanitized_history, expert_specs, metrics_csv_path)
        print("[RESULT] Training history saved to", history_path)
        print("[RESULT] Training metrics saved to", metrics_csv_path)

    checkpoint_path = output_dir / "moe_checkpoint.pth"
    expert_models_to_save = (
        embedding_models if (embedding_models is not None and args.baseline is None) else None
    )
    expert_trainable_flag = (
        bool(expert_models_to_save)
        and any(_expert_is_trainable(expert) for expert in expert_models_to_save)
    )
    save_complete_checkpoint(
        router=router if args.baseline is None else None,
        classifier=classifier,
        expert_models=expert_models_to_save,
        expert_specs=expert_specs,
        comm_to_idx=comm_to_idx,
        task_type=args.task,
        num_classes=num_classes,
        topk=effective_topk if args.baseline is None else 1,
        dropout=args.dropout,
        mapping=mapping,
        output_path=checkpoint_path,
        model_type=model_type,
        backbone_state_dict=backbone_state_dict if args.baseline in {"single", "imagenet"} else None,
        backbone_meta=backbone_meta if args.baseline in {"single", "imagenet"} else None,
        expert_trainable=expert_trainable_flag,
    )

    if args.baseline is None and args.save_router:
        torch.save(router.state_dict(), output_dir / "router_state_dict.pth")
        print("[INFO] Router state_dict saved")
    if args.save_classifier:
        torch.save(classifier.state_dict(), output_dir / "classifier_state_dict.pth")
        print("[INFO] Classifier state_dict saved")
    if mapping is not None:
        with (output_dir / "snr_mobility_mapping.json").open("w", encoding="utf-8") as fh:
            json.dump({int(k): v for k, v in mapping.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
