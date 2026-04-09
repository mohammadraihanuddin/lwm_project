#!/usr/bin/env python3
"""Train a router that selects top-k LWM backbones and feeds a shared classifier.

This variant differs from ``train_top1_router.py`` by:
  * loading each expert checkpoint only for its backbone (classifier discarded)
  * extracting 128-d embeddings with the standard mean-pooled LWM features
  * selecting the top-k experts per sample (default k=2) via a router network
  * feeding each selected embedding through a shared Res1DCNN head
  * weighting the per-embedding logits using the router probabilities

This updated version also adds RFPrompt CLI passthrough and checkpoint metadata:
  * ``--expert-rfprompt-global``
  * ``--expert-rfprompt-spectral``
  * ``--expert-rfprompt-temporal``
  * ``--expert-rfprompt-condition``
  * ``--expert-rfprompt-use-router``
  * ``--expert-rfprompt-pool-prompts``

Important: this file now PARSES and PASSES the RFPrompt configuration, but
``task2/mobility_utils.py::prepare_model(...)`` must also accept these kwargs and
construct the actual RFPrompt wrapper.
"""

from __future__ import annotations

import argparse
import csv
import inspect
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
from MoE.train_embedding_router_old import (  # type: ignore
    _expert_is_trainable,
    sanitize_history_for_serialization,
    write_training_metrics_csv,
    train_router,
    evaluate_router,
    stack_expert_embeddings,
    compute_selected_expert_embeddings,
    gather_topk_embeddings,
    _classifier_class_weights,
    focal_loss,
    compute_class_weights_from_indices,
    compute_class_weights_from_loader,
    locoop_nuisance_loss,
    train_task_model,
    evaluate_task_model,
    train_oracle_baseline,
    evaluate_oracle_baseline,
    SingleModelBackbone,
    ImageNetBackbone,
    train_single_model,
    evaluate_single_model,
    evaluate_test_metrics,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

COMM_CANONICAL = {"lte": "LTE", "wifi": "WiFi", "5g": "5G"}
ExpertIndex = int


@dataclass(slots=True)
class SampleEntry:
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
    comm_upper = canonical_comm_name(comm)
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

    preferred_names = {"LTE": "lteExpert.pth", "WiFi": "wifiExpert.pth", "5G": "5gExpert.pth"}
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
    comm_upper = canonical_comm_name(comm)
    possible_locations = [
        REPO_ROOT / "models" / "experts" / "task2" / f"{comm_upper}_models",
        REPO_ROOT / "task2" / "mobility_benchmark" / comm.lower(),
    ]
    for base_dir in possible_locations:
        if not base_dir.exists():
            continue
        if "_models" in base_dir.name:
            candidates = sorted(base_dir.glob("*.pth"))
            if candidates:
                return candidates[-1]
        run_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
        if run_dirs:
            for run_dir in reversed(run_dirs):
                epoch_dir = run_dir / "epoch_checkpoints"
                if epoch_dir.exists():
                    epochs = sorted(epoch_dir.glob("epoch_*.pth"))
                    if epochs:
                        return epochs[-1]
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
                        f"[WARN] {comm}: per-combo limit ({per_combo_limit}) is below requested total ({target}); consider relaxing --max-per-combo or per-class caps."
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
            num_samples = get_sample_count_fast(path_str)
        except Exception as exc:
            print(f"[WARN] Failed to load {path_str}: {exc}")
            continue
        if num_samples == 0:
            continue

        remaining_for_combo = (per_combo_limit - already if per_combo_limit is not None else num_samples)
        allowed = min(num_samples, remaining_for_combo)
        if remaining is not None:
            allowed = min(allowed, remaining)
        if allowed <= 0:
            continue

        if allowed == num_samples:
            chosen_indices = np.arange(num_samples)
        else:
            chosen_indices = rng.choice(num_samples, size=allowed, replace=False)

        entry_meta = SampleMetadata(
            path=path,
            comm=meta.comm,
            modulation=meta.modulation,
            snr=meta.snr,
            mobility=meta.mobility,
            rate=meta.rate,
            source=path_str,
        )
        batch_entries = [
            SampleEntry(path=path, index=int(idx), metadata=entry_meta)
            for idx in chosen_indices.tolist()
        ]
        entries.extend(batch_entries)

        combo_counts[combo_key] += int(len(chosen_indices))
        if remaining is not None:
            remaining -= int(len(chosen_indices))
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
        unmet = [combo for combo, target in combo_targets.items() if combo_counts[combo] < target]
        if unmet:
            print(f"[WARN] {comm}: target not met for {len(unmet)} combo(s); consider lowering per-class requirements.", flush=True)
    print(f"[DATA] {comm}: gathered {len(entries)} samples after scanning {files_processed} files", flush=True)
    return entries


def iterate_batches(loader: Iterable, desc: str, *, log_every: Optional[int] = None) -> Iterable:
    if tqdm is not None:
        for item in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            yield item
        return
    try:
        total = len(loader)  # type: ignore[arg-type]
    except Exception:
        total = None
    if log_every is None:
        log_every = max(1, total // 10) if total else 50
    for idx, batch in enumerate(loader, start=1):
        if idx == 1 or idx % log_every == 0 or (total is not None and idx == total):
            if total is not None:
                print(f"[Progress] {desc}: {idx}/{total}", flush=True)
            else:
                print(f"[Progress] {desc}: batch {idx}", flush=True)
        yield batch


def _get_cache_capacity(default: int = 32) -> int:
    override = os.environ.get("LWM_FILE_CACHE_SIZE")
    if not override:
        return default
    try:
        return max(1, int(override))
    except ValueError:
        print(f"[WARN] Ignoring invalid LWM_FILE_CACHE_SIZE={override!r}; using default {default}", flush=True)
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
        print(f"[WARN] Unknown LWM_PRELOAD_DTYPE={override!r}; defaulting to {default}", flush=True)
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
                        return int(float(parts[1]) * 1024)
    except Exception:
        pass
    return None


@lru_cache(maxsize=_FILE_CACHE_SIZE)
def _load_file_spectrograms(path_str: str) -> np.ndarray:
    arr = load_all_samples(path_str)
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16, copy=False)
    return arr


def get_sample_count_fast(path: str) -> int:
    try:
        arr = load_all_samples(path)
        return int(arr.shape[0])
    except Exception:
        return 0


def load_spec_tensor(entry: SampleEntry) -> torch.Tensor:
    path_str = str(entry.path)
    specs = _load_file_spectrograms(path_str)
    if entry.index < 0 or entry.index >= specs.shape[0]:
        raise IndexError(f"Sample index {entry.index} out of range for {path_str}")
    sample = specs[entry.index]
    if sample.ndim != 2:
        raise ValueError(f"Expected 2-D spectrogram, got shape {sample.shape}")
    return torch.from_numpy(sample).clone().float()


class EmbeddingRouterDataset(Dataset):
    def __init__(self, entries: Sequence[SampleEntry], comm_labels: np.ndarray, task_labels: np.ndarray, preload: bool = True) -> None:
        if not (len(entries) == len(comm_labels) == len(task_labels)):
            raise ValueError("Dataset inputs must share the same length")
        self.entries = list(entries)
        self.comm_labels = torch.from_numpy(comm_labels.astype(np.int64, copy=False))
        self.task_labels = torch.from_numpy(task_labels.astype(np.int64, copy=False))
        self.metadata = [entry.metadata for entry in self.entries]
        self.preload = preload
        self.spectrograms = None
        self.preload_dtype = _resolve_preload_dtype()
        file_groups: Optional[Dict[str, List[Tuple[int, int]]]] = None
        if self.preload:
            element_size = torch.tensor([], dtype=self.preload_dtype).element_size()
            required_bytes = len(entries) * 128 * 128 * element_size
            required_gb = required_bytes / 1e9
            max_gb = _parse_float_env("LWM_PRELOAD_MAX_GB")
            available_bytes = _available_ram_bytes()
            allow_preload = True
            if max_gb is not None and required_gb > max_gb:
                print(f"[WARN] Requested preload requires {required_gb:.2f} GB, exceeding LWM_PRELOAD_MAX_GB={max_gb:.2f}; falling back to streaming.", flush=True)
                allow_preload = False
            elif available_bytes is not None and required_bytes > available_bytes * 0.8:
                print(f"[WARN] Requested preload requires {required_gb:.2f} GB but only {available_bytes / 1e9:.2f} GB appears available; falling back to streaming.", flush=True)
                allow_preload = False
            if not allow_preload:
                self.preload = False
        if self.preload:
            from collections import defaultdict
            groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
            for idx, entry in enumerate(entries):
                groups[str(entry.path)].append((idx, entry.index))
            file_groups = groups
            try:
                self.spectrograms = torch.empty((len(entries), 128, 128), dtype=self.preload_dtype)
            except RuntimeError as exc:
                print(f"[WARN] Failed to allocate preload buffer ({exc}); falling back to streaming.", flush=True)
                self.preload = False
                self.spectrograms = None
        if self.preload and self.spectrograms is not None:
            if tqdm is not None:
                iter_files = tqdm(file_groups.items(), desc="Loading files", leave=False, total=len(file_groups))
            else:
                iter_files = file_groups.items()
            for path_str, indices_list in iter_files:
                file_data = load_all_samples(path_str)
                for sample_idx, file_offset in indices_list:
                    self.spectrograms[sample_idx] = torch.from_numpy(file_data[file_offset]).to(dtype=self.preload_dtype)
            print(f"[INFO] Preloaded {self.spectrograms.shape[0]:,} spectrograms ({self.spectrograms.element_size() * self.spectrograms.nelement() / 1e9:.2f} GB)")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        if self.preload and self.spectrograms is not None:
            spec = self.spectrograms[idx].to(dtype=torch.float32)
        else:
            spec = load_spec_tensor(self.entries[idx])
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


def snr_mobility_labels_from_metadata(metadata: Sequence[SampleMetadata], *, snr_order: Sequence[str], mobility_order: Sequence[str]) -> Tuple[np.ndarray, Dict[int, Tuple[str, str]]]:
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


def prepare_dataset(*, data_root: Path, cities: Sequence[str], comm_types: Sequence[str], snrs: Optional[Sequence[str]], mobilities: Optional[Sequence[str]], modulations: Optional[Sequence[str]], fft_folders: Optional[Sequence[str]], max_samples_per_comm: int, max_per_combo: Optional[int], max_samples_per_class: int, val_samples_per_class: int, test_samples_per_class: int, task: str, seed: int, preload: bool = True) -> Tuple[EmbeddingRouterDataset, Dict[str, int], Optional[Dict[int, Tuple[str, str]]]]:
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
        snr_order = sorted({meta.snr for meta in metadata}, key=snr_sort_key) if snrs is None else [snr for snr in snrs if any(meta.snr == snr for meta in metadata)]
        mobility_order = sorted({meta.mobility for meta in metadata}) if mobilities is None else [mob for mob in mobilities if any(meta.mobility == mob for meta in metadata)]
        task_labels, mapping = snr_mobility_labels_from_metadata(metadata, snr_order=snr_order, mobility_order=mobility_order)

    dataset = EmbeddingRouterDataset(entries, comm_labels, task_labels, preload=preload)
    print(f"\n[DATA] Collected {len(dataset)} total samples:")
    for comm_name, comm_idx in sorted(comm_to_idx.items(), key=lambda x: x[1]):
        count = int((comm_labels == comm_idx).sum())
        print(f"  {comm_name}: {count:,} samples")
    print()
    return dataset, comm_to_idx, mapping


def stratified_split(labels: np.ndarray, *, train_ratio: float, val_ratio: float, max_train_per_class: int = 0, val_samples_per_class: int = 0, test_samples_per_class: int = 0, zero_shot_eval: bool = False, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0, 1)")
    if val_samples_per_class <= 0 and test_samples_per_class <= 0 and train_ratio + val_ratio >= 1.0:
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
        n_val = min(val_samples_per_class, n_total) if val_samples_per_class > 0 else int(math.floor(val_ratio * n_total))
        if val_samples_per_class > 0 and n_val < val_samples_per_class:
            print(f"[WARN] Class {label}: requested {val_samples_per_class} validation samples but only {n_total} available")
        remaining_after_val = max(n_total - n_val, 0)
        n_test = min(test_samples_per_class, remaining_after_val) if test_samples_per_class > 0 else int(math.floor(base_test_ratio * n_total))
        if test_samples_per_class > 0 and n_test < test_samples_per_class:
            print(f"[WARN] Class {label}: requested {test_samples_per_class} test samples but only {remaining_after_val} available after validation")
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

        start_train = n_val + n_test
        train_indices.extend(idx[start_train:start_train + n_train])
        val_indices.extend(idx[:n_val])
        test_indices.extend(idx[n_val:n_val + n_test])

    return (np.sort(np.array(train_indices, dtype=np.int64)), np.sort(np.array(val_indices, dtype=np.int64)), np.sort(np.array(test_indices, dtype=np.int64)))


class RouterNet(nn.Module):
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
        return self.classifier(self.features(x))


class TaskClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(128), Res1DCNNHead(128, num_classes, dropout=dropout))

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
        rfprompt_global: int = 4,
        rfprompt_spectral: int = 4,
        rfprompt_temporal: int = 2,
        rfprompt_condition: int = 2,
        rfprompt_use_router: bool = False,
        rfprompt_pool_prompts: bool = False,
    ) -> None:
        super().__init__()
        self._trainable = bool(trainable)
        self._prompt_hybrid = bool(prompt_hybrid)
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
                self.stats = {"normalization": "per_sample", "mean": 0.0, "std": 1.0}
        else:
            self.stats = {"normalization": "per_sample", "mean": 0.0, "std": 1.0}

        normalization_stats = None
        if self.stats["normalization"] != "per_sample":
            normalization_stats = {
                "normalization": self.stats["normalization"],
                "mean": self.stats["mean"],
                "std": self.stats["std"],
            }

        prepare_kwargs = dict(
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
        # Backward compatibility: only pass RFPrompt-specific kwargs when prepare_model supports them.
        try:
            sig = inspect.signature(prepare_model)
            if "rfprompt_global" in sig.parameters:
                prepare_kwargs.update(
                    rfprompt_global=rfprompt_global,
                    rfprompt_spectral=rfprompt_spectral,
                    rfprompt_temporal=rfprompt_temporal,
                    rfprompt_condition=rfprompt_condition,
                    rfprompt_use_router=rfprompt_use_router,
                    rfprompt_pool_prompts=rfprompt_pool_prompts,
                )
        except Exception:
            pass

        model = prepare_model(**prepare_kwargs)
        model.train(self._trainable)
        self.model = model.to(device)

    @property
    def trainable(self) -> bool:
        return self._trainable

    def set_trainable(self, trainable: bool) -> None:
        self._trainable = bool(trainable)
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        uses_prompt_wrapper = bool(
            getattr(self.model, "pool", None) is not None
            or hasattr(self.model, "layer_prompts")
            or hasattr(self.model, "rfprompt")
            or hasattr(self.model, "rfprompt_config")
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

    def forward_prenormalized(self, specs: torch.Tensor, *, allow_grad: Optional[bool] = None, return_tokens: bool = False) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
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


def build_dataloaders(dataset: EmbeddingRouterDataset, *, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def _subset(indices: np.ndarray) -> torch.utils.data.Subset:
        return torch.utils.data.Subset(dataset, indices.tolist())
    use_cuda = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None
    train_shuffle = len(train_idx) > 0
    train_drop_last = len(train_idx) > 0
    train_loader = DataLoader(_subset(train_idx), batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, pin_memory=use_cuda, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor, drop_last=train_drop_last)
    val_loader = DataLoader(_subset(val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(_subset(test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    return train_loader, val_loader, test_loader


def aggregate_comm_probs(probs: torch.Tensor, group_map: Mapping[int, List[int]]) -> torch.Tensor:
    num_comm = len(group_map)
    agg = torch.zeros(probs.size(0), num_comm, device=probs.device, dtype=probs.dtype)
    for comm_idx, expert_indices in group_map.items():
        if expert_indices:
            agg[:, comm_idx] = probs[:, expert_indices].sum(dim=1)
    return agg


def router_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, group_map: Mapping[int, List[int]]) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    agg = aggregate_comm_probs(probs, group_map)
    agg = torch.clamp(agg, min=1e-12)
    return F.nll_loss(agg.log(), targets)


def build_group_map(experts: Sequence[ExpertSpec], comm_to_idx: Mapping[str, int]) -> Dict[int, List[int]]:
    grouping: Dict[int, List[int]] = {idx: [] for idx in comm_to_idx.values()}
    for expert_idx, spec in enumerate(experts):
        grouping[comm_to_idx[spec.comm]].append(expert_idx)
    return grouping

# ---- training/eval utilities below are unchanged in logic ----
# (Kept as-is except checkpoint/prompt-config plumbing.)

# Due to size, the rest of the training / evaluation functions are identical to your original file,
# except for the prompt-config additions in parse_args(), load_experts(), checkpoint save/load, and
# the fresh-training call to load_experts() in main().
#
# Paste your original functions from:
#   train_router
#   evaluate_router
#   stack_expert_embeddings
#   compute_selected_expert_embeddings
#   gather_topk_embeddings
#   _classifier_class_weights
#   focal_loss
#   compute_class_weights_from_indices
#   compute_class_weights_from_loader
#   locoop_nuisance_loss
#   train_task_model
#   evaluate_task_model
#   train_oracle_baseline
#   evaluate_oracle_baseline
#   SingleModelBackbone
#   ImageNetBackbone
#   train_single_model
#   evaluate_single_model
#   evaluate_test_metrics
# exactly as in your current version.

# -------------------- parse args (UPDATED) --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("spectrograms"))
    parser.add_argument("--cities", nargs="*", default=["city_1_losangeles"])
    parser.add_argument("--comm-types", nargs="*", default=["LTE", "WiFi", "5G"])
    parser.add_argument("--snrs", nargs="*", default=None)
    parser.add_argument("--mobilities", nargs="*", default=["pedestrian", "vehicular"])
    parser.add_argument("--modulations", nargs="*", default=None)
    parser.add_argument("--fft-folders", nargs="*", default=None)
    parser.add_argument("--task", choices=("modulation", "snr_mobility"), default="snr_mobility")
    parser.add_argument("--max-samples-per-comm", type=int, default=0)
    parser.add_argument("--max-per-combo", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    parser.add_argument("--val-samples-per-class", type=int, default=0)
    parser.add_argument("--test-samples-per-class", type=int, default=0)
    parser.add_argument("--zero-shot-eval", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--router-epochs", type=int, default=2)
    parser.add_argument("--task-epochs", type=int, default=25)
    parser.add_argument("--selection-metric", type=str, default="val_f1", choices=("val_f1", "val_loss"))
    parser.add_argument("--log-magnitude", action="store_true")
    parser.add_argument("--router-lr", type=float, default=5e-4)
    parser.add_argument("--router-warmup-lr", type=float, default=3e-4)
    parser.add_argument("--classifier-lr", type=float, default=2e-3)
    parser.add_argument("--expert-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--router-loss-weight", type=float, default=0.05)
    parser.add_argument("--load-balance-weight", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--routing-topk", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--preload-data", action="store_true", default=True)
    parser.add_argument("--no-preload-data", dest="preload_data", action="store_false")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--expert", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("MoE/runs/embedding_router"))
    parser.add_argument("--save-router", action="store_true")
    parser.add_argument("--save-classifier", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-router-warmup", action="store_true")
    parser.add_argument("--baseline", choices=["oracle", "single", "imagenet"], default=None)
    parser.add_argument("--gating-noise-std", type=float, default=0.1)
    parser.add_argument("--gating-noise-epochs", type=int, default=5)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--expert-use-prompts", action="store_true")
    parser.add_argument("--expert-num-prompts", type=int, default=16)
    parser.add_argument("--expert-prompt-style", type=str, default="deep", choices=("deep", "l2p", "soft", "rfprompt"))
    parser.add_argument("--expert-pool-size", type=int, default=10)
    parser.add_argument("--expert-selection-size", type=int, default=5)
    parser.add_argument("--expert-hybrid-prompts", action="store_true")
    parser.add_argument("--expert-rfprompt-global", type=int, default=4)
    parser.add_argument("--expert-rfprompt-spectral", type=int, default=4)
    parser.add_argument("--expert-rfprompt-temporal", type=int, default=2)
    parser.add_argument("--expert-rfprompt-condition", type=int, default=2)
    parser.add_argument("--expert-rfprompt-use-router", action="store_true")
    parser.add_argument("--expert-rfprompt-pool-prompts", action="store_true")
    parser.add_argument("--locoop-lambda", type=float, default=0.0)
    parser.add_argument("--locoop-margin", type=float, default=0.0)
    parser.add_argument("--locoop-nuisance-frac", type=float, default=0.5)
    parser.add_argument("--task-loss", type=str, default="ce", choices=("ce", "focal", "weighted_ce"))
    parser.add_argument("--focal-gamma", type=float, default=2.0)
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
        stats_candidate = Path(REPO_ROOT / "models" / f"{comm}_models" / "dataset_stats.json").resolve()
        stats_path = stats_candidate if stats_candidate.exists() else None
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
    rfprompt_global: int = 4,
    rfprompt_spectral: int = 4,
    rfprompt_temporal: int = 2,
    rfprompt_condition: int = 2,
    rfprompt_use_router: bool = False,
    rfprompt_pool_prompts: bool = False,
) -> List[EmbeddingExpert]:
    embeddings: List[EmbeddingExpert] = []
    for spec in specs:
        desc = f"[INFO] Loading expert '{spec.name}' ({spec.comm}) from {spec.checkpoint}"
        if trainable:
            desc += " [trainable]"
        if use_prompts:
            desc += f" [prompts={prompt_style}]"
        if use_prompts and prompt_style == "rfprompt":
            desc += f" [rfprompt g={rfprompt_global} s={rfprompt_spectral} t={rfprompt_temporal} c={rfprompt_condition} router={int(rfprompt_use_router)} pool_prompts={int(rfprompt_pool_prompts)}]"
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
                rfprompt_global=rfprompt_global,
                rfprompt_spectral=rfprompt_spectral,
                rfprompt_temporal=rfprompt_temporal,
                rfprompt_condition=rfprompt_condition,
                rfprompt_use_router=rfprompt_use_router,
                rfprompt_pool_prompts=rfprompt_pool_prompts,
            )
        )
    return embeddings

# ---- checkpoint helpers updated with prompt_config ----

def save_complete_checkpoint(*, router: Optional[RouterNet], classifier: TaskClassifier, expert_models: Optional[Sequence[EmbeddingExpert]], expert_specs: Sequence[ExpertSpec], comm_to_idx: Mapping[str, int], task_type: str, num_classes: int, topk: int, dropout: float, mapping: Optional[Dict[int, Tuple[str, str]]], output_path: Path, model_type: str = "embedding_router_moe", backbone_state_dict: Optional[Dict[str, torch.Tensor]] = None, backbone_meta: Optional[Dict[str, Any]] = None, expert_trainable: bool = False, prompt_config: Optional[Dict[str, Any]] = None) -> None:
    checkpoint = {
        "model_type": model_type,
        "task": task_type,
        "num_classes": num_classes,
        "topk": topk,
        "dropout": dropout,
        "comm_to_idx": dict(comm_to_idx),
        "experts": [
            {"name": spec.name, "comm": spec.comm, "checkpoint": str(spec.checkpoint), "stats_path": str(spec.stats_path) if spec.stats_path else None}
            for spec in expert_specs
        ],
        "classifier_state_dict": classifier.state_dict(),
        "mapping": {int(k): v for k, v in mapping.items()} if mapping else None,
        "expert_trainable": bool(expert_trainable),
        "prompt_config": prompt_config,
    }
    if router is not None:
        checkpoint["router_state_dict"] = router.state_dict()
    if backbone_state_dict is not None:
        checkpoint["backbone_state_dict"] = backbone_state_dict
    if backbone_meta is not None:
        checkpoint["backbone_meta"] = backbone_meta
    if expert_models is not None:
        def _is_trainable(expert: nn.Module) -> bool:
            if hasattr(expert, "trainable"):
                return bool(getattr(expert, "trainable"))
            if hasattr(expert, "module") and hasattr(expert.module, "trainable"):
                return bool(expert.module.trainable)
            return False
        if any(_is_trainable(expert) for expert in expert_models):
            checkpoint["expert_state_dicts"] = [
                {"name": spec.name, "state_dict": {k: v.cpu() for k, v in expert.state_dict().items()}}
                for spec, expert in zip(expert_specs, expert_models)
            ]
    torch.save(checkpoint, output_path)
    print(f"[INFO] Complete checkpoint saved to {output_path}")


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidate = (REPO_ROOT / path).resolve() if not path.is_absolute() else path
    return candidate


def _checkpoint_to_expert_specs(checkpoint: Mapping[str, Any]) -> List[ExpertSpec]:
    specs: List[ExpertSpec] = []
    for expert in checkpoint.get("experts", []):
        checkpoint_path = _resolve_repo_path(expert["checkpoint"])
        stats_path = None
        stats_path_str = expert.get("stats_path")
        if stats_path_str:
            candidate = _resolve_repo_path(stats_path_str)
            if candidate.exists():
                stats_path = candidate
        specs.append(ExpertSpec(name=expert["name"], comm=canonical_comm_name(expert["comm"]), checkpoint=checkpoint_path, stats_path=stats_path))
    return specs


def _normalize_comm_mapping(comm_to_idx_raw: Mapping[str, Any]) -> Dict[str, int]:
    return {canonical_comm_name(str(k)): int(v) for k, v in comm_to_idx_raw.items()}


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


def _build_checkpoint_components(checkpoint: Mapping[str, Any], device: torch.device, *, train_mode: bool) -> Dict[str, Any]:
    dropout = float(checkpoint.get("dropout", 0.1))
    num_classes = int(checkpoint["num_classes"])
    expert_specs = _checkpoint_to_expert_specs(checkpoint)
    expert_trainable_flag = bool(checkpoint.get("expert_trainable", False))
    expert_state_dicts = checkpoint.get("expert_state_dicts")
    name_to_state: Dict[Optional[str], Any] = {}
    if expert_state_dicts:
        name_to_state = {entry.get("name"): entry.get("state_dict") for entry in expert_state_dicts if isinstance(entry, Mapping)}

    use_prompts = False
    prompt_style = "deep"
    num_prompts = 16
    pool_size = 10
    selection_size = 4
    rfprompt_global = 4
    rfprompt_spectral = 4
    rfprompt_temporal = 2
    rfprompt_condition = 2
    rfprompt_use_router = False
    rfprompt_pool_prompts = False

    prompt_config = checkpoint.get("prompt_config")
    if isinstance(prompt_config, Mapping):
        use_prompts = bool(prompt_config.get("use_prompts", use_prompts))
        prompt_style = str(prompt_config.get("prompt_style", prompt_style))
        num_prompts = int(prompt_config.get("num_prompts", num_prompts))
        pool_size = int(prompt_config.get("pool_size", pool_size))
        selection_size = int(prompt_config.get("selection_size", selection_size))
        rfprompt_global = int(prompt_config.get("rfprompt_global", rfprompt_global))
        rfprompt_spectral = int(prompt_config.get("rfprompt_spectral", rfprompt_spectral))
        rfprompt_temporal = int(prompt_config.get("rfprompt_temporal", rfprompt_temporal))
        rfprompt_condition = int(prompt_config.get("rfprompt_condition", rfprompt_condition))
        rfprompt_use_router = bool(prompt_config.get("rfprompt_use_router", rfprompt_use_router))
        rfprompt_pool_prompts = bool(prompt_config.get("rfprompt_pool_prompts", rfprompt_pool_prompts))

    if name_to_state and not isinstance(prompt_config, Mapping):
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
        rfprompt_global=rfprompt_global,
        rfprompt_spectral=rfprompt_spectral,
        rfprompt_temporal=rfprompt_temporal,
        rfprompt_condition=rfprompt_condition,
        rfprompt_use_router=rfprompt_use_router,
        rfprompt_pool_prompts=rfprompt_pool_prompts,
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
        router.train() if train_mode else router.eval()

    classifier = TaskClassifier(num_classes=num_classes, dropout=dropout).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.train() if train_mode else classifier.eval()

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

# ---- keep your remaining helper functions (resume/load predictor/etc.) unchanged ----

# To keep this file manageable in chat, the rest of the unchanged code is omitted here.
# Merge the updated blocks above into your current file, then in main() make sure to:
#
# 1) Build prompt_config before checkpoint saving:
#
# prompt_config = {
#     "use_prompts": bool(getattr(args, "expert_use_prompts", False)),
#     "num_prompts": int(getattr(args, "expert_num_prompts", 16)),
#     "prompt_style": str(getattr(args, "expert_prompt_style", "deep")),
#     "pool_size": int(getattr(args, "expert_pool_size", 10)),
#     "selection_size": int(getattr(args, "expert_selection_size", 5)),
#     "rfprompt_global": int(getattr(args, "expert_rfprompt_global", 4)),
#     "rfprompt_spectral": int(getattr(args, "expert_rfprompt_spectral", 4)),
#     "rfprompt_temporal": int(getattr(args, "expert_rfprompt_temporal", 2)),
#     "rfprompt_condition": int(getattr(args, "expert_rfprompt_condition", 2)),
#     "rfprompt_use_router": bool(getattr(args, "expert_rfprompt_use_router", False)),
#     "rfprompt_pool_prompts": bool(getattr(args, "expert_rfprompt_pool_prompts", False)),
#     "expert_hybrid_prompts": bool(getattr(args, "expert_hybrid_prompts", False)),
# }
#
# 2) In the fresh-training load_experts(...) call, pass the RFPrompt args.
#
# 3) In both save_complete_checkpoint(...) calls, pass:
#       prompt_config=prompt_config if args.baseline is None else None
#
# 4) In your driver, use:
#       --expert-use-prompts
#       --expert-prompt-style rfprompt
#       --expert-num-prompts 12
#       --expert-rfprompt-global 4
#       --expert-rfprompt-spectral 4
#       --expert-rfprompt-temporal 2
#       --expert-rfprompt-condition 2
#
# Final note:
# prepare_model(...) must accept the new RFPrompt kwargs. Without that backend change,
# this file will parse/pass the config but RFPrompt still will not work end-to-end.
