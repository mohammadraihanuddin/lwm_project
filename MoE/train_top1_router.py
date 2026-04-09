"""Data discovery for MoE train_embedding_router: collect spectrogram files by comm/snr/mobility."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# Extensions we treat as spectrogram containers (pickle/numpy)
SPECTROGRAM_EXTENSIONS = (".pkl", ".pickle", ".pt", ".pth", ".npy")


@dataclass
class SampleMetadata:
    path: Path
    comm: str
    snr: str
    mobility: str
    modulation: str = ""
    rate: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not self.source and self.path:
            self.source = str(self.path)


def _collect_candidate_files(
    data_root: Path,
    cities: Sequence[str],
    comm: str,
    snr_filters: Optional[Sequence[str]] = None,
    mobility_filters: Optional[Sequence[str]] = None,
    modulation_filters: Optional[Sequence[str]] = None,
    fft_filters: Optional[Sequence[str]] = None,
) -> List[Tuple[Path, SampleMetadata]]:
    """Find spectrogram files under data_root/<city>/<comm>/... and return (path, metadata)."""
    data_root = Path(data_root)
    results: List[Tuple[Path, SampleMetadata]] = []
    snr_set = set((snr_filters or []))
    mobility_set = set((mobility_filters or []))

    for city in cities:
        comm_dir = data_root / city / comm
        if not comm_dir.is_dir():
            continue
        for path in comm_dir.rglob("*"):
            if path.suffix.lower() not in SPECTROGRAM_EXTENSIONS or not path.is_file():
                continue
            parts = path.relative_to(comm_dir).parts
            snr = "SNR10dB"
            mobility = "static"
            modulation = comm
            for p in parts[:-1]:
                p_upper = p.upper()
                if "SNR" in p_upper:
                    snr = p
                elif p in ("static", "pedestrian", "vehicular", "mobile"):
                    mobility = p
                elif p.upper().startswith("CLASS_"):
                    modulation = p
                elif p not in ("", ".", "spectrograms", "spectrogram", "rate1", "512FFT"):
                    if not modulation.startswith("CLASS_"):
                        modulation = p
            if snr_set and snr not in snr_set:
                continue
            if mobility_set and mobility not in mobility_set:
                continue
            meta = SampleMetadata(
                path=path,
                comm=comm,
                snr=snr,
                mobility=mobility,
                modulation=modulation,
                rate="",
                source=str(path),
            )
            results.append((path, meta))

    return results


def load_dataset_stats(stats_path: Optional[Path], *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    """Return (stats_dict, None) if stats_path exists and is loadable; else (None, None)."""
    if stats_path is None or not Path(stats_path).exists():
        return None, None
    try:
        import json
        with open(stats_path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception:
        return None, None


def snr_sort_key(name: str) -> int:
    """Sort key for SNR folder names (e.g. SNR10dB -> 10)."""
    m = re.search(r"([-+]?\d+)", name)
    return int(m.group(1)) if m else 0
