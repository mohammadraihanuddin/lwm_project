#!/usr/bin/env python3
"""
Collect Stage-A (data-scale) sweep metrics from outputs/ after syncing from Palmetto.

Expected directory names:
  outputs/mod_class_moe_{ieee|realworld_iq}_{EPOCHS}ep_N{cap}_{frozen|pft|rfprompt}/metrics.json

Usage:
  python scripts/collect_stage_a_sweep_metrics.py
  python scripts/collect_stage_a_sweep_metrics.py --epochs 100 --root outputs --latex

Options:
  --missing   Print runs that are missing metrics.json
  --csv       Print CSV rows
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Regex: mod_class_moe_<dataset>_<ep>ep_N<cap>_<mode>
DIR_RE = re.compile(
    r"^mod_class_moe_(?P<ds>ieee|realworld_iq|phoenix)_(?P<ep>\d+)ep_N(?P<n>\d+)_(?P<mode>frozen|pft|rfprompt)$"
)


def load_metrics(path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None
    acc = data.get("test_accuracy")
    f1 = data.get("test_f1")
    if acc is not None:
        acc = float(acc)
    if f1 is not None:
        f1 = float(f1)
    return acc, f1


def fmt_pair(acc: Optional[float], f1: Optional[float], width: int = 3) -> str:
    if acc is None or f1 is None:
        return r"\TBD"
    return f"{acc:.{width}f} / {f1:.{width}f}"


def discover(
    root: Path, epochs: int
) -> Dict[Tuple[str, int, str], Tuple[Optional[float], Optional[float]]]:
    """Map (dataset, N, mode) -> (acc, f1)."""
    out: Dict[Tuple[str, int, str], Tuple[Optional[float], Optional[float]]] = {}
    if not root.is_dir():
        return out
    ep_str = str(epochs)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = DIR_RE.match(child.name)
        if not m:
            continue
        if m.group("ep") != ep_str:
            continue
        ds = m.group("ds")
        n = int(m.group("n"))
        mode = m.group("mode")
        metrics_path = child / "metrics.json"
        acc, f1 = load_metrics(metrics_path) if metrics_path.is_file() else (None, None)
        out[(ds, n, mode)] = (acc, f1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("outputs"), help="Outputs directory (default: ./outputs)")
    ap.add_argument("--epochs", type=int, default=100, help="Epochs tag in run name (default: 100)")
    ap.add_argument("--latex", action="store_true", help="Print LaTeX rows for paper tables")
    ap.add_argument("--csv", action="store_true", help="Print CSV")
    ap.add_argument("--missing", action="store_true", help="List expected runs without metrics.json")
    args = ap.parse_args()

    root = args.root.resolve()
    data = discover(root, args.epochs)
    ns = [100, 200, 400, 800, 1600]
    modes = ["frozen", "pft", "rfprompt"]
    datasets = ["realworld_iq", "ieee", "phoenix"]

    if args.missing:
        missing: List[str] = []
        for ds in datasets:
            for n in ns:
                for mode in modes:
                    key = (ds, n, mode)
                    acc, f1 = data.get(key, (None, None))
                    if acc is None:
                        ep = args.epochs
                        name = f"mod_class_moe_{ds}_{ep}ep_N{n}_{mode}"
                        missing.append(name)
        if not missing:
            print("No missing runs (all metrics present).", file=sys.stderr)
        else:
            print("Missing metrics.json for:")
            for name in sorted(missing):
                print(name)
        return 0

    if args.csv:
        print("dataset,N,mode,accuracy,f1")
        for ds in datasets:
            for n in ns:
                for mode in modes:
                    acc, f1 = data.get((ds, n, mode), (None, None))
                    print(
                        f"{ds},{n},{mode},"
                        f"{'' if acc is None else f'{acc:.6f}'},"
                        f"{'' if f1 is None else f'{f1:.6f}'}"
                    )
        return 0

    if args.latex:
        for ds, caption_label in [
            ("realworld_iq", "tab:rw_scale"),
            ("ieee", "tab:ieee_scale"),
            ("phoenix", "tab:phoenix_scale"),
        ]:
            if ds == "realworld_iq":
                title = "Real-World IQ"
            elif ds == "ieee":
                title = "IEEE Dataport"
            else:
                title = "DeepMIMO Phoenix"
            print(f"% --- {title} ({caption_label}) ---")
            for n in ns:
                cells = [str(n)]
                for mode in modes:
                    acc, f1 = data.get((ds, n, mode), (None, None))
                    cells.append(fmt_pair(acc, f1))
                print("    " + " & ".join(cells) + r" \\")
            print()
        return 0

    # Default: human-readable summary
    for ds in datasets:
        print(f"\n=== {ds} ===")
        header = f"{'N':>5}  {'frozen':>18}  {'pft':>18}  {'rfprompt':>18}"
        print(header)
        print("-" * len(header))
        for n in ns:
            parts = [f"{n:>5}"]
            for mode in modes:
                acc, f1 = data.get((ds, n, mode), (None, None))
                if acc is None:
                    parts.append(f"{'—':>18}")
                else:
                    parts.append(f"{acc:.3f} / {f1:.3f}".rjust(18))
            print("  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
