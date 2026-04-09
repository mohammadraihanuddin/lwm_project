#!/usr/bin/env python3
"""
Modulation classification with MoE — frozen experts (with optional LoRA).

This is the single-dataset driver for the "frozen experts" baseline plus
optional LoRA. It prepares `sys.argv` and then calls
`MoE.train_embedding_router.main`.

Key behaviour (controlled by environment variables):

- MOD_MOE_DATASET:
    ieee          → datasets/spectrograms_ieee / city_ieee_dataport
    realworld_iq  → datasets/spectrograms_realworld_iq / city_realworld_iq
    radioml       → datasets/spectrograms_radioml / city_radioml
  If not set, the dataset may also be passed as a first CLI arg, e.g.
    PYTHONPATH=. MOD_MOE_DATASET=ieee python drivers/run_frozen.py

- MOD_MOE_TASK_EPOCHS:
    Number of task epochs (default "100").
    We set patience=5 for ≤10 epochs, else 15.

- MOD_MOE_BATCH_SIZE:
    Batch size for MoE training (default "32").

- MOD_MOE_MAX_SAMPLES_PER_CLASS:
    Maximum train samples per class (shots cap). "0" means use all.

- MOD_MOE_USE_LORA, MOD_MOE_LORA_R, MOD_MOE_LORA_ALPHA:
    When MOD_MOE_USE_LORA=1 and the underlying backbone / mobility_utils
    supports it, LoRA adapters are applied inside the experts while the base
    weights remain frozen. This script itself only sets up the standard MoE
    CLI; LoRA is toggled via these env vars inside the model code.

All runs write their outputs to:
    REPO_ROOT / "outputs" / RUN_SUBDIR
where RUN_SUBDIR depends on the dataset and epochs, e.g.
    mod_class_moe_ieee_100ep
    mod_class_moe_realworld_iq_100ep
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPERTS_DIR = REPO_ROOT / "experts"
LTE_CKPT = EXPERTS_DIR / "LTE_expert.pth"
WIFI_CKPT = EXPERTS_DIR / "WiFi_expert.pth"
FIVE_G_CKPT = EXPERTS_DIR / "5G_expert.pth"

# Dataset config: (data_root_relative, city_folder_name, default_run_subdir)
DATASET_CONFIG: dict[str, tuple[str, str, str]] = {
    "ieee": ("datasets/spectrograms_ieee", "city_ieee_dataport", "mod_class_moe_ieee_100ep"),
    "realworld_iq": ("datasets/spectrograms_realworld_iq", "city_realworld_iq", "mod_class_moe_realworld_iq_100ep"),
    "radioml": ("datasets/spectrograms_radioml", "city_radioml", "mod_class_moe_radioml_100ep"),
    "phoenix": ("datasets/spectrograms", "city_4_phoenix", "mod_class_moe_phoenix_100ep"),
}


def _get_dataset() -> str:
    """Resolve dataset from CLI arg or MOD_MOE_DATASET env."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        candidate = sys.argv[1].strip().lower()
        if candidate in DATASET_CONFIG:
            return candidate
    env_ds = os.environ.get("MOD_MOE_DATASET", "").strip().lower()
    if env_ds in DATASET_CONFIG:
        return env_ds
    raise SystemExit(
        f"[mod_class_moe] Please specify dataset via MOD_MOE_DATASET "
        f"or as first CLI arg. Options: {list(DATASET_CONFIG.keys())}"
    )


def _resolve_paths(dataset: str) -> tuple[Path, str, Path]:
    rel_root, city_name, default_run = DATASET_CONFIG[dataset]

    data_root_env = os.environ.get("MOD_MOE_DATA_ROOT")
    data_root = Path(data_root_env) if data_root_env else REPO_ROOT / rel_root
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"[mod_class_moe] Spectrogram data not found: {data_root}. "
            f"For {dataset}, export to {rel_root} (e.g. scripts/export_*_to_spectrograms_dir.py)."
        )

    city_dir = data_root / city_name
    if not city_dir.is_dir():
        raise FileNotFoundError(
            f"[mod_class_moe] City folder not found: {city_dir}. Expected {data_root}/{city_name}."
        )

    comms_found = [c for c in ("LTE", "WiFi", "5G") if (city_dir / c).is_dir()]
    if not comms_found:
        raise FileNotFoundError(
            f"[mod_class_moe] No comm folders (LTE, WiFi, 5G) under {city_dir}. "
            "Export with per-sample 'tech' so LTE/WiFi/5G subdirs exist."
        )

    run_name_env = os.environ.get("MOD_MOE_RUN_NAME")
    run_subdir = run_name_env if run_name_env else default_run
    output_dir = REPO_ROOT / "outputs" / run_subdir

    print(
        f"[mod_class_moe] Dataset: {dataset} | data root: {data_root} | city: {city_name} | comms: {comms_found}"
    )
    print("[mod_class_moe] Experts: frozen (--expert-lr 0; LoRA controlled via env, if enabled)")

    return data_root, city_name, output_dir


def _check_experts() -> None:
    for name, p in [("LTE", LTE_CKPT), ("WiFi", WIFI_CKPT), ("5G", FIVE_G_CKPT)]:
        if not p.is_file():
            raise FileNotFoundError(f"[mod_class_moe] Expert checkpoint not found for {name}: {p}")


def main() -> None:
    dataset = _get_dataset()
    _check_experts()

    data_root, city_name, output_dir = _resolve_paths(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_epochs = int(os.environ.get("MOD_MOE_TASK_EPOCHS", "100"))
    patience = 5 if task_epochs <= 10 else 15
    batch_size = int(os.environ.get("MOD_MOE_BATCH_SIZE", "32"))
    max_per_class = int(os.environ.get("MOD_MOE_MAX_SAMPLES_PER_CLASS", "0"))

    print(
        f"[mod_class_moe] Config: epochs={task_epochs} patience={patience} "
        f"batch={batch_size} max_per_class={max_per_class}"
    )

    # Build argv for MoE.train_embedding_router
    argv: list[str] = [
        "train_embedding_router",
        "--data-root",
        str(data_root),
        "--cities",
        city_name,
        "--task",
        "modulation",
        "--comm-types",
        "LTE",
        "WiFi",
        "5G",
        "--mobilities",
        "static",
        "--snrs",
        "SNR10dB",
        "--max-samples-per-class",
        str(max_per_class),
        "--val-samples-per-class",
        "0",
        "--test-samples-per-class",
        "0",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        "0",
        "--no-preload-data",
        "--router-epochs",
        "2",
        "--task-epochs",
        str(task_epochs),
        "--patience",
        str(patience),
        "--expert-lr",
        "0",  # frozen experts (LoRA, if any, is controlled internally via env)
        "--expert",
        f"lte_base=LTE:{LTE_CKPT}",
        "--expert",
        f"wifi_base=WiFi:{WIFI_CKPT}",
        "--expert",
        f"5g_base=5G:{FIVE_G_CKPT}",
        "--output-dir",
        str(output_dir),
        "--save-router",
        "--save-classifier",
    ]

    sys.argv = argv
    from MoE.train_embedding_router import main as run_moe  # type: ignore

    run_moe()


if __name__ == "__main__":
    main()

