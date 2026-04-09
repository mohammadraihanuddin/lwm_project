#!/usr/bin/env python3
"""
Modulation classification with MoE — RFPrompt experts (backbone frozen).

Environment configuration:
- MOD_MOE_DATASET:
    ieee, realworld_iq, radioml
- MOD_MOE_TASK_EPOCHS:
    task epochs (default: 50)
- MOD_MOE_BATCH_SIZE:
    batch size (default: 32)
- MOD_MOE_MAX_SAMPLES_PER_CLASS:
    max train samples per class (default: 0 = all)
- MOD_MOE_NUM_PROMPTS:
    total RFPrompt tokens (default: 20). Groups are auto-split.
- MOD_MOE_DATA_ROOT:
    optional override for dataset root
- MOD_MOE_RUN_NAME:
    optional override for output subdirectory
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

DATASET_CONFIG: dict[str, tuple[str, str, str]] = {
    "ieee": (
        "datasets/spectrograms_ieee",
        "city_ieee_dataport",
        "mod_class_moe_ieee_100ep_rfprompt",
    ),
    "realworld_iq": (
        "datasets/spectrograms_realworld_iq",
        "city_realworld_iq",
        "mod_class_moe_realworld_iq_100ep_rfprompt",
    ),
    "radioml": (
        "datasets/spectrograms_radioml",
        "city_radioml",
        "mod_class_moe_radioml_100ep_rfprompt",
    ),
}


def _get_dataset() -> str:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        ds = sys.argv[1].strip().lower()
        if ds in DATASET_CONFIG:
            return ds

    env_ds = os.environ.get("MOD_MOE_DATASET", "").strip().lower()
    if env_ds in DATASET_CONFIG:
        return env_ds

    raise SystemExit(
        f"[mod_class_moe_rfprompt] Please specify dataset via MOD_MOE_DATASET "
        f"or as first CLI arg. Options: {list(DATASET_CONFIG.keys())}"
    )


def _resolve_paths(dataset: str) -> tuple[Path, str, Path]:
    rel_root, city_name, default_run = DATASET_CONFIG[dataset]

    data_root_env = os.environ.get("MOD_MOE_DATA_ROOT")
    data_root = Path(data_root_env) if data_root_env else (REPO_ROOT / rel_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"[mod_class_moe_rfprompt] Spectrogram data not found: {data_root}. "
            f"For {dataset}, expected under {rel_root} unless MOD_MOE_DATA_ROOT is set."
        )

    city_dir = data_root / city_name
    if not city_dir.is_dir():
        raise FileNotFoundError(
            f"[mod_class_moe_rfprompt] City folder not found: {city_dir}"
        )

    comms_found = [c for c in ("LTE", "WiFi", "5G") if (city_dir / c).is_dir()]
    if not comms_found:
        raise FileNotFoundError(
            f"[mod_class_moe_rfprompt] No comm folders (LTE, WiFi, 5G) under {city_dir}."
        )

    run_name_env = os.environ.get("MOD_MOE_RUN_NAME")
    run_subdir = run_name_env if run_name_env else default_run
    output_dir = REPO_ROOT / "outputs" / run_subdir

    print(
        f"[mod_class_moe_rfprompt] Dataset: {dataset} | data root: {data_root} | "
        f"city: {city_name} | comms found: {comms_found}"
    )
    print("[mod_class_moe_rfprompt] Mode: RFPrompt (structured prompts, frozen backbone)")

    return data_root, city_name, output_dir


def _check_experts() -> None:
    for name, path in [
        ("LTE", LTE_CKPT),
        ("WiFi", WIFI_CKPT),
        ("5G", FIVE_G_CKPT),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"[mod_class_moe_rfprompt] Expert checkpoint not found for {name}: {path}"
            )


def main() -> None:
    dataset = _get_dataset()
    _check_experts()

    data_root, city_name, output_dir = _resolve_paths(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_epochs = int(os.environ.get("MOD_MOE_TASK_EPOCHS", "50"))
    patience = 5 if task_epochs <= 10 else 15
    batch_size = int(os.environ.get("MOD_MOE_BATCH_SIZE", "32"))
    max_per_class = int(os.environ.get("MOD_MOE_MAX_SAMPLES_PER_CLASS", "0"))

    print(
        f"[mod_class_moe_rfprompt] Config: epochs={task_epochs} "
        f"patience={patience} batch={batch_size} max_per_class={max_per_class}"
    )

    argv: list[str] = [
        "train_embedding_router",
        "--data-root", str(data_root),
        "--cities", city_name,
        "--task", "modulation",
        "--comm-types", "LTE", "WiFi", "5G",
        "--mobilities", "static",
        "--snrs", "SNR10dB",
        "--max-samples-per-class", str(max_per_class),
        "--val-samples-per-class", "0",
        "--test-samples-per-class", "0",
        "--batch-size", str(batch_size),
        "--num-workers", "0",
        "--no-preload-data",
        "--router-epochs", "2",
        "--task-epochs", str(task_epochs),
        "--patience", str(patience),

        # RFPrompt prompt-only mode
        "--expert-use-prompts",
        "--expert-prompt-style", "rfprompt",
        "--expert-num-prompts", str(int(os.environ.get("MOD_MOE_NUM_PROMPTS", "20"))),

        "--expert-lr", "5e-4",

        "--expert", f"lte_base=LTE:{LTE_CKPT}",
        "--expert", f"wifi_base=WiFi:{WIFI_CKPT}",
        "--expert", f"5g_base=5G:{FIVE_G_CKPT}",

        "--output-dir", str(output_dir),
        "--save-router",
        "--save-classifier",
    ]

    sys.argv = argv
    from MoE.train_embedding_router import main as run_moe  # type: ignore

    run_moe()


if __name__ == "__main__":
    main()