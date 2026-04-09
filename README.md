# RFPrompt-MoE

Clean, paper-focused repository for modulation classification experiments with:

- Frozen Experts
- Partial Fine-Tuning (PFT)
- RFPrompt (our proposed method)
- Conventional baselines:
  - ResNet18
  - EfficientNet-B0
  - MobileNetV3-Small
  - CNN

## Repository Scope

This repository is intentionally minimal and publication-ready.

Included:
- core MoE training pipeline
- RFPrompt adaptation path
- Frozen/PFT/RFPrompt drivers
- conventional baseline benchmark script
- metrics aggregation scripts

Excluded:
- datasets
- checkpoints
- large outputs/logs

## Project Structure

- `MoE/` - router + expert loading + task training
- `task2/mobility_utils.py` - RFPrompt wrapper and model preparation helpers
- `task1/train_mcs_models.py` - utility stubs used by MoE pipeline
- `pretraining/pretrained_model.py` - LWM model definition
- `drivers/` - MoE experiment entrypoints
  - `run_frozen.py`
  - `run_pft.py`
  - `run_rfprompt.py`
- `scripts/`
  - `run_stage_a_sweep.sh`
  - `collect_stage_a_sweep_metrics.py`
  - `train_baseline_benchmarks.py`

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Data / Checkpoint Layout

Expected expert checkpoints:
- `experts/LTE_expert.pth`
- `experts/WiFi_expert.pth`
- `experts/5G_expert.pth`

Expected spectrogram datasets:
- `datasets/spectrograms_ieee/city_ieee_dataport/...`
- `datasets/spectrograms_realworld_iq/city_realworld_iq/...`

## Raw IQ -> Spectrogram Conversion

If you start from raw IQ data, use these scripts to generate the folder layout expected by training:

- `scripts/iq_to_spectrogram.py` - reusable IQ-to-spectrogram utility
- `scripts/export_ieee_to_spectrograms_dir.py` - export IEEE Dataport `.pt` to spectrogram folders
- `scripts/export_realworld_iq_to_spectrograms_dir.py` - export Real-World IQ `subset_train.h5` to spectrogram folders

Examples:

```bash
# IEEE Dataport prepared .pt -> spectrogram directory
python scripts/export_ieee_to_spectrograms_dir.py \
  --input datasets/prepared/ieee_dataport.pt \
  --output datasets/spectrograms_ieee \
  --city city_ieee_dataport

# Real-World IQ HDF5 -> spectrogram directory
python scripts/export_realworld_iq_to_spectrograms_dir.py \
  --input-root datasets/realworld_iq_raw/dataset \
  --output datasets/spectrograms_realworld_iq \
  --city city_realworld_iq
```

## A) MoE Runs (Frozen / PFT / RFPrompt)

```bash
export PYTHONPATH="$PWD"
export MOD_MOE_DATASET=ieee
export MOD_MOE_DATA_ROOT="$PWD/datasets/spectrograms_ieee"
export MOD_MOE_TASK_EPOCHS=30
export MOD_MOE_BATCH_SIZE=32
export MOD_MOE_MAX_SAMPLES_PER_CLASS=400
```

```bash
python drivers/run_frozen.py ieee
python drivers/run_pft.py ieee
python drivers/run_rfprompt.py ieee
```

Outputs are written under `outputs/<run_name>/` with `metrics.json`, `training_history.json`, and `training_metrics.csv`.

## B) Stage-A Sweep

```bash
bash scripts/run_stage_a_sweep.sh ieee
python scripts/collect_stage_a_sweep_metrics.py --root outputs --epochs 100
python scripts/collect_stage_a_sweep_metrics.py --root outputs --epochs 100 
```

## C) Conventional Baseline Benchmarks (From Scratch)

The script below trains **from scratch** (`weights=None`) on few-shot splits.
These runs are used to populate the paper's Table IV baseline rows.

```bash
python scripts/train_baseline_benchmarks.py \
  --data-root datasets/spectrograms_ieee \
  --city city_ieee_dataport \
  --shots 0 2 4 8 16 32 64 128 \
  --models resnet18 efficientnet_b0 mobilenet_v3_small cnn \
  --epochs 30 \
  --batch-size 32 \
  --output-dir outputs/baseline_benchmarks_ieee
```

Result file:
- `outputs/baseline_benchmarks_ieee/baseline_benchmarks_summary.json`

## Dataset Sources (Full Downloads)

- M. Girmay and A. Shahid, "Dataset: IQ samples of LTE, 5G NR, Wi-Fi, ITS-G5, and C-V2X PC5," 2023. [Online]. Available: [https://dx.doi.org/10.21227/72qq-z464](https://dx.doi.org/10.21227/72qq-z464)
- N. Belousov and M. Ronkin, "Real-world IQ dataset for automatic radio modulation recognition under multipath channels," 2026. [Online]. Available: [https://data.mendeley.com/datasets/tjzsbph49x/2](https://data.mendeley.com/datasets/tjzsbph49x/2)

