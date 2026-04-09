#!/usr/bin/env bash
set -euo pipefail

# Stage-A sweep helper for paper tables.
# Usage:
#   bash scripts/run_stage_a_sweep.sh ieee
#   bash scripts/run_stage_a_sweep.sh realworld_iq

DATASET="${1:-ieee}"
EPOCHS="${MOD_MOE_TASK_EPOCHS:-100}"
CAPS=(100 200 400 800 1600)

export PYTHONPATH="${PYTHONPATH:-$PWD}"
export MOD_MOE_DATASET="$DATASET"

for cap in "${CAPS[@]}"; do
  echo "=== DATASET=${DATASET} N=${cap} mode=frozen ==="
  MOD_MOE_MAX_SAMPLES_PER_CLASS="$cap" \
  MOD_MOE_RUN_NAME="mod_class_moe_${DATASET}_${EPOCHS}ep_N${cap}_frozen" \
  python drivers/run_frozen.py "$DATASET"

  echo "=== DATASET=${DATASET} N=${cap} mode=pft ==="
  MOD_MOE_MAX_SAMPLES_PER_CLASS="$cap" \
  MOD_MOE_RUN_NAME="mod_class_moe_${DATASET}_${EPOCHS}ep_N${cap}_pft" \
  python drivers/run_pft.py "$DATASET"

  echo "=== DATASET=${DATASET} N=${cap} mode=rfprompt ==="
  MOD_MOE_MAX_SAMPLES_PER_CLASS="$cap" \
  MOD_MOE_RUN_NAME="mod_class_moe_${DATASET}_${EPOCHS}ep_N${cap}_rfprompt" \
  python drivers/run_rfprompt.py "$DATASET"
done

echo
echo "Sweep complete. Aggregate with:"
echo "python scripts/collect_stage_a_sweep_metrics.py --root outputs --epochs ${EPOCHS}"
