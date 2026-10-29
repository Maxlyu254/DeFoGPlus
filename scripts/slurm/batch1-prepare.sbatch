#!/bin/bash -l
#SBATCH --job-name=defog-batch1-prepare
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
: "${BATCH1_ROOT:?Pass BATCH1_ROOT}"
source "$BATCH1_ROOT/environment.sh"
cd "$BATCH1_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTHONPATH="$BATCH1_SOURCE:$BATCH1_SOURCE/src"
"$BATCH1_PYTHON" -m unittest discover -s "$BATCH1_SOURCE/tests" -p 'test_batch1.py' -v
"$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/batch1.py" prepare \
    --protocol "${BATCH1_PROTOCOL:-batch1}" \
    --data-dir "$BATCH1_DATA" --cache-dir "$BATCH1_CACHE"
"$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/batch1.py" preflight \
    --protocol "${BATCH1_PROTOCOL:-batch1}" \
    --data-dir "$BATCH1_DATA" --cache-dir "$BATCH1_CACHE" \
    --run-dir "$BATCH1_ROOT/cpu-preflight" --threads 4
test -f "$BATCH1_ROOT/cpu-preflight/READY.json"
date -u +%FT%TZ > "$BATCH1_ROOT/READY"
