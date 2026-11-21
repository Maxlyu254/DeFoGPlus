#!/bin/bash -l
#SBATCH --job-name=defog-h-prepare
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source "${BATCH1_ROOT:?}/environment.sh"
cd "$BATCH1_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTHONPATH="$BATCH1_SOURCE:$BATCH1_SOURCE/src"
"$BATCH1_PYTHON" -m unittest discover -s "$BATCH1_SOURCE/tests" -p 'test_h_profile.py' -v
"$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/qm9_h_profile.py" prepare
test -f "$BATCH1_ROOT/READY"
