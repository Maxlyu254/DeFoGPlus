#!/bin/bash -l
#SBATCH --job-name=defog-B1-seed1-finished
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --mail-user=xinyan14@illinois.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
: "${CONTINUATION_ROOT:?Pass CONTINUATION_ROOT}"
source /scratch/xinyan14/motiflow-experiments/qm9-batch3/batch3-matrix-20261108/environment.sh
set -a
source "$HOME/.config/defog/wandb-env.sh"
set +a
export WANDB_MODE=online
"$BATCH1_PYTHON" "$CONTINUATION_ROOT/source/scripts/continue_b1.py" sync --continuation-root "$CONTINUATION_ROOT"
