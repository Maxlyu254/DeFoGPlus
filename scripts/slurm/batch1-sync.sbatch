#!/bin/bash -l
#SBATCH --job-name=defog-batch1-sync
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
set -euo pipefail
: "${BATCH1_ROOT:?Pass BATCH1_ROOT}"
source "$BATCH1_ROOT/environment.sh"
set -a
source "$HOME/.config/defog/wandb-env.sh"
set +a
export WANDB_MODE=online
# Only the explicitly named scalar/config offline sessions are synced.
# Model/data/sample binaries are never registered as W&B artifacts.
"$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/batch1_cluster.py" sync --batch-root "$BATCH1_ROOT"
