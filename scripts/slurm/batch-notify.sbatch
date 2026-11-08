#!/bin/bash -l
#SBATCH --job-name=defog-batch-finished
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --mail-user=xinyan14@illinois.edu
#SBATCH --mail-type=END,FAIL
set -euo pipefail
: "${BATCH1_ROOT:?Pass BATCH1_ROOT}"
source "$BATCH1_ROOT/environment.sh"
cd "$BATCH1_ROOT"
# Slurm sends the email when this final CPU job exits; no mail credentials or
# custom mail delivery are needed. The manager supplies its afterany dependency.
"$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/finalize_batch.py" --batch-root "$BATCH1_ROOT"
