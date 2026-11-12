#!/bin/bash -l
#SBATCH --job-name=defog-B1-seed1-resume
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
set -euo pipefail
: "${CONTINUATION_ROOT:?Pass CONTINUATION_ROOT}"
batch_root="/scratch/xinyan14/motiflow-experiments/qm9-batch3/batch3-matrix-20261108"
source "$batch_root/environment.sh"
run_root="$batch_root/runs/B1-seed1"
set -a
source "$HOME/.config/defog/wandb-env.sh"
set +a
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$BATCH1_SOURCE:$BATCH1_SOURCE/src" WANDB_MODE=offline PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export BATCH1_JOB_START_UNIX
BATCH1_JOB_START_UNIX=$(date +%s)
export BATCH1_GPU_HEALTH_LOG="$CONTINUATION_ROOT/logs/gpu-${SLURM_JOB_ID}.txt"
nvidia-smi -q > "$BATCH1_GPU_HEALTH_LOG" 2>&1 || true
"$BATCH1_PYTHON" "$CONTINUATION_ROOT/source/scripts/continue_b1.py" verify --continuation-root "$CONTINUATION_ROOT"
srun --ntasks=1 "$BATCH1_PYTHON" "$CONTINUATION_ROOT/source/scripts/check_cuda.py"
checkpoint=$("$BATCH1_PYTHON" -c 'import json,os; print(json.load(open(os.environ["CONTINUATION_ROOT"]+"/continuation.json"))["checkpoint"])')
trap 'scancel --signal=USR1 "${SLURM_JOB_ID}.1" || true' USR1
srun --ntasks=1 "$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/batch1.py" train \
    --protocol batch3 --configuration B1 --seed 1 --device cuda --threads 4 \
    --data-dir "$BATCH1_DATA" --cache-dir "$BATCH1_CACHE" --run-dir "$run_root" \
    --archive-root "$BATCH1_ARCHIVE" --wandb-mode offline --resume "$checkpoint" &
child=$!
while true; do
    set +e
    wait "$child"
    result=$?
    set -e
    if kill -0 "$child" 2>/dev/null; then continue; fi
    exit "$result"
done
