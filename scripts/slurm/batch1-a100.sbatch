#!/bin/bash -l
#SBATCH --job-name=defog-batch1
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
set -euo pipefail
: "${BATCH1_ROOT:?Pass BATCH1_ROOT}"
source "$BATCH1_ROOT/environment.sh"
cd "$BATCH1_ROOT"
test -f "$BATCH1_ROOT/READY"
variants=(B0 B1 A0 A1 M0 M1)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
run_root="$BATCH1_ROOT/runs/$variant-seed0"
mkdir -p "$run_root/logs"
# Never enable shell tracing here: credentials stay private and out of logs.
set -a
source "$HOME/.config/defog/wandb-env.sh"
set +a
: "${WANDB_API_KEY:?Missing W&B authentication}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHONPATH="$BATCH1_SOURCE:$BATCH1_SOURCE/src"
export BATCH1_JOB_START_UNIX
BATCH1_JOB_START_UNIX=$(date +%s)
nvidia-smi -q > "$run_root/logs/gpu-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 10 > "$run_root/logs/gpu-monitor-${SLURM_JOB_ID}.csv" &
monitor_pid=$!
step=0
forward_warning() {
    scancel --signal=USR1 "${SLURM_JOB_ID}.${step}" || true
}
finish() {
    code=$?
    kill "$monitor_pid" 2>/dev/null || true
    printf '%s\n' "$code" > "$run_root/logs/exit-${SLURM_JOB_ID}.txt"
}
trap finish EXIT
trap forward_warning USR1
run_step() {
    srun --ntasks=1 "$@" &
    child=$!
    # wait can be interrupted by the scheduler warning; keep waiting for the
    # Python process to save at its next epoch boundary and exit itself.
    while true; do
        set +e
        wait "$child"
        result=$?
        set -e
        if kill -0 "$child" 2>/dev/null; then continue; fi
        return "$result"
    done
}
# Each long run first exercises the full architecture, backward pass, GPU
# sampler, checkpointing and strict/legacy evaluators on a tiny two-epoch pool.
run_step "$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/batch1.py" train --smoke \
    --configuration "$variant" --device cuda --threads 4 --wandb-mode disabled \
    --data-dir "$BATCH1_DATA" --cache-dir "$BATCH1_CACHE" \
    --run-dir "$BATCH1_ROOT/gpu-smoke/$variant-${SLURM_JOB_ID}"
step=1
run_step "$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/batch1.py" train \
    --configuration "$variant" --device cuda --threads 4 --wandb-mode offline \
    --data-dir "$BATCH1_DATA" --cache-dir "$BATCH1_CACHE" \
    --run-dir "$run_root" --archive-root "$BATCH1_ARCHIVE"
