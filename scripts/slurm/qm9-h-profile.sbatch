#!/bin/bash -l
#SBATCH --job-name=defog-h-profile
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
source "${BATCH1_ROOT:?}/environment.sh"
cd "$BATCH1_ROOT"
test -f "$BATCH1_ROOT/READY"
run_root="$BATCH1_ROOT/runs/Hprofile-seed0"
mkdir -p "$run_root/logs"
set -a
source "$HOME/.config/defog/wandb-env.sh"
set +a
: "${WANDB_API_KEY:?Missing W&B authentication}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTHONPATH="$BATCH1_SOURCE:$BATCH1_SOURCE/src"
export BATCH1_GPU_HEALTH_LOG="$run_root/logs/gpu-${SLURM_JOB_ID}.txt"
export BATCH1_GPU_FAILURE_MARKER="$run_root/logs/gpu-health-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
nvidia-smi -q > "$BATCH1_GPU_HEALTH_LOG" 2>&1 || true
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 10 > "$run_root/logs/gpu-monitor-${SLURM_JOB_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
srun --ntasks=1 "$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/check_cuda.py"
# A manifest prevents retrying any partially executed resource profile.
"$BATCH1_PYTHON" -c 'import os,json;from pathlib import Path; p=Path(os.environ["BATCH1_ROOT"])/"runs/Hprofile-seed0/manifest.json"; p.write_text(json.dumps({"phase":"profiling"}))'
for variant in B0 M1; do
    for batch in 128 256 512 1024; do
        srun --ntasks=1 "$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/qm9_h_profile.py" probe \
            --variant "$variant" --kind train --batch-size "$batch"
    done
    for batch in 256 512 1024 2048; do
        srun --ntasks=1 "$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/qm9_h_profile.py" probe \
            --variant "$variant" --kind sample --batch-size "$batch"
    done
done
"$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/qm9_h_profile.py" select
for variant in B0 M1; do
    srun --ntasks=1 "$BATCH1_PYTHON" "$BATCH1_SOURCE/scripts/qm9_h_profile.py" profile --variant "$variant"
done
