#!/bin/bash -l
#SBATCH --job-name=defog-budget-a100
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${BENCH_RUN_ROOT:?Pass BENCH_RUN_ROOT when submitting}"
test -f "$BENCH_RUN_ROOT/setup/READY"
env_prefix=$(cat "$BENCH_RUN_ROOT/setup/environment-prefix.txt")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTHONPATH="$BENCH_RUN_ROOT/source:$BENCH_RUN_ROOT/source/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$env_prefix/bin:$PATH"
mkdir -p "$BENCH_RUN_ROOT/results"
date -u +%FT%TZ > "$BENCH_RUN_ROOT/results/STARTED"
hostname > "$BENCH_RUN_ROOT/results/hostname.txt"
nvidia-smi -q > "$BENCH_RUN_ROOT/results/gpu-details.txt"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$BENCH_RUN_ROOT/results/gpu-monitor.csv" &
monitor_pid=$!
finish() {
    code=$?
    kill "$monitor_pid" 2>/dev/null || true
    printf '%s\n' "$code" > "$BENCH_RUN_ROOT/results/exit-code.txt"
    date -u +%FT%TZ > "$BENCH_RUN_ROOT/results/FINISHED"
    if (( code != 0 )); then printf 'exit %s\n' "$code" > "$BENCH_RUN_ROOT/results/FAILED"; fi
}
trap finish EXIT
trap 'date -u +%FT%TZ > "$BENCH_RUN_ROOT/results/TIME_LIMIT_WARNING"' USR1
/usr/bin/time -v -o "$BENCH_RUN_ROOT/results/process.time" \
    srun --ntasks=1 "$env_prefix/bin/python" "$BENCH_RUN_ROOT/source/benchmarks/budget_qm9.py" run \
    --data-dir /scratch/xinyan14/motiflow-benchmarks/data/qm9 \
    --output "$BENCH_RUN_ROOT/results" --epochs 20 --seed 0
test -f "$BENCH_RUN_ROOT/results/summary.json"
