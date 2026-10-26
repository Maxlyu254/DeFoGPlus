#!/bin/bash -l
#SBATCH --job-name=defog-prepare
#SBATCH --account=xinyan14-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
: "${BENCH_RUN_ROOT:?Pass BENCH_RUN_ROOT when submitting}"
module load miniforge3/25.9.1
export CONDA_PKGS_DIRS=/scratch/xinyan14/motiflow-benchmarks/conda-pkgs
export PIP_CACHE_DIR=/scratch/xinyan14/motiflow-benchmarks/pip-cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=disabled PYTHONUNBUFFERED=1
mkdir -p "$BENCH_RUN_ROOT/setup" "$BENCH_RUN_ROOT/results"
source_root="$BENCH_RUN_ROOT/source"
env_prefix="${BENCH_ENV_PREFIX:-$BENCH_RUN_ROOT/environment}"
python "$source_root/benchmarks/locked_environment.py" "$source_root/pixi.lock" "$BENCH_RUN_ROOT/setup"

trap 'code=$?; if (( code != 0 )); then printf "setup failed: exit %s\n" "$code" > "$BENCH_RUN_ROOT/setup/FAILED"; fi' EXIT
if [[ ! -x "$env_prefix/bin/python" ]]; then
    mamba create --yes --prefix "$env_prefix" --file "$BENCH_RUN_ROOT/setup/conda-explicit.txt"
fi
# Pixi installs PyPI artifacts externally: its lock does not include pip or
# wheel in the environment. Bootstrap build tools before installing locked
# source distributions with --no-build-isolation.
"$env_prefix/bin/python" -m ensurepip --upgrade
"$env_prefix/bin/python" -m pip install --no-deps 'pip==25.1.1' 'setuptools==68.0.0' 'wheel==0.45.1'
printf '%s\n' 'Installer bootstrap: pip=25.1.1 setuptools=68.0.0 wheel=0.45.1 (outside Pixi runtime lock).' > "$BENCH_RUN_ROOT/setup/installer-bootstrap.txt"
"$env_prefix/bin/python" -m pip install --no-deps --no-build-isolation -r "$BENCH_RUN_ROOT/setup/pip-locked.txt"
"$env_prefix/bin/python" -m pip install --no-deps --no-build-isolation --editable "$source_root"

# Some older torch builds cannot bridge NumPy 2. Only apply this recorded
# compatibility fallback if that exact failure is observed in the import check.
import_check='import graph_tool, torch, numpy, rdkit, torch_geometric, pytorch_lightning; a=numpy.array([1.0]); assert torch.from_numpy(a).numpy()[0] == 1.0; print(torch.__version__, torch.version.cuda, numpy.__version__, rdkit.__version__, pytorch_lightning.__version__)'
if ! "$env_prefix/bin/python" -c "$import_check" > "$BENCH_RUN_ROOT/setup/import-check.log" 2>&1; then
    cat "$BENCH_RUN_ROOT/setup/import-check.log"
    if grep -Eq 'Numpy is not available|_ARRAY_API|compiled using NumPy 1' "$BENCH_RUN_ROOT/setup/import-check.log"; then
        "$env_prefix/bin/python" -m pip install --no-deps 'numpy==1.26.4'
        printf '%s\n' 'NumPy 1.26.4 substituted after verified torch/NumPy ABI failure; see import-check.log.' > "$BENCH_RUN_ROOT/setup/environment-deviation.txt"
        "$env_prefix/bin/python" -c "$import_check" > "$BENCH_RUN_ROOT/setup/import-check-after-fallback.log" 2>&1
    else
        exit 1
    fi
fi
cat "$BENCH_RUN_ROOT/setup/import-check.log"
"$env_prefix/bin/python" -m pip freeze > "$BENCH_RUN_ROOT/setup/pip-freeze.txt"
conda list --prefix "$env_prefix" --explicit > "$BENCH_RUN_ROOT/setup/conda-installed-explicit.txt"
/usr/bin/time -v -o "$BENCH_RUN_ROOT/setup/data-preparation.time" \
    "$env_prefix/bin/python" "$source_root/benchmarks/budget_qm9.py" prepare \
    --data-dir /scratch/xinyan14/motiflow-benchmarks/data/qm9 \
    --output "$BENCH_RUN_ROOT/setup/data"
test -f "$BENCH_RUN_ROOT/setup/data/READY"
"$env_prefix/bin/python" "$source_root/benchmarks/budget_qm9.py" preflight \
    --data-dir /scratch/xinyan14/motiflow-benchmarks/data/qm9 \
    --output "$BENCH_RUN_ROOT/setup/preflight"
test -f "$BENCH_RUN_ROOT/setup/preflight/READY"
printf '%s\n' "$env_prefix" > "$BENCH_RUN_ROOT/setup/environment-prefix.txt"
printf '%s\n' 'environment, data, metric self-test, CPU train/sample/checkpoint preflight ready' > "$BENCH_RUN_ROOT/setup/READY"
