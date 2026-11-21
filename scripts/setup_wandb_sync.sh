#!/bin/bash
# Separate from the locked training environment. No credentials are read here.
set -euo pipefail
reader_python="${BATCH1_PYTHON:-/scratch/xinyan14/motiflow-benchmarks/qm9-a100-20261026T041904Z/environment/bin/python}"
sync_root=/scratch/xinyan14/motiflow-tools/wandb-sync/0.31.2
if [[ ! -x "$sync_root/bin/python" ]]; then
    "$reader_python" -m venv "$sync_root"
fi
"$sync_root/bin/python" -m pip install --disable-pip-version-check wandb==0.31.2
"$sync_root/bin/python" -m pip freeze > "$sync_root/installed-requirements.txt"
