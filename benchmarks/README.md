# Vanilla DeFoG budget harness

The model source and presets are unchanged. `budget_qm9.py` constructs the same
QM9 model directly, measuring dataset setup, full-data training with 0 and 4
workers, checkpoint save/load/resume, generation, serialization, strict connected
VUN, legacy relaxed metrics, and two integrated evaluation epochs.

The harness deliberately omits `compute_reference_metrics()` over the entire
training dataset: the resulting reference ratios are unused for this molecular
timing workload. That omitted startup work is explicitly recorded. Full dataset
download, processing, and train/val/test SMILES caching still occur in the CPU
preparation job.

`locked_environment.py` exports exact Linux artifact URLs from the repository's
Pixi lock; installation uses the same Conda packages and PyPI artifacts without
a new dependency solve. This lock uses PyTorch 2.4.0/CUDA 11.8 and Lightning 2.5.2,
which differ from some versions in environment.yaml. Installed package lists
and any narrowly triggered NumPy ABI compatibility fallback are recorded.

Pixi's runtime lock does not contain an in-environment pip or wheel installer.
Preparation explicitly bootstraps pip 25.1.1, setuptools 68.0.0, and wheel 0.45.1
before installing locked PyPI artifacts. These installer additions are recorded
separately. Conda explicit exports include the lock's MD5 checksums. Set
`BENCH_ENV_PREFIX` on the CPU preparation submission to resume installation in
an existing environment; never share it between simultaneously running setup
or benchmark jobs.

Slurm scripts expect an isolated, immutable source snapshot at
`$BENCH_RUN_ROOT/source` and logs outside that source tree. Submit CPU preparation
first, then the A100 job with an `afterok` dependency. A100 reservation: one GPU,
8 CPU cores, 64 GiB host RAM, four-hour limit. CPU preparation has a three-hour
limit and consumes no GPU. No A10 job is part of this submission.

Artifacts:

- `setup/READY` means imports, QM9 preparation, and the strict metric self-test
  succeeded; `setup/FAILED` signals failure.
- `results/events.jsonl` is the incremental timing/metric record.
- `results/summary.json` exists only when every benchmark phase completes.
- `results/FAILED` or a nonzero Slurm exit code signals failure. A time limit can
  leave partial events; never interpret disappearance from squeue as success.
- `results/gpu-monitor.csv`, checkpoint files, raw sample pickles, strict metric
  JSON, legacy metric logs, and `/usr/bin/time` reports support later analysis.

Stage order: 20 epochs/workers=0; 20 epochs/workers=4; checkpoint loading; 128
warmup samples; 2k and 10k generation at inference batch 2048; two more 2k repeats;
2k at batches 256/512/1024; resume the faster worker configuration for two epochs,
each with 2k sampling and checkpoint writing. All sampling uses 500 steps. Pilot
quality scores describe a 20-epoch checkpoint, not a fully trained model.

The metric self-test uses known valid, duplicate, novel, disconnected, and
overvalent carbon graphs, and verifies VUN from counts. Static checks should
include Python compilation, shell syntax, and lock export before submission.
Runtime import/self-tests execute in the CPU job before the A100 dependency can
run. A CPU preflight additionally runs one optimizer step, saves/reloads a
checkpoint, and generates four samples. Only this smoke test uses batch 4 and
two sampling steps; actual GPU benchmarks retain batch 1024 and 500 steps.
Loader tuning, timer callbacks, and separate metrics do not alter the prior,
network, precision, optimizer, or sampling transition rules.
