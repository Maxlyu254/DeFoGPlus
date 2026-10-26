# Vanilla DeFoG budget benchmark on Campus Cluster

Recorded 2026-10-25; results update 2026-10-26 UTC. Status: CPU preparation and
A100 benchmark completed successfully, with measured results saved locally. The user
requested A100 execution in place of the initial A10-first proposal below.

## Measured result

See `benchmarks/results/qm9-a100-retry-20261026T042900Z/report.md` for the full
report, `analysis.json` for derived values, and `raw/` for original events,
summary, resolved configurations, metrics, GPU monitoring, and Slurm accounting.

- CPU preparation 11822422: COMPLETED / 0:0, 5 min 49 sec.
- A100 benchmark 11822489: COMPLETED / 0:0, 30 min 59 sec (0.516 GPU-hours).
- Actual GPU: A100-SXM4-80GB; float32, batch 1024, 9-layer vanilla QM9 model.
- Steady training: 22.49 s/epoch at 0 workers; 18.69 s/epoch at 4 workers.
- Sampling: 61.4 s/2k, 307.8 s/10k at 500 steps and batch cap 2048.
- Strict CPU chemistry metrics: about 0.53 s/2k, 2.66 s/10k. Legacy relaxed
  evaluator adds about 2.89 s/2k and 14.12 s/10k if also requested.
- Peak device memory observed by nvidia-smi: 9.10 GiB; job-step host MaxRSS:
  4.42 GiB. Checkpoint save/load/resume verified.
- 1000-epoch projection with 12 curve evaluations and 3 final evaluations:
  5.67 GPU-hours per seed, 17.0 GPU-hours for 3 seeds; initially reserve 8 hours
  per seed (24 hours across 3 reservations) to allow margin.
- Strict connected 20-epoch snapshot over 10k attempts: V=84.83%, U=97.95%,
  N=62.13%, VUN=51.62%. Legacy relaxed VUN=57.25% under a different policy.
- This is one training seed and an early checkpoint. Final 1000-epoch quality,
  ZINC/MOSES cost, FCD, and visualization overhead are unmeasured.
- Raw standalone generation checkpoint_age=0 is a metadata artifact after direct
  weight loading; the loaded checkpoint was trained for 20 epochs / 1920 steps.

## Submitted run

### Latest retry (2026-10-26 UTC)

Original job 11821406 failed in 3 minutes 2 seconds with
`environment/bin/python: No module named pip`. Pixi's lock lists runtime Conda
and PyPI packages, but Pixi itself supplies the installer; exporting the lock
does not put pip/wheel in the environment. A100 job 11821407 was automatically
canceled by its failed dependency and consumed no GPU time. Therefore no
`results/summary.json` could have been produced.

Corrections: explicit ensurepip/pip/setuptools/wheel bootstrap, Conda artifact
checksums in exports, and a CPU preflight covering an optimizer step, checkpoint
save/load, sampling, and strict metric counts before GPU eligibility.

- New CPU setup job: **11822422**.
- New dependent A100 job: **11822489**, same one-A100/four-hour resource request.
- Retry run root:
  `/scratch/xinyan14/motiflow-benchmarks/qm9-a100-retry-20261026T042900Z`.
- Reuses the original fully installed 8.8 GB Conda environment, without deleting
  old logs or source snapshots; bootstrap and Python-package installation
  resume there. No original jobs remain active against that environment.
- New setup has installed the Python packages and passed imports plus the
  torch/NumPy bridge check (torch 2.4.0.post301, CUDA 11.8, NumPy 2.3.1, RDKit
  2025.03.4, Lightning 2.5.2). Data/model/metric checks subsequently passed and
  the dependent A100 job completed every benchmark phase successfully.

```bash
ssh xinyan14-cc 'squeue -j 11822422,11822489'
ssh xinyan14-cc 'sacct -j 11822422,11822489 --format=JobID,State,Elapsed,ExitCode'
```

### Original submission (failed setup)

- CPU preparation: Slurm job **11821406**, IllinoisComputes, 8 cores, 64 GiB,
  three-hour limit. Exports/installs exact Pixi lock artifacts, checks imports
  and the NumPy bridge, prepares QM9/SMILES, and runs a strict-metric self-test.
- Benchmark: Slurm job **11821407**, IllinoisComputes-GPU, one full A100, 8 CPU
  cores, 64 GiB host RAM, four-hour limit; `afterok:11821406` dependency. It is
  canceled if preparation fails, avoiding allocation of an unprepared GPU.
- Remote run root:
  `/scratch/xinyan14/motiflow-benchmarks/qm9-a100-20261026T041904Z`.
- Immutable uploaded source snapshot under `source/`, checksum manifest in
  `defog-benchmark-submission.json`, and scheduler IDs in `submission.json`.
- Logs: `logs/setup-11821406.out/.err` and `logs/a100-11821407.out/.err`.
- Incremental records: `results/events.jsonl`; complete output:
  `results/summary.json`. An empty squeue is not proof of success; inspect sacct
  State/ExitCode, FAILED markers, and the summary file.
- Implemented phase order is recorded in `benchmarks/README.md`. Training is 20
  epochs each at 0 and 4 workers, with a two-epoch resume/integrated check.
- Deviations: unused full-training-set molecular reference-metric evaluation is
  omitted; cold data/SMILES setup is measured in the CPU job. Exact locked
  environment uses torch 2.4.0/CUDA 11.8 and Lightning 2.5.2. Any verified NumPy
  ABI fallback is explicitly logged. Legacy metric timing occurs on CPU inside
  the allocated GPU job, so its time counts as allocated GPU time.
- Static Python/shell/lock-export checks and uploaded source checksums passed.
  Dependency/import/data/runtime checks execute in the preparation job; GPU
  functionality is checked when the A100 job starts.

Check jobs with:

```bash
ssh xinyan14-cc 'squeue -j 11821406,11821407'
ssh xinyan14-cc 'sacct -j 11821406,11821407 --format=JobID,State,Elapsed,ExitCode'
```

The remaining sections preserve the original plan; the A100 submission above
supersedes the A10-first scheduling proposal, not the baseline model settings.

## Objective

Measure the resources for a vanilla QM9 DeFoG training run and its periodic/final
molecular evaluation. Produce a measured budget for one 1000-epoch seed, three
training seeds, and future matched ablations. Distinguish setup, training,
generation, molecular metrics, serialization, and Slurm queue delays.

Start with QM9 without H. Repeat short dataset-specific profiles for ZINC250k
and MOSES later; do not extrapolate their cost from QM9 node counts alone.

## Fixed reference workload

- Branch year2; record exact git commit and instrumentation diff.
- `+experiment=qm9_no_h dataset=qm9`, full model rather than debug experiment.
- 9 layers, shared marginal node/edge priors, existing RRWP/molecular features.
- Reference train batch 1024, optimizer/lr/default training distortion unchanged.
- Reference sampling: 500 steps, polydec, eta=0, omega=0, sample.search=False.
- One GPU, float32 initially; no mixed-precision/architecture/prior modifications.
- Fixed dataset splits and a recorded training seed.
- W&B disabled for primary timing; visualization/chain/movie generation disabled.
- Benchmark cold initialization and cached initialization separately.
- Keep correctness fixes and evaluator optimizations explicitly documented;
  distinguish unmodified evaluator timing from any optimized alternative.

## Proposed stages

### 0. Setup and correctness

Verify the current environment files (`pixi.lock`, `pixi.toml`, environment.yaml),
install/resolve one reproducible environment, and record package/build versions.
Check imports, CUDA, full-size forward/backward, sampling, checkpoint save and
resume. Prepare/cache QM9 on shared scratch and record cold download/processing
time, cached startup, dataset size, and preprocessing/reference-metric costs.

Use allocated compute resources for substantive processing and all GPU work.
Start with a 1-epoch full-model smoke run, not `+experiment=debug` (which changes
model size and sets CPU use). If batch 1024 OOMs, record the failure and profile
feasible batches; label these separately instead of silently changing baseline.

### 1. Pure training cost

On one A10, run 20 complete full-data epochs with sampling/validation disabled
for this stage. Separate the first two epochs as warmup and measure the remaining
18; extend the sample if timing remains unstable. Include the first two in total
cost records, but use steady-state timings for the projection.

Record examples/sec, optimizer steps/sec, epoch median/p90, GPU allocated and
reserved peak memory, host peak memory, utilization, and data loading stalls.
Add correct CUDA synchronization at coarse timing boundaries; do not synchronize
each step in a way that changes the baseline throughput.

Compare a small worker-count sweep (0 vs 4) using otherwise identical workloads.
Start with 8 CPU cores and 64 GiB host RAM as a reservation, then right-size from
measurements. Bound CPU math-library threads to prevent worker oversubscription.
Record both the literal-preset and selected practical loader configuration.

Measure saving and resuming one checkpoint, its size, startup overhead after
resume, and sufficient restart behavior to plan around partition walltimes.
Do not retain a checkpoint every epoch simply for timing.

### 2. Generation and metrics

Use a saved checkpoint and record its training age (pilot quality is not final
quality). Warm up generation separately (e.g. 128 molecules). At 500 steps:

- Measure 2000 attempts for the learning-curve workload; target three independent
  sampling repetitions if they fit the pilot reservation.
- Measure 10000 attempts directly once for the final-report workload; do not rely
  only on 5x extrapolation from the 2000-attempt measurement.
- Test a small inference-batch sweep against the reference policy; record GPU
  memory/throughput and keep graph-size distributions matched across cases.
- Time generation, serialization, strict acceptance/canonicalization,
  deduplication, novelty membership checks, and legacy relaxed metrics separately.
- Where practical, evaluate saved raw samples in a CPU-only job so metric time
  does not occupy an idle GPU. Include the extra startup/I/O in the comparison.
- Keep periodic visualizations disabled. Measure any desired artifact-production
  overhead separately.

Current evaluator V/U/N use inconsistent strict vs relaxed molecule sets; fix or
add the planned consistent VUN evaluator before quality interpretation. Retain
legacy scores/timing for compatibility. A pilot's valid-molecule fraction can
affect CPU metric cost, so remeasure metrics at later checkpoints.

### 3. Integrated verification and GPU comparison

Run a short integrated job with at least two scheduled 2000-attempt evaluations
and checkpoint writes. Verify that measured wall time matches the sum of the
separately measured phases; account for startup, logging, and synchronization.

If available, repeat representative training and evaluation workloads on one
full A100 with the same model, precision, and reference batch. Record actual
device name/memory and utilization. Do not treat a 20 GB H100 MIG slice as a full
H100. MIG/H200 are secondary options, not necessary for the first profile.

## Pilot reservation envelope

Proposed initial envelope (caps, NOT runtime estimates or measured costs):

- Setup/preprocessing: one CPU job, initially up to one hour; unresolved package
  installation/download time may require separate scheduling.
- A10 training/smoke/instrumentation: up to 2 allocated GPU-hours.
- A10 generation/metrics/integrated checks: up to 2 allocated GPU-hours.
- Optional full-A100 representative comparison: up to 2 allocated GPU-hours.

First A10 pass: at most 4 allocated GPU-hours under this proposal, plus CPU/setup.
With optional comparison: at most 6 allocated GPU-hours. Stop/record incomplete
stages at the cap and use measured throughput to revise the plan. These are not
promises that all stages will fit, and no jobs have yet been launched.

## Accounting and extrapolation

For one seed, use seconds consistently:

    total = train_startup + 1000 * steady_epoch_time
            + sum(curve_eval_time_per_checkpoint)
            + R * final_eval_10000_time
            + checkpoint_and_restart_overhead

The current learning-curve plan has 12 evaluation points including the untrained
epoch-0 snapshot: 0,1,5,10,25,50,100,200,400,600,800,1000. Use R=3 fresh final
sampling repetitions as a proposed reporting budget. Avoid double-counting warm
startup and checkpoint I/O already included in measured aggregate timings.
Project three independent training seeds by summing their complete workloads;
environment setup and shared dataset preprocessing are one-time costs.

Report allocated GPU-hours as GPUs * allocated elapsed hours, separately from
CPU-only metric core-hours. If metrics remain inside a GPU job, their elapsed
time counts toward allocated GPU-hours even if GPU utilization is zero.
Report queue delay separately; do not infer future queue times from one job.
Produce central and conservative projections using epoch/evaluation variability
and a clearly labeled scheduling margin (initially 20-30%, revised after data).
Do not claim a dollar price without an actual allocation/billing policy.

Record Slurm Submit/Start/End, Elapsed, State, ExitCode, AllocTRES, TotalCPU and
MaxRSS (including job steps); combine accounting with application timings and
sampled nvidia-smi. Some cluster accounting GPU metrics may be unavailable.

## Implementation details to audit first

- `train.num_workers` controls the actual loader. QM9 preset's
  `dataset.num_workers=16` is not the same setting; train default is 0.
- `sample()` currently uses inference batch `2 * train.batch_size`; do not
  change training batch just to tune inference. Add a benchmark-level explicit
  sampling-batch setting without changing the model or stochastic process.
- `general.name=debug` activates fast_dev_run; do not use it for performance.
- Existing checkpoint cadence is tied to validation/sample intervals; benchmark
  checkpoint saving explicitly when disabling validation.
- A `general.num_sample_fold` value is not an independent training repetition.
  Existing code generates a combined pool and evaluates slices; record true
  per-repeat seeds and do not silently pool uniqueness across repetitions.
- Benchmark direct sampler calls without visualization; setting output counts
  to zero alone may not disable every serialization/logging side effect.
- Novelty currently uses repeated membership checks against its reference;
  profile this cost. If switching to a canonical set or removing verbose output,
  label the optimization and confirm identical accepted molecules/counts.
- FCD is explicitly disabled in the molecular metrics implementation. Exclude
  it from the vanilla timing and budget it separately if enabled later.
- Environment compatibility/import fixes, if needed, must preserve model
  semantics and be recorded. The updated branch is not the original Python 3.9
  environment; do not silently mix environments when comparing hardware.

## Deliverable

Measured table per GPU: training batch/workers, seconds/epoch, examples/sec,
GPU/host memory, seconds to generate 2k/10k at 500 steps, CPU metric cost,
startup/restart/checkpoint cost, and extrapolated one-seed/three-seed GPU-hours.
Include a recommendation for GPU, loader settings, inference batch, evaluation
frequency, Slurm chunk length, and uncertainty. Confirm projections against a
short integrated run before committing to full training.
