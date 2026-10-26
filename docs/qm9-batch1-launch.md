# Preparing and launching the six-model batch

Source branch: `experiments/qm9-structural-priors-batch1`, based on `year2`.
Code is developed locally, committed to the user's DeFoGPlus fork, and fetched
on Campus Cluster. Original model modules are unchanged; the experiment layer
is in `src/experiments/batch1`. The agreed protocol is in
[`qm9-batch1-plan.md`](qm9-batch1-plan.md).

Local HTTPS GitHub push authentication was unavailable during preparation;
the cluster's existing SSH GitHub authentication works. Commits can be sent
locally with `git push xinyan14-cc:/u/xinyan14/motiflow
HEAD:refs/heads/sync/batch1-fixes`, then fast-forwarded into the clean experiment
branch on the cluster and pushed to `origin` there. This still transfers source
entirely through Git. Never copy uncommitted source files over the checkout.

## Before submission

Use `ssh xinyan14-cc`, then work in `/u/xinyan14/motiflow`.
The existing locked benchmark environment and processed QM9 files are reused.
No installation, new data download or GPU allocation is needed to stage code.

```bash
python3 scripts/batch1_cluster.py stage \
  --batch-root /scratch/xinyan14/motiflow-experiments/qm9-batch1/batch-YYYYMMDDTHHMMSSZ
```

Staging creates an immutable `git archive` of HEAD, its `SOURCE_COMMIT` marker,
an environment-path file without credentials, and `launch.json`. It submits
nothing. The batch directory must be new; do not overwrite previous attempts.
Inspect the plan with the following dry run:

```bash
python3 scripts/batch1_cluster.py submit --batch-root /absolute/batch/path
```

When authorized to launch, add `--submit` to that exact command. Default array
concurrency is two A100s; `--concurrency 1` serializes the six configurations.
The helper records job IDs after each submission and rejects duplicates.

## Jobs created by an explicit submission

1. CPU prepare: mathematical/metric unit tests, train-only cached statistics,
   and all six full-architecture CPU smoke tests including pause/resume.
2. Dependent A100 array: B0, B1, A0, A1, M0, M1. Each allocation has one A100,
   eight CPU cores, 32 GiB RAM and an eight-hour cap. Each task runs a tiny GPU
   smoke test first, then trains its configuration to 1000 epochs. A failed
   smoke test prevents that task's long run.
3. Dependent CPU sync: mirrors the explicitly named offline scalar/config W&B
   sessions and reconciles GPU accounting, even if an array task failed.

The six caps total 48 GPU-hours, plus the previous pilot's 0.516 hours. CPU
checks do not use GPUs. The expected total remains about 36 GPU-hours; the caps
are limits, not expected consumption. There are no automatic retries or extra
seeds. `budget-ledger.json` uses `sacct` for every submitted array task; inspect
all failures and reconcile that ledger before arranging any continuation.

Slurm warns three minutes before timeout. The Python runner saves at the next
epoch boundary and exits paused (75). Selected/recovery models are backed up
after each checkpoint. The runner accepts `--resume latest`; continuation must
be separately budgeted, not submitted by a retry loop. Interrupted final
sampling may be rerun from the retained final checkpoint.

## Artifacts and observability

The staged directory contains `runs/<configuration>-seed0/` with:

- `manifest.json`, resolved config, per-attempt provenance and checkpoint index;
- append-only `metrics/{train,validation,curve,final,events}.jsonl`, CSV exports
  and final `metrics/summary.json`;
- atomic full checkpoints at all 17 curve epochs, plus latest/previous rolling
  recovery checkpoints every 25 epochs, with best/final index pointers;
- raw generated graphs, initial atoms, strict/legacy metrics, canonical SMILES,
  composition diagnostics and small aggregate trajectories under `samples/`;
- offline W&B sessions and scheduler/GPU-utilization logs.

The private file `~/.config/defog/wandb-env.sh` is sourced without shell tracing;
its contents are never committed or copied into manifests. W&B project:
`defog-qm9-batch1`, entity:
`maxlyu254-university-of-illinois-urbana-champaign`. Local records remain
authoritative if cloud sync fails. Checkpoints, raw samples and datasets are
not uploaded to W&B. Interrupted offline attempts have separate session IDs
and a common logical-run group.

Working files and prior caches live in `/scratch/xinyan14/motiflow-experiments/`.
Selected/recovery checkpoints and metric copies live in
`/u/xinyan14/motiflow-artifacts/qm9-batch1/<batch>/<configuration>-seed0/`.
Reserve 30–40 GiB scratch space excluding the existing environment and dataset;
selected/recovery home backups require additional space. These files stay out
of Git. Only code, source configs, Slurm scripts, tests, plans and curated small
reports belong in Git.

```bash
python3 scripts/batch1_cluster.py status --batch-root /absolute/batch/path
squeue -u xinyan14
sacct -j ARRAY_JOB_ID --format=JobID,State,Elapsed,AllocTRES,ExitCode
```

Array disappearance from `squeue` alone does not establish success. Confirm
`COMPLETED` in `sacct`, exit status zero, and each run's completed summary at
epoch 1000. CPU sync has its own success/failure status.

## Statistical details fixed for this batch

Degree weights use training-only represented neighbor counts, smoothed with
100 pseudoatoms toward the global atom-type mean. Density estimates use 20
pseudographs per size. Unobserved composition sizes fall back to independent
original atom marginals; counts and smoothing are recorded in the cache.
Existing-bond types use the original baseline marginals for every variant.
Reference identities for M0/M1 are independent of the clean training target
conditional on N. Their auxiliary features persist throughout sampling; Q is
computed once. Fixed A0/A1 atoms incur no node-prediction loss.

Curve generation uses common fixed seeds for size draws and reference profiles
where appropriate. All final repetitions use fresh seeds. Every evaluation
records actual completed epoch and a model-weight hash. Evaluation RNG streams
are isolated from training. Loader shuffling is deterministic by completed
epoch and seed; full checkpoints include RNG and optimizer/loop state.
