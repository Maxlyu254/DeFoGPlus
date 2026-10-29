# Batch 2: authorized fixed-1000 replication

User instruction: exactly 1000 epochs, no drift checks or adaptive extension,
and queue all seven tasks together without a user-imposed concurrency throttle.
This supersedes the convergence proposal. Submission is authorized.

| Array task | Configuration | Training seed |
| --- | --- | --- |
| 0 | B0 | 1 |
| 1 | B0 | 2 |
| 2 | A0 | 1 |
| 3 | M0 | 1 |
| 4 | M0 | 2 |
| 5 | M1 | 1 |
| 6 | M1 | 2 |

Implementation: `configs/batch2.yaml`, `scripts/batch2_cluster.py`, and
`scripts/batch1.py --protocol batch2`, reusing the verified model/prior layer.
Batch1 defaults retain their historical protocol.

Batch 2 disables best-VUN selection and reports only the epoch-1000 checkpoint
with three fresh 10k repeats. Curves retain the original 17-point schedule;
epochs 400 through 1000 use 10k pools, with strict metrics for a saved 2k subset
for comparison with early/batch1 curves. Raw graphs, strict/legacy metrics,
structural diagnostics, train/validation losses and timings are retained.
No drift estimation or convergence-driven stopping is implemented.

Resources: one A100, eight cores, 32 GiB RAM and seven hours per task.
`--array=0-6` has no `%` throttle. No scheduler requeue or automatic retry.
Maximum GPU reservation is 49 hours within the fresh 50-hour budget; the
previous batch is not charged. Expected runtime is approximately 6-7 hours once
allocated. Scheduler availability/policy can limit simultaneous starts; no
completion deadline is guaranteed.

CPU preparation gates all seven tasks together after unit/cache/full-model
smoke-resume checks. Each GPU task runs a tiny GPU smoke, then its 1000-epoch
experiment. A dependent CPU job syncs W&B and updates accounting. All jobs are
submitted at once; training seeds are not queued in separate waves.

Staged root:
`/scratch/xinyan14/motiflow-experiments/qm9-batch2/batch2-fixed1000-20261028`.

- `launch.json`: immutable source commit, task mapping and job IDs.
- `budget-ledger.json`: fresh batch accounting.
- `source/`: Git archive used by jobs.
- `runs/<configuration>-seed<seed>/`: metrics, checkpoints, samples, GPU logs,
  offline W&B and provenance. Final endpoint: `metrics/summary.json`.
- `logs/`: Slurm stdout/stderr.

Backups: `/u/xinyan14/motiflow-artifacts/qm9-batch2/batch2-fixed1000-20261028/`.
Reuse the verified training-only batch1 prior cache. Source credentials privately
from `~/.config/defog/wandb-env.sh`; no credentials or runtime artifacts in Git.
W&B project: `defog-qm9-batch2`, existing authenticated entity.

```bash
python3 scripts/batch2_cluster.py stage --batch-root /absolute/new/batch/path
python3 scripts/batch2_cluster.py submit --batch-root /absolute/batch/path --submit
python3 scripts/batch2_cluster.py status --batch-root /absolute/batch/path
```

Development is local and synchronization uses Git. Submission records will be
added under `docs/submissions/`. Before retrying, inspect launch.json and sacct.
