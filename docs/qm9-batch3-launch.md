# Batch3: eight-run QM9 matrix completion

The user authorized submission with final batch email enabled on 2026-11-08.
Submitted: preparation **12262180**, eight-task A100 array **12262181**,
sync/retry controller **12262182**, final email **12262185**.
Executed immutable source: `19e128846be285bb5ad0d8cd6cae4c4688634182`.
Submission record: `docs/submissions/qm9-batch3-12262181.json`.
Check current scheduler state and launch.json before retries; do not duplicate.
This is the eight-run ordering agreed in `qm9-matrix-completion-plan.md`.

| Array task | Configuration | Seed | Purpose |
| --- | --- | --- | --- |
| 0 | B1 | 1 | Size-dependent prior replication |
| 1 | B1 | 2 | Size-dependent prior replication |
| 2 | A0 | 2 | Fixed-atom baseline replication |
| 3 | A1 | 1 | Atom-aware prior replication |
| 4 | A1 | 2 | Atom-aware prior replication |
| 5 | M0 | 2 | Replace startup failure |
| 6 | M1 | 1 | Replace startup failure |
| 7 | M1 | 2 | Replace startup failure |

All train for exactly 1000 epochs. `configs/batch3.yaml` inherits batch2 with
only the batch name and W&B project changed (`defog-qm9-batch3`). Same backbone,
optimizer, priors, checkpoint/evaluation schedule, fixed final selection, and
three fresh final 10k repeats. No drift checks or adaptive extensions.

One A100, eight cores, 32 GiB and seven hours per task. Array `0-7` has no
concurrency throttle. This batch uses the user's explicit 56-hour budget
exception, including any automatic replacements; historical batch2 charges stay
in their original ledger and are not silently erased or charged again here.
The initial requested time limits sum to 56 hours; actual completed/failed
allocation is reconciled before any replacements. At most two reset-startup
retries per task; replacements require available budget. Excluded nodes:
ccc0388 and ccc0390, plus any newly discovered failed hosts.

`scripts/batch3_cluster.py` uses the shared checked staging/submission/controller
implementation. Each staged source is an immutable Git archive. CPU preparation
runs unit tests and all-six full-model train/sample/resume checks before GPUs
can start. A GPU tensor/health check and model smoke gate each long run.

One final CPU notifier emails **xinyan14@illinois.edu**, after the final W&B
sync/controller including all automatic retry waves. Intermediate jobs have
mail disabled. Native Slurm email reports COMPLETE/INCOMPLETE and run counts;
full details are in batch-summary.json. See `batch-email-notifications.md` and
`gpu-health-retries.md` for interrupted-submission safeguards.

Staged root:
`/scratch/xinyan14/motiflow-experiments/qm9-batch3/batch3-matrix-20261108`.
Runs: `runs/<configuration>-seed<seed>/`, final `metrics/summary.json`.
Backups:
`/u/xinyan14/motiflow-artifacts/qm9-batch3/batch3-matrix-20261108/`.
Historical successful runs and failed-allocation evidence are preserved.
No D0/D1/A2 experiments are included; profiling is required before their full
training.

Verification before submission: 28 CPU unit/infrastructure tests and all-six
full-model train/sample/resume preflight checks passed under the cluster Python
environment. Evidence:
`/scratch/xinyan14/motiflow-experiments/qm9-batch3/development-checks/batch3-20261108/READY.json`.
Slurm confirms array0-7, seven hours per task, ccc0388/ccc0390 excluded,
GPU dependency afterok:12262180, controller afterany:12262181_*, final notifier
afterany:12262182, and MailUser=xinyan14@illinois.edu with MailType=END,FAIL.
All jobs were pending at initial verification. No artificial array throttle.
