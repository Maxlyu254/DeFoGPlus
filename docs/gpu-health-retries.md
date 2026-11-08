# Automatic retries for GPU reset failures

Implemented for newly staged batch2-style experiments. No jobs were submitted
or existing immutable batches modified by this change. The next batch's runner
should reuse this controller; its eight-task plan is still a separate launch.

## Eligibility

The GPU script saves nvidia-smi health output, then runs the CUDA tensor check.
If that check fails with a CUDA runtime error AND the saved health output says
`GPU requires reset` or `GPU Recovery Action: Reset`, it writes a marker scoped
to the exact array/task and exits 86. An afterany CPU controller confirms the
task is FAILED with exit 86:0, validates the marker's array/task and hostname
against Slurm accounting, and verifies the run has no training manifest,
training history, checkpoint or final summary before retrying it.

An unclassified CUDA initialization error without reset evidence, incorrect GPU
type, missing dependency, training exception, OOM, timeout or cancellation is
not automatically retried. Mid-training hardware failures are also excluded:
automated checkpoint recovery requires a separate design. Historical failures
that occurred before these markers existed are not retroactively eligible.

## Scheduling and limits

After the current GPU array ends, its CPU sync/controller job submits an array
containing only eligible task indices. It keeps the same immutable source,
task mapping, seeds, epochs, run directories and scientific settings. Existing
logs are named by allocation ID, so previous failure evidence remains.

Failed hostnames are added to the batch's node exclusions. Newly staged batches
default to excluding ccc0388 and ccc0390, the two known affected nodes. Existing
queued tasks are not migrated; discovered exclusions apply to replacement
waves, and a later batch needs to carry any newly discovered exclusions forward.

Default maximum: **two replacements per task**, three total allocations.
At the end of a wave, actual allocated GPU-hours from every attempt plus any
prior-attempt charge are summed. Replacements are submitted only if their full
requested time limits fit under the batch ceiling. If the entire eligible
replacement set does not fit, it stops instead of arbitrarily choosing tasks.
The normal ceiling is 50 A100-hours; the next eight-run batch has the user's
separate budget exception, which its launch metadata must explicitly encode.

`launch.json` records retry_waves, excluded_nodes, retry_stopped and cumulative
retry accounting. `budget-ledger.json` includes all arrays. The controller uses
a filesystem lock and rereads state before decisions. It records retry_intent
before calling sbatch; if submission outcome is uncertain or the watcher cannot
be recorded, it stops for manual reconciliation rather than risking duplicate
GPU allocations. Incomplete scheduler accounting also stops automatic actions.

W&B sync waits until replacement waves finish or retries stop. Unresolved
failures cause the final CPU controller to exit nonzero after syncing available
results. There are no unconditional retry loops or automatic budget increases.

Slurm's afterany dependency runs after completion irrespective of success, and
an array can select a sparse set of task indices. See the official
[sbatch reference](https://slurm.schedmd.com/sbatch.html) and
[job-array documentation](https://slurm.schedmd.com/job_array.html).

## Use and validation

`scripts/batch2_cluster.py stage` now records `max_gpu_retries=2`. Set
`--max-gpu-retries 0` to disable it when staging a new batch. Existing launch
records without this field retain their original no-automatic-retry behavior.
No GPU job is submitted by staging, tests, or documentation changes. Initial
launch still requires the explicit `submit --submit` path and authorization.
Once launched, the recorded policy authorizes bounded health retries within
that batch's budget.

Validation: 11 CPU tests exercise mixed success/failure selection, exclusions,
active/completed replacement protection, retry caps, budget and prior charges,
incomplete accounting, interrupted submission, stale markers, existing training
artifacts, disabled policy, all-wave ledger totals, and the real CUDA marker
script with a simulated torch module. Shell syntax checks pass. No live
failure/retry cycle was induced and no GPU resources were requested for testing.
