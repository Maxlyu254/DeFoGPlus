# One completion notification per batch

Recipient confirmed by the user: **xinyan14@illinois.edu**.
Live Slurm configuration checked on 2026-11-08: `MailProg=/usr/bin/smail`.
No custom mail sender, mail credentials, or background monitoring process is
required. No jobs or test emails were submitted during implementation.

## Behavior

New batch2-style launches submit one small CPU notification job from
`scripts/slurm/batch-notify.sbatch`. Only this job has `--mail-type=END,FAIL`;
preparation, training arrays, retry arrays and sync/controllers explicitly
disable mail. The notifier initially depends on `afterany` of the sync job,
so it runs after completion even if some runs failed.

Before creating a replacement wave, the retry controller holds the same pending
notification job. After creating the replacement sync/controller, it changes
the notifier dependency to `afterany:<new-controller-id>` and releases it.
This keeps intermediate GPU failures and retry waves from generating batch-end
mail. There is one notification job for the whole batch, not one per task or
per retry. Initial sync and notification jobs are submitted held, recorded, then
released in order, preventing a startup race with the initial submission.

If the controller is interrupted while submitting a replacement or updating
the notifier dependency, it leaves `retry_intent` in launch.json and can leave
the notifier held. Reconcile those jobs before releasing it; no claim of full
completion is made while submission state is uncertain. Cancelling the notifier
itself prevents its normal completion notification.

## Email content and result files

Slurm supplies the standard job email. The final CPU script labels its job
`defog-COMPLETE-<completed>of<total>` or
`defog-INCOMPLETE-<completed>of<total>`, which appears in the Slurm job identity.
The actual standard email body is controlled by the cluster; the detailed
JSON report is not attached or sent as a custom message.

`scripts/finalize_batch.py` writes `<batch-root>/batch-summary.json` with run
counts/names, the final sync/controller state, GPU budget snapshot and stopped
retry reasons. Full success requires every run to have a completed summary at
the requested epoch and the final sync/controller to be COMPLETED. Training
failures or a failed W&B sync therefore produce INCOMPLETE, even though the
notification job itself can finish successfully. The notification job's Slurm
output contains the detailed report. An error in the notification script itself
uses Slurm's failure email.

## Configuration and checks

`scripts/batch2_cluster.py stage` now defaults to
`--mail-user xinyan14@illinois.edu`. Override that argument for another recipient
or pass `--mail-user ''` to disable notification at submission. The recipient
and notifier job ID are saved in launch.json. No existing immutable launches
were modified. The next eight-run batch runner should carry this same policy.

The standalone sbatch file must be submitted with an explicit final-controller
dependency and BATCH1_ROOT; using the manager is preferred because it updates
the dependency across retries automatically. Merely adding mail to the initial
GPU array would report its end before replacement waves finish.

Validation: 15 CPU infrastructure tests pass, including initial submission
ordering, mail only on the notifier, dependency changes across retry waves,
interrupted updates, and final success/failure counts. Python compilation and
shell syntax checks pass. End-to-end email delivery and live dependency changes
have not been exercised with a submitted job.

## Verified W&B synchronization (2026-11-21)

The original W&B0.20.1 CLI could print `done` and mark files synced after a
server-schema failure, so its exit status was insufficient. All current batch
managers and the B1 continuation now call `scripts/wandb_sync.py`.
Uploads/API reads use a separate pinned W&B0.31.2 environment at
`/scratch/xinyan14/motiflow-tools/wandb-sync/0.31.2`; run
`bash scripts/setup_wandb_sync.sh` once to provision it. The training environment
remains unchanged and only decodes local offline records for verification.

The helper compares every saved scalar history value at its original step
with the unsampled cloud history, checks exact row counts, and writes
`<offline-session>/verified-sync.json` only after success. It allows bounded
cloud-ingestion delay, ignores legacy `.synced` markers, and checks existing
cloud runs before uploading so repeated calls do not append duplicate history.
An existing but mismatched cloud run fails verification rather than receiving
a blind replay. New recovery projects are explicitly PRIVATE. A CLI zero exit
with missing history fails the controller and therefore yields INCOMPLETE mail.
No checkpoints or raw graph pools are registered as W&B artifacts.

For the already-queued explicit-H profile12622943, CPU sync12622945 and final
notifier12622951 were temporarily held, then released after the runtime
`BATCH1_MANAGER` was redirected to immutable operations source `ed094dc` and the
sidecar interpreter configured. The original model source5bea299, GPU job,
and notification dependency chain were preserved. Previous runtime settings
are backed up beside that operations archive; changes are recorded in launch.json.

Official references: [Slurm sbatch mail/dependency options](https://slurm.schedmd.com/sbatch.html)
and [scontrol hold, update and release](https://slurm.schedmd.com/scontrol.html).
