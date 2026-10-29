# Batch 1 readiness: 2026-10-26

The implementation is prepared for B0/B1/A0/A1/M0/M1, QM9 without explicit H,
one seed each, 1000 epochs. Following the user's launch authorization, jobs
**11846895** (CPU preparation), **11846896** (A100 array), and **11846897**
(W&B sync) were submitted. Executed source is `cd90286`; this documentation
update does not modify the staged source. Initial verification found CPU
preparation running and both dependent jobs pending. See
[`submission metadata`](submissions/qm9-batch1-11846896.json) before retrying.
Completion update, 2026-10-28: all six 1000-epoch tasks, CPU preparation and W&B
sync completed with exit 0:0. [Results and assessment](results/qm9-batch1-11846896/report.md)
include audited metrics, learning curves and actual cost. GPU checks below
describe what was deferred at preparation time; they have now passed in all
six allocations.
Original DeFoG model/rate/dataset modules are unchanged from `year2`; extensions
live in `src/experiments/batch1` on
`experiments/qm9-structural-priors-batch1` in the user's DeFoGPlus fork.

The prepared batch directory is
`/scratch/xinyan14/motiflow-experiments/qm9-batch1/batch1-ready-20261026`.
Its `launch.json` and immutable source archive record the complete commit ID.
See [`qm9-batch1-launch.md`](qm9-batch1-launch.md) for submission/status commands.
Staging and submission without `--submit` do not create jobs.

## Completed verification

- Python compilation and Bash syntax checks passed.
- Ten CPU unit tests passed in the existing cluster environment. They cover
  train-only cache identity, normalized product priors, vertex permutation
  symmetry, path endpoints, original-rate parity, zero-support Kolmogorov
  consistency, frozen-node rates, final-step atom preservation, consistent
  strict VUN counts, and offline W&B sessions.
- All six full nine-layer models passed small CPU forward/backward, sampling,
  strict/legacy evaluation, checkpoint save, and epoch-boundary resume tests.
  Each paused after epoch 1, resumed to epoch 2, and retained curve records at
  completed epochs 0/1/2. Resumed optimizer steps were 2.
- B0 direct noising and the two-step smoke sampler matched the original DeFoG
  implementations exactly for the test seeds. This is a smoke-test comparison,
  not a claim of GPU bitwise replay across hardware or software versions.
- A0/A1 had zero atom/composition changes in all smoke curve evaluations,
  including the final sampler step. M0/M1 completed with persistent reference
  features and the conditional rate implementation.
- A separate CPU smoke run exercised offline W&B alongside the original
  training metrics and final aggregate summaries. No cloud upload was made.
  Installed W&B 0.20.1 accepts explicit offline-directory sync commands.
- Selected checkpoint/home-backup files were checked against SHA-256 indices;
  training-prior caches were also backed up. A final serialization smoke test
  verified embedded code/environment/data provenance and optimizer progress.
- The staging/submission/status helper was exercised in dry-run mode. Six
  eight-hour caps plus the previous pilot reserve 48.516 of 50 GPU-hours.
  Duplicate submissions, a second already-submitted batch, scheduler requeue
  and automatic retries are blocked.

The six-model integration run used commit `9f9a87e`; offline logging was tested
with `fae7199` and `f9a53df`; checkpoint provenance was verified with `4785bff`.
Later readiness documentation does not change those tested model paths.
Detailed test artifacts remain outside Git at
`/scratch/xinyan14/motiflow-experiments/qm9-batch1/development-checks/`:
`preflight-final`, `offline-backup-final`, and `provenance-final`, with sibling
logs. Tiny smoke metrics are implementation checks, not research results.

| Configuration | Parameters | Trained full checkpoint, MiB |
| --- | ---: | ---: |
| B0/B1 | 8,174,217 | 121.28 |
| A0/A1 | 8,174,217 | 117.09 |
| M0/M1 | 8,175,241 | 121.30 |

Initial snapshots were approximately 31.4 MiB. The checkpoint-space reservation
of 20 GiB across six full runs remains conservative. Quota checks showed ample
headroom: approximately 1 GiB of a 100 GiB home quota and 16.34 GiB of a 10 TiB
scratch quota in use before the final small serialization check.

## Checks deferred until authorized launch

No GPU was allocated for the new implementation. Each A100 task first runs a
small full-model GPU smoke test inside its eight-hour allocation and aborts if
that fails. New variants' full-batch GPU memory, throughput and 1000-epoch costs
remain unmeasured; the budget is based on the successful vanilla A100 pilot.
Cloud project write access is checked by the later CPU sync job; local
JSONL/CSV and offline sessions remain usable if that sync fails.

## Ready launch shape

CPU preparation/tests -> A100 array `0-5%2` -> CPU W&B sync/accounting. Each
model uses one A100, eight cores, 32 GiB RAM and an eight-hour cap. The six GPU
tasks include their GPU smoke checks within those caps. Checkpoints, generated
graphs, logs, caches and credentials are excluded from Git. Runtime artifacts
stay in scratch, and selected/recovery checkpoints plus metrics/prior caches
are backed up under `/u/xinyan14/motiflow-artifacts/qm9-batch1/`.
