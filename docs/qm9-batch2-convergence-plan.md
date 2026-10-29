# Batch 2 proposal: fresh 50 A100-hours, validity-first convergence

Updated 2026-10-28. User requested a fresh 50-hour budget and new training seeds
continued until validity and VUN mostly stabilize. This supersedes the proposed
100-epoch budget-limited follow-up. It is a plan, not a job submission.

## Matrix and priorities

| Model | New training seeds | Coverage including existing seed 0 | Purpose |
| --- | --- | --- | --- |
| B0 | 1,2 | 3 seeds | Baseline |
| M0 | 1,2 | 3 seeds | Apparent mature-validity advantage |
| M1 | 1,2 | 3 seeds | Matched structural-edge-prior test against M0 |
| A0 | 1 | 2 seeds | Confirm early validity-learning advantage |

Seven new runs. Defer B1/A1 and A0 seed 2. A balanced four-model/two-new-seed
matrix would take about 49 hours to epoch 1000 even without best-VUN reporting,
leaving no useful extension margin. Prefer convergence and the main matched
comparisons over filling that eighth matrix cell. A0's evidence remains less
replicated and must be labeled accordingly.

Retain QM9 without H, original split, batch size, backbone, optimizer/lr,
training representation, 500-step sampler, and frozen prior-statistics cache.
Use matched training seeds, graph-size draws and reference draws where
applicable. Training seed 0 is exploratory; seeds 1/2 are replication.

## Horizon and stability

Plan for all runs through epoch 800, then review at 800,1000,1200. Save
recoverable state and continue in blocks of 200 when needed and funded.
Do not assume epoch 1000 is converged: the original 2k curves retain some
late drift and noise, especially in VUN.

Proposed operational definition of mostly stable, fixed before launch:

- Evaluate the last three late checkpoints spanning 400 epochs.
- Estimated validity drift across that window is no more than 0.2 percentage
  points, and VUN drift no more than 2 points; inspect intermediate behavior
  for sustained trends or oscillation. These are practical tolerances, not
  universal statistical thresholds or claims of exact convergence.
- Use 10k pools at late checkpoints, with independent confirmation samples
  when a run appears near the stopping boundary. Aggregate same-size pools
  separately; account for uncertainty. A nonsignificant trend alone is not
  evidence of stability. If resolution is insufficient, label it uncertain
  and either increase sampling or defer the conclusion.
- Require both validity and VUN to satisfy the rule. A high or maximal VUN
  does not trigger stopping or checkpoint selection.

Keep M0/M1 on a common horizon within each new training seed: if one needs an
extension, budget a paired extension. Compare other models at prespecified
common checkpoints (800 and 1000 where available), rather than silently mixing
each run's different terminal epoch. Report terminal/stability results separately.

If the budget cannot fund a needed extension, stop safely and explicitly report
"budget-limited; stability not established". Do not relabel the spending cap as
convergence. The 50-hour ceiling takes priority over extensions or extra seeds.

## Evaluation

- Early curve: epochs 0,1,5,10,20,25,50,100,200, with 2k attempts per point.
- Late curve: epochs 400,600,800,1000 and extensions every 200, with 10k attempts.
- Keep early/late pool sizes clearly labeled. Use 2k subsets of late pools for
  same-size full learning-curve comparison; use full 10k pools for late stability.
- Final reporting: three fresh independent 10k pools at the stopping checkpoint,
  generated once after the training endpoint is determined. No extra best-VUN
  evaluation. Checkpoint selection/stopping and final reporting pools are separate.
- Save all scheduled checkpoints, plus latest/previous recovery every 25 epochs.
- Primary outcomes: strict validity at common late epochs, failed attempts per
  10k, early validity AUC and training/GPU time to useful validity thresholds.
- Secondary outcomes: matched-checkpoint U/N/VUN, failure causes, composition
  changes, size-conditioned validity, and held-out validation structural/chemical
  distribution checks. Validity is not a complete measure of distribution fit.

Average sampling repetitions within a training seed; compare paired seed-level
differences for B0/M0/M1. Do not count checkpoints or sampling repetitions as
independent trained models. Three seeds remain a small replication study;
wide intervals must remain inconclusive.

## Budget and staging

Measured training-only time through 1000 epochs: B0 5.33 hours, A0 5.27,
M0 5.44, M1 5.49. The seven runs require approximately 37.80 training hours
if all reach 1000. Late 10k monitoring, early curves, final reporting and startup
put the expected total around 43-45 hours. Reserve approximately 5-7 hours for
plateau confirmation, extensions and runtime variation. These are projections,
not a guarantee that all seven runs can stabilize before the cap.

Use a fresh ledger starting at zero for this batch; do not subtract the previous
38.37 hours. Include all debug/smoke/retry allocations in this new 50-hour ledger.
Reserve final-reporting costs before allocating extensions. Reconcile sacct
after each stage and do not reserve or consume more than the batch limit.

Launch sequence after separate implementation/authorization:

1. B0/M0/M1/A0 seed 1 through the first stability review. Measure actual costs.
2. Continue unstable comparisons where affordable; launch B0/M0/M1 seed 2 with
   enough remaining allocation for convergence checks and final reporting.
3. Use remaining budget on unstable matched comparisons, not extra variants.

The current runner needs convergence-aware stopping, separate late monitoring
and final reporting, stable cross-epoch subsampling, and a new batch ledger.
The old runner's automatic best-VUN selection is unsuitable unchanged. No jobs
have been submitted and no runner changes are authorized by this planning file.
