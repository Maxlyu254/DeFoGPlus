# QM9 batch2: partial completion and validity-first assessment

Verified live on 2026-11-08. Source `1636edc`; array `11917238`.

## Execution and budget

Four of seven training tasks completed 1000 epochs and 96,000 optimizer steps.
The other three failed in the CUDA precheck before model training.

| Task | Configuration | Status | Allocated time | Node |
| --- | --- | --- | --- | --- |
| 11917238_0 | B0 seed1 | COMPLETED | 6:32:54 | ccc0389 |
| 11917238_1 | B0 seed2 | COMPLETED | 6:30:04 | ccc0387 |
| 11917238_2 | A0 seed1 | COMPLETED | 6:07:48 | ccc0387 |
| 11917238_3 | M0 seed1 | COMPLETED | 6:29:18 | ccc0389 |
| 11917238_4 | M0 seed2 | FAILED | 0:00:22 | ccc0390 |
| 11917238_5 | M1 seed1 | FAILED | 0:00:17 | ccc0390 |
| 11917238_6 | M1 seed2 | FAILED | 0:00:16 | ccc0390 |

Preparation 11917237 and W&B sync 11917239 completed with exit 0:0. Sync logs
confirm four successful run uploads to the `defog-qm9-batch2` project.
All three failed tasks received the same GPU on ccc0390, UUID
`GPU-920d77b7-4a43-1213-8011-5a76bd7edff5`; its nvidia-smi logs report
`GPU Recovery Action: Reset`. PyTorch failed CUDA initialization. These are
allocation failures, not observed model failures. Excluding ccc0388 worked,
but another unhealthy GPU was scheduled on ccc0390.

Retry allocation: 25.68306 A100-hours. Including the original failed attempt:
**25.73972 of 50 hours used; 24.26028 remain**. No new jobs were submitted
during this assessment. Completing only the three missing runs would reserve
21 hours at the current seven-hour limit, within the remaining budget. A
targeted retry must exclude both ccc0388 and ccc0390 and preserve completed runs;
the current whole-batch retry guard intentionally refuses partial-success batches.

## Final validity (primary) and overall performance

Each entry below is the mean of three fresh, separate 10k-attempt evaluations
of the epoch-1000 checkpoint. SD is sampling variability conditional on that
trained checkpoint, not uncertainty across training seeds. Strict connected,
canonical-isomeric identity, no correction; 500 sampling steps throughout.

| Configuration | Seed | Validity %, mean ± sampling SD | Uniqueness % | Novelty % | VUN %, mean ± sampling SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0 (batch1) | 98.810 ± 0.095 | 96.124 | 31.656 | 30.067 ± 0.454 |
| B0 | 1 | 99.007 ± 0.101 | 96.260 | 32.688 | 31.153 ± 0.332 |
| B0 | 2 | 99.100 ± 0.082 | 96.007 | 31.937 | 30.387 ± 0.837 |
| A0 | 0 (batch1) | 98.850 ± 0.147 | 96.075 | 30.572 | 29.033 ± 0.414 |
| A0 | 1 | **99.280 ± 0.062** | 96.310 | 30.817 | 29.467 ± 0.556 |
| M0 | 0 (batch1) | 99.047 ± 0.060 | 96.439 | 33.135 | 31.650 ± 0.269 |
| M0 | 1 | 99.067 ± 0.055 | 96.494 | 33.127 | 31.667 ± 0.460 |
| M1 | 0 (batch1 only) | 99.033 ± 0.051 | 96.136 | 33.814 | 32.193 ± 0.240 |

The original suggestion that M0/M1 deliver the best final validity is not
established. A0 seed1 achieves the highest observed final validity and exceeds
the same-seed B0 by **0.273 pp**, about 27.5% fewer failed attempts (72 vs 99.3
per 10k). A0's seed0 advantage was only 0.040 pp. This is promising, with a
variable effect size across two training seeds. A0 changes the task by fixing
the final atoms; this is a system-level benefit, not evidence for a structural
edge prior.

M0 exceeds matched B0 validity by **0.237 pp at seed0** but just **0.060 pp at
seed1**, the latter about six fewer failures per 10k (93.3 vs 99.3). That
replication is too small to separate confidently from sampling/training
variation. B0 seed2 (99.100%) also exceeds either observed M0 result, although
that unmatched-seed comparison is not a controlled effect estimate.
M0's nearly identical results across its two seeds are encouraging but do not
establish low seed variance from only two observations.

B0 across three seeds averages 98.972% validity, with 0.148 pp SD across seed
means. The within-seed sampling SD and this across-seed SD describe different
levels of variability; neither should be turned into nine independent training
replicates. Seed0 was the exploratory screen, not independent confirmation.
No formal training-seed significance claims are justified by these small,
incomplete groups.

## Validity learning speed (primary)

Early curves use one 2k-attempt pool per checkpoint. The AUC is trapezoidal
integration of validity over scheduled epochs 0–100 divided by 100, expressed
as a percentage. It describes the observed sparse curve, not another replicate.

| Configuration | Seed | Epoch10 V % | Epoch25 V % | Epoch50 V % | Epoch100 V % | Normalized validity AUC 0–100 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0 | 78.85 | 91.40 | 95.35 | 97.45 | 89.05 |
| A0 | 0 | 85.95 | 93.45 | 96.50 | 98.50 | 91.10 |
| M0 | 0 | 75.45 | 90.05 | 94.40 | 97.55 | 87.60 |
| B0 | 1 | 77.85 | 86.05 | 93.80 | 95.65 | 86.30 |
| A0 | 1 | 80.60 | 92.35 | 95.25 | 96.45 | 89.22 |
| M0 | 1 | 74.75 | 86.70 | 96.70 | 96.80 | 87.61 |
| B0 | 2 | 70.35 | 90.45 | 93.75 | 96.30 | 86.49 |

**A0's early validity advantage is the clearest repeated result.** Its 0–100
validity AUC exceeds same-seed B0 by 2.05 pp at seed0 and 2.92 pp at seed1;
first observed 90% validity is epoch20 in both A0 seeds, versus epoch25/50 in
the paired B0 seeds. Exact crossing times are interval-censored by the
evaluation schedule and sensitive to 2k sampling noise. A0 is not above every
competitor at every epoch: M0 seed1 overtakes it at epoch50.

M0 does not show a consistent early learning advantage over B0: its AUC
contrast changes from -1.45 pp at seed0 to +1.32 pp at seed1. Baseline variation
is substantial: B0 validity at epoch10 ranges from 70.35% to 78.85% across seeds.

Late validity is near 99%, but there is still some improvement between epoch400
and epoch1000: +0.71 pp for B0 seed1, +0.66 for B0 seed2, +0.61 for A0 seed1,
and +0.61 for M0 seed1, using the full 10k curve pools. These single-pool points
must not replace the three-repeat final estimates above.

## VUN (secondary overall performance)

M0's VUN advantage over same-seed B0 repeats in direction, but shrinks from
**+1.583 pp** at seed0 to **+0.513 pp** at seed1. Its uniqueness is also higher
in both seeds; novelty contributes to the VUN improvement. This is preliminary
evidence for a modest benefit of the overall M0 construction, with uncertain
effect size.

A0's final VUN is lower than B0 in both paired seeds (-1.033 and -1.687 pp),
mainly through lower novelty. This does not negate its validity advantage.
M0 exceeds A0 VUN by 2.617 and 2.200 pp in the two seeds; the final-validity
ordering between these two configurations reverses between seeds.

There is **no new evidence for M1 versus M0**, because both new M1 runs failed
before training. The isolated Chung-Lu edge-prior effect remains unconfirmed.

VUN has not universally plateaued by epoch1000. In the full 10k curve pools,
epoch400→1000 VUN falls from 33.73→31.11% (B0 seed1), 34.85→30.82% (B0 seed2),
32.37→28.81% (A0 seed1), and 35.40→31.88% (M0 seed1). No adaptive stopping or
extension was used. A declining novelty/VUN curve alone does not establish
overfitting or worse distribution fitting; held-out distribution diagnostics
remain relevant. Do not select checkpoints by maximum VUN.

## Artifacts and audit

- `assessment.json`: grouped descriptive results, paired contrasts, budget.
- `final-comparison.csv`: per-seed final means and sampling SD, diagnostics.
- `learning-speed.csv`: early validity AUC and first observed thresholds.
- `learning-curves.csv`: full-pool and matched-2k metrics; use matched 2k for
  cross-batch U/N/VUN curves because pool size changes deduplication outcomes.
- `learning-curves.png` / `.pdf`: per-seed validity and VUN curves and endpoints.

`scripts/analyze_batch2.py` validates all completed-run epoch/step counts,
scheduled evaluations, metric count identities, recomputed means/SD, final
weight fingerprints against the final curve, and shared dataset splits,
canonical training reference, validation subset, and train-only prior cache.
This is a metric/provenance audit; model checkpoints were not downloaded or
loaded. A0 preserves atoms and compositions throughout all saved curve points.
M0 seed1 changes approximately 44.9% of atoms and 90.5% of represented
compositions between initialization and output, consistent with its intended
composition-changing semantics.

Raw downloaded metrics/logs: ignored local
`artifacts/qm9-batch2-retry-11917238/`. Cluster raw runs/checkpoints:
`/scratch/xinyan14/motiflow-experiments/qm9-batch2/batch2-retry-20261029/`.
Backups: `/u/xinyan14/motiflow-artifacts/qm9-batch2/batch2-retry-20261029/`.
W&B: https://wandb.ai/maxlyu254-university-of-illinois-urbana-champaign/defog-qm9-batch2
