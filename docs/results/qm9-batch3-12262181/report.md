# QM9 batch3 results and combined assessment

Verified live on 2026-11-12. Executed immutable source `19e1288`.
No replacement or continuation jobs were submitted during this assessment.

## Execution status

| Array task | Configuration | State | Allocated time |
| --- | --- | --- | --- |
| 12262181_0 | B1 seed1 | TIMEOUT | 7:00:14 |
| 12262181_1 | B1 seed2 | COMPLETED | 6:31:16 |
| 12262181_2 | A0 seed2 | COMPLETED | 6:14:44 |
| 12262181_3 | A1 seed1 | COMPLETED | 6:12:57 |
| 12262181_4 | A1 seed2 | COMPLETED | 6:17:28 |
| 12262181_5 | M0 seed2 | COMPLETED | 6:32:07 |
| 12262181_6 | M1 seed1 | COMPLETED | 6:24:48 |
| 12262181_7 | M1 seed2 | COMPLETED | 6:32:33 |

The seven successful runs completed 1000 epochs and 96,000 optimizer steps.
Preparation 12262180 completed. Sync/controller 12262182 uploaded all eight
offline sessions (including the partial B1 run), then intentionally exited
nonzero because one training task remained incomplete. The final notification
job 12262185 completed and wrote batch-summary.json with INCOMPLETE, 7/8 runs.
This verifies notifier execution; inbox delivery cannot be independently verified.
There are no active user jobs and no automatic retry waves were submitted.

Total allocation: **51.76861 A100-hours**, within the authorized 56-hour
exception; **4.23139 hours remain** under that ceiling. Historical batch2
charges remain in that batch's separate ledger.

## B1 seed1 timeout

This was not the previous GPU-reset failure. CUDA initialization passed and
training ran normally for hundreds of epochs on ccc0387. Slurm terminated the
allocation at its seven-hour time limit. The reset-only automatic retry policy
correctly did not restart it.

The run logged **848 completed epochs / 81,408 steps**, but the latest durable
recovery checkpoint is **epoch825** (physically verified, approximately 123 MiB).
Its final reporting summary is absent. The last scheduled curve is epoch800:
strict validity 99.19%, VUN 31.07% from one 10k pool; this is not an epoch1000
result and is excluded from final comparisons.

Recorded epoch time averaged 19.28 seconds, with a maximum of 20.79 seconds.
However, the last train row was written at 2026-11-10 12:33:20 UTC and Slurm
terminated the job at approximately 14:18:32 UTC: about 105 minutes without
another completed-epoch row. The available application logs do not identify
the blocking operation. This is not evidence that ordinary epoch computation
needs seven hours, nor evidence of the earlier reset fault. Further diagnosis
would need the process/GPU/telemetry trace. No epoch848 recovery checkpoint was
saved, so the signal path did not produce graceful recovery in this incident.

Recommended recovery is a targeted continuation from epoch825, rather than
retraining from zero. Approximately 175 epochs plus remaining curve/final
evaluation should fit a two-hour request at normal observed speeds, but a
recurring stall is not excluded. Reconcile post-checkpoint log rows/attempts
when resuming so epochs826–848 are not double-counted. This continuation is
not submitted by this report; it lies outside the reset-only retry policy.

## New epoch1000 results

Each entry averages three fresh 10k-attempt pools. Strict connected validity,
canonical-isomeric identity, no correction; 500 sampling steps. U and N use
the same accepted strict molecules. Sampling SD is in final-per-seed.csv.

| Configuration | Seed | Validity % | Uniqueness % | Novelty % | VUN % |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 2 | 99.077 | 96.242 | 32.109 | 30.617 |
| A0 | 2 | 99.247 | 96.272 | 31.405 | 30.007 |
| A1 | 1 | 99.097 | 96.196 | 30.474 | 29.050 |
| A1 | 2 | 99.217 | 95.989 | 31.189 | 29.703 |
| M0 | 2 | 99.087 | 96.679 | 33.749 | 32.330 |
| M1 | 1 | 99.027 | 96.472 | 33.409 | 31.917 |
| M1 | 2 | 99.160 | 96.581 | 34.009 | 32.570 |

## Combined final comparison: all available training seeds

Average sampling repetitions within each training seed, then average the seed
means. The SD below is across trained-model seed means, not across nine sample
pools. It includes finite sampling noise. Seed0 was the exploratory screen.
B1 has only seeds0,2; all other configurations have seeds0,1,2. Therefore its
unpaired group mean must not be treated as a matched three-seed contrast.

| Configuration | Complete seeds | Validity %, mean ± seed SD | U % | N % | VUN %, mean ± seed SD |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 | 0,1,2 | 98.972 ± 0.148 | 96.130 | 32.094 | 30.536 ± 0.558 |
| B1 | 0,2 | 98.968 ± 0.153 | 96.305 | 32.097 | 30.592 ± 0.035 |
| A0 | 0,1,2 | 99.126 ± 0.239 | 96.219 | 30.931 | 29.502 ± 0.488 |
| A1 | 0,1,2 | 99.059 ± 0.180 | 96.108 | 30.678 | 29.207 ± 0.440 |
| M0 | 0,1,2 | 99.067 ± 0.020 | 96.538 | 33.337 | 31.882 ± 0.388 |
| M1 | 0,1,2 | 99.073 ± 0.075 | 96.397 | 33.744 | 32.227 ± 0.328 |

### Final validity: primary

A0 has the highest observed mean final validity. Its matched B0 improvements
are +0.040, +0.273 and +0.147 percentage points in seeds0,1,2. Thus the positive
direction repeats, although the effect size varies substantially. This is a
benefit of the fixed-atom construction as a whole; knowing final atoms and
generating only bonds makes the task different from B0 joint generation.

The isolated prior changes have little evidence of improved final validity:

- B1 minus B0, seeds0,2: +0.050 and -0.023 pp.
- A1 minus A0, seeds0,1,2: +0.013, -0.183 and -0.030 pp.
- M1 minus M0, seeds0,1,2: -0.013, -0.040 and +0.073 pp; mean +0.0067 pp.

M0's original validity advantage over B0 does not hold consistently: +0.237,
+0.060 and -0.013 pp across the three seeds. M1 exceeds B0 by +0.223, +0.020
and +0.060 pp, but its two confirmation seeds have much smaller gains than
the exploratory seed. These near-saturation effects are small relative to
training and sampling variability. The findings do not establish a robust
final-validity gain from a structurally informed edge prior.

### Validity learning speed: primary

The normalized trapezoidal validity AUC over scheduled epochs0–100 gives the
following paired differences, in percentage points:

| Contrast | Seed0 | Seed1 | Seed2 |
| --- | ---: | ---: | ---: |
| A0 minus B0 | +2.052 | +2.919 | +2.124 |
| A1 minus A0 | -0.314 | -0.036 | -0.116 |
| M0 minus B0 | -1.448 | +1.318 | +0.790 |
| M1 minus M0 | +0.115 | +0.452 | +0.130 |

A0's early validity advantage is the clearest repeated learning-speed result.
The atom-aware edge prior does not improve on it: A1 has slightly lower AUC
in all three seeds. M0 has no consistent advantage over B0. M1 has small
positive AUC differences over M0 in all three seeds (mean +0.233 pp), but
their magnitude is much smaller than A0's advantage over B0. Sparse 2k-pool
curves and only three training seeds do not establish a meaningful acceleration
from this small effect. Do not treat curve points as independent replications.

### VUN: secondary overall performance

M0 and M1 have higher final VUN than same-seed B0 in every seed. M0's paired
advantages are +1.583, +0.513 and +1.943 pp; M1's are +2.127, +0.763 and
+2.183 pp. This supports a repeated overall-performance benefit of the M
construction, while the final-validity advantage remains weak.

M1 exceeds M0 VUN by **+0.543, +0.250 and +0.240 pp** (mean +0.344 pp).
The direction now replicates across both new seeds. Mean novelty rises from
33.337 to 33.744%, while mean uniqueness falls slightly from 96.538 to 96.397%;
validity is essentially tied. This is a small secondary benefit from the
reference-based edge prior, not a validity breakthrough or a formal significance
claim. Its size is comparable to per-checkpoint sampling variability, so more
precise pools/seeds would be needed to establish a small population effect.

A1's VUN is below A0 in all seeds (-0.167, -0.417, -0.303 pp). A0's VUN is
below B0 in all seeds (-1.033, -1.687, -0.380 pp), mostly through lower novelty.
That does not negate A0's primary validity/learning-speed benefit. No best-VUN
checkpoint ranking is used. Higher VUN alone is not complete distribution fit,
and declining novelty alone does not prove overfitting.

## Artifacts, audit and remaining matrix

The script `scripts/analyze_qm9_matrix.py` checks metric count identities,
epoch/step counts, evaluation grids, dataset/reference/cache provenance,
final weight fingerprints against the final curve, and recomputed final means
and sampling SD. All 17 complete runs passed. It preserves the incomplete B1
curve separately and does not substitute epoch800 for an epoch1000 result.
Checkpoint files were not downloaded/loaded during this assessment.

- assessment.json: grouped results, paired differences, partial-run status, budget.
- final-per-seed.csv: all 17 final estimates and sampling SD.
- learning-speed.csv: early validity statistics, including partial B1 seed1.
- learning-curves.csv: full-pool and matched-2k values, complete-run flag.
- learning-curves.png/.pdf: completed-run comparisons; B1 seed1 omitted so group
  membership does not change across epochs. Curve bands show training-seed SD;
  final error bars show conditional sampling SD. U/N/VUN curves use matched
  2k pools, avoiding differences caused by late-pool deduplication size.

The six implemented configurations now have **17/18 completed cells**; only
B1 seed1 remains. The original nine-model/three-seed matrix has **17/27 complete**:
one continuation plus nine unimplemented D0/D1/A2 cells remain. Their full
training still requires profiling and separate authorization.

Raw metrics/logs: ignored local `artifacts/qm9-batch3-12262181/`.
Cluster root: `/scratch/xinyan14/motiflow-experiments/qm9-batch3/batch3-matrix-20261108/`.
W&B: https://wandb.ai/maxlyu254-university-of-illinois-urbana-champaign/defog-qm9-batch3
