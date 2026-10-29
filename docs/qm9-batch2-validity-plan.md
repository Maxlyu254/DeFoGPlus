# Validity-first reassessment and proposed batch 2

The reassessment below remains applicable. Its budget-limited 100-epoch batch
proposal is superseded by the user's fresh 50-hour budget and convergence goal:
see [`current convergence plan`](qm9-batch2-convergence-plan.md).

Updated 2026-10-28 following the user's clarification. This is a proposal,
not authorization to implement or submit a new experiment batch.

## Research priority

The primary objective is to improve learning of the dataset distribution.
Primary empirical outcomes are strict final validity and the validity learning
curve, including training cost to reach useful validity. VUN is a secondary
joint performance diagnostic at the same checkpoints. Do not rank methods or
choose their training horizon using their best VUN. The previously proposed
epoch-25 VUN confirmation study is superseded by this plan.

Validity measures chemical feasibility, not the entire molecular distribution.
Also compare held-out validation distributions of size, atom composition, bond
types and represented degrees. Novelty decline alone does not prove overfitting.

## Reassessment of the completed seed-0 screen

Final results below use three fresh 10k pools at epoch 1000. SD describes
sampling variability, not training-seed variability. Learning-curve values use
2k pools and are not interchangeable with the 10k reporting estimates.

| Model | Final validity %, mean ± SD | Failed attempts per 10k | Final VUN % |
| --- | ---: | ---: | ---: |
| B0 | 98.81 ± 0.10 | 119.0 | 30.07 |
| B1 | 98.86 ± 0.13 | 114.0 | 30.57 |
| A0 | 98.85 ± 0.15 | 115.0 | 29.03 |
| A1 | 98.86 ± 0.05 | 113.7 | 28.87 |
| M0 | 99.05 ± 0.06 | 95.3 | 31.65 |
| M1 | 99.03 ± 0.05 | 96.7 | 32.19 |

M0/M1 have the highest final validity, about +0.22 to +0.24 percentage points
over B0, corresponding to about 19-20% fewer failed attempts. This small
absolute effect can matter near saturation, but needs independent training
replication. M0 and M1 are effectively tied on validity. The isolated edge-prior
comparisons B1-B0, A1-A0 and M1-M0 show no clear final-validity improvement.

| Model | Validity at 5 | At 10 | At 25 | At 50 | At 100 | Normalized validity AUC, epochs 0-100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 57.00 | 78.85 | 91.40 | 95.35 | 97.45 | 89.05 |
| B1 | 55.30 | 76.00 | 91.35 | 94.90 | 97.60 | 88.68 |
| A0 | 63.70 | 85.95 | 93.45 | 96.50 | 98.50 | 91.10 |
| A1 | 60.85 | 85.00 | 92.20 | 96.85 | 98.25 | 90.79 |
| M0 | 46.75 | 75.45 | 90.05 | 94.40 | 97.55 | 87.60 |
| M1 | 45.80 | 72.55 | 90.60 | 94.70 | 98.10 | 87.72 |

AUC is trapezoidal integration over the scheduled observations divided by 100;
it describes the sampled curve and assumes linear interpolation between points.
It is not an independent replication or an exact continuous-time measurement.

A0/A1 learn validity faster early. A0 reaches observed 90% validity at epoch 20,
versus epoch 25 for B0/M0/M1; observed 98% is at epoch 100 for A0/A1/M1 and 200
for B0/M0. Crossings are interval-censored by the evaluation grid, and 2k-pool
noise near thresholds can move the observed crossing. At epoch 10 A0 exceeds
B0 by 7.10 points and M0 by 10.50 points. This is a useful system-level result,
but A models receive the clean atom identities and only learn bonds; the
advantage does not isolate a better edge initialization or an equally hard task.
M variants initially learn more slowly and have slightly higher final validity.

The degree-aware edge prior does not consistently accelerate learning: the
0-100 normalized validity AUC is slightly worse for B1 vs B0 and A1 vs A0, and
nearly tied for M1 vs M0. Claims that M1 learns validity faster are unsupported.
Final M1 VUN exceeds M0 because of novelty rather than a validity advantage.

## Proposed next batch within approximately 11.63 remaining A100-hours

Run B0, A0, M0, M1 with new training seeds 1 and 2, each through **100 epochs**:
eight short runs. Retain B0 as baseline, A0 for rapid validity learning, M0 for
the apparent mature-validity gain, and M1 as M0's matched structural-prior
comparison. Defer B1/A1 because their isolated prior changes currently show
neither a clear endpoint-validity nor learning-speed advantage.

Primary outcomes:

1. Strict validity at the fixed endpoint, **epoch 100**, using three separate
   10k sampling pools. This is a 100-epoch efficiency endpoint, not proof of
   superior fully converged validity at epoch 1000.
2. Normalized validity AUC through 100 epochs, and first-observed/sustained
   90%, 95% and 98% validity crossings. Treat unobserved crossings as censored;
   report crossing intervals and independent confirmation near thresholds.
3. Training seconds and allocated GPU time to reach the same validity.
   Distinguish training-only time from time spent generating evaluation pools.

Secondary outcomes: U/N/VUN at the same fixed checkpoints; invalid, valence and
disconnected counts; size-conditioned validity; composition/degree/bond-type
diagnostics. Use common curve epochs 0,1,5,10,20,25,50,75,100, common sampling
settings and matched reference/size draws. Curves use 2k pools; final reporting
uses 10k pools. No best-VUN checkpoint selection or duplicate best/final 10k
evaluation is needed in this confirmation batch.

Evaluate the saved epoch-100 checkpoints for seed 0 under the same fresh 10k
reporting protocol, giving endpoint results for three training seeds. Average
the three sampling repetitions within each seed, then compare paired
training-seed differences. Report each seed rather than treating sampling
repetitions or checkpoints as independent trained models. Seed 0 is exploratory;
the two new seeds provide the independent replication check.

Estimated cost from measured rates: eight short runs including curve and final
evaluation about 8 A100-hours; seed-0 endpoint reevaluation about 1 hour. Allow
roughly 10-11.5 hours with contingency. Exact caps and the live budget ledger
must be reconciled before submission. A possible reservation is eight 75-minute
allocations plus one 75-minute seed-0 evaluation allocation: 11.25 GPU-hours.
This requires a fixed-endpoint short-run/evaluation configuration; the current
runner's default 1000 epochs and VUN-selected final evaluation must not be used
unchanged. This plan does not submit those jobs.

## Mature-validity confirmation needs a separate budget decision

Independent 1000-epoch replications are needed to confirm the approximately
0.23-point final-validity gain of M0/M1. Two new seeds for B0 and one promising
M variant require four full runs, roughly 24-26 A100-hours under the previous
protocol. Repeating all four variants for two new seeds is roughly 50 hours.
The short batch above confirms learning efficiency and guides that decision;
it cannot substitute for mature-validity replication. With small training-seed
counts, wide intervals should be labeled inconclusive rather than proof of no
effect. Report failed-attempt reductions as well as absolute percentage points.
