# QM9 main-configuration ablation budget

Estimate made 2026-10-26 from the successful A100-SXM4-80GB benchmark in
`benchmarks/results/qm9-a100-retry-20261026T042900Z/analysis.json`.
This is a projection; the eight augmented configurations are not implemented
or benchmarked yet. No jobs are submitted by this estimate.

Update: the user does not want the approximately 200-GPU-hour study. The current
request is six configurations (B0/B1/A0/A1/M0/M1), one seed each through 1000
epochs, within approximately 50 GPU-hours. See `docs/qm9-batch1-plan.md` for that
plan; it has not been executed. The full matrix below is a reference estimate.

## Scope and assumptions

- All nine main configurations: B0, B1, D0, D1, A0, A1, A2, M0, M1.
- QM9 without H, 1000 epochs, three training seeds each: 27 complete runs.
- Full existing backbone; training batch 1024 and four loader workers.
- 500-step sampling with inference batch cap 2048.
- Twelve 2000-attempt learning-curve evaluations per run.
- Three independent 10000-attempt final sampling repetitions per checkpoint.
- Report final-epoch and validation-selected checkpoints. If they differ,
  evaluate both with fresh final samples; if they are the same, reuse that set.
- Baseline-equivalent timing for unmeasured variants. All feature/prior overhead,
  setup/debugging, and cold restart differences must be checked during screening.
- No ZINC/MOSES/GuacaMol, secondary augmentations, sampler searches, FCD,
  visualization, implementation labor, or queue delay included.

## Per training seed

Measured baseline projection: training 5.192 h, curve evaluations 0.207 h, one
checkpoint's three final evaluations 0.260 h, plus small startup/checkpoint I/O.

- One final checkpoint: 5.666 GPU-hours per seed.
- Two distinct final checkpoints: 5.926 GPU-hours per seed.

The earlier vanilla report budgeted one checkpoint's final repetitions. The
second checkpoint adds 0.260 GPU-hours per run, or 7.01 GPU-hours across 27 runs.

## Total

| Scope | Baseline-equivalent A100 GPU-hours |
| --- | --- |
| Nine configurations, one seed each, 1000 epochs | 51.0-53.3 |
| Nine configurations, three seeds each, 1000 epochs | 153.0-160.0 |
| Full three-seed matrix plus 25% contingency | 191.2-200.0 |
| Twenty-seven 8-hour Slurm reservations, maximum requested | 216 |

For planning, use approximately 200 A100 GPU-hours, requesting up to 8 hours per
full seed. Allocated elapsed usage can be less than the sum of requested limits;
these values do not imply dollar billing. Queue delay is separate.

Ideal compute-time lower bounds at the baseline-equivalent rate: 153-160 hours
on one GPU, 76.5-80 hours on two, or 38.2-40 hours on four, if concurrency is
available. These do not predict queue waits or stage scheduling.

## Screening and avoiding double counting

The initial one-seed 200-epoch screen uses eight scheduled curve evaluations
(epochs 0,1,5,10,25,50,100,200), with no full final-repetition campaign:

- About 1.184 GPU-hours/configuration, 10.65 GPU-hours for all nine.
- Initially request up to 2 hours per screening job (18 hours across nine caps).
- Save optimizer, RNG, and model state so seed 0 can continue to epoch 1000.
  If all nine advance, their screening training and scheduled evaluations are
  already included in the 153-160-hour total; do not add 10.65 hours again.
- Splitting each seed-0 run into a 2-hour screening reservation and a 6-hour
  continuation, plus 18 additional 8-hour seed runs, still requests 216 hours
  in total. Extra restart overhead is covered by the margin, not measured here.
- If screening runs are discarded and all full runs restarted, add about 11
  GPU-hours of actual screening work to the estimate.

Use screening to replace baseline-equivalent estimates with measured per-model
rates. Frozen-atom models and conditional models need not have identical runtime.
Screening can also justify advancing fewer configurations, but preserve matched
controls for every promoted augmentation.
