# Vanilla DeFoG A100 budget benchmark results

Measured 2026-10-26. CPU job 11822422 and A100 job 11822489 both completed with exit code 0:0. The full benchmark produced `raw/summary.json`; no benchmark phases failed.

## Workload and provenance

- Source: year2, a35327f3bfa709e8da344567968004ca0e644831, unmodified model.
- GPU: NVIDIA A100-SXM4-80GB; one GPU, 8 allocated CPU cores, 64 GiB allocated host RAM.
- QM9 without H: 97,734 processed training graphs, 20,042 validation, 13,055 test; 96 batches/epoch at training batch 1024.
- Full 9-layer model, 8,174,217 parameters; float32, marginal priors, existing structural features.
- PyTorch 2.4.0.post301 / CUDA 11.8, Lightning 2.5.2, NumPy 2.3.1, RDKit 2025.03.4. NumPy fallback was not required.
- Sampling: 500 steps, polydec, eta=0, omega=0, no visualization.
- Two separate 20-epoch training profiles at seed 0, followed by evaluations of the faster profile and a successful two-epoch checkpoint resume.

## Actual consumed allocation

| Job | State / exit | Elapsed allocation |
| --- | --- | --- |
| CPU preparation 11822422 | COMPLETED / 0:0 | 5 min 49 sec, no GPU |
| A100 benchmark 11822489 | COMPLETED / 0:0 | 30 min 59 sec = 0.516 GPU-hours |

CPU preparation reused the previously installed Conda packages and environment. Its elapsed time is not a fresh-install estimate. Cold QM9/data/SMILES preparation took 212.28 sec inside the application (228.43 sec including process startup). CPU preflight trained one batch and verified checkpoint reload and sampling.

The A100 job waited 24 min 49 sec after submission, including its dependency; after preparation completed it waited another 19 min 37 sec. These observed waits are not forecasts of future queue times.

## Training throughput

Steady-state excludes the first two epochs. Timers stop before the model's epoch-end metric logging; full Trainer.fit wall time is shown separately.

| Loader workers | Epoch median | Epoch p90 | Median graphs/sec | 20-epoch fit | Peak allocated GPU | Peak reserved GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 22.49 s | 22.53 s | 4345 | 457.77 s | 6.97 GiB | 7.36 GiB |
| 4 | 18.69 s | 19.48 s | 5229 | 378.33 s | 6.98 GiB | 7.37 GiB |

Four workers reduced steady epoch time by 16.9% (20.4% throughput increase). Five-second GPU monitoring averaged approximately 70% utilization in steady workers=0 training and 83% with workers=4. Main-job-step peak host RSS was 4.42 GiB. Across all phases, nvidia-smi peak device memory was 9.10 GiB; allocator reservation reached 8.14 GiB during integrated training/evaluation. The `/usr/bin/time` report around srun only measures launcher resources; use Slurm's python step for host memory.

Checkpoint saves outside fit took 0.41-0.46 sec for 122.1 MiB files. Weight loading took 0.206 sec; this excludes a complete new-process startup. Cached model/data setup took approximately 0.86-1.15 sec, while Python imports took 20.34 sec. Integrated saves took 0.62-0.73 sec.

## Generation and metric overhead

| Attempts | Inference batch cap | Generation time | Molecules/sec | Peak allocated GPU |
| --- | --- | --- | --- | --- |
| 2,000 | 256 | 119.95 s | 16.67 | 0.18 GiB |
| 2,000 | 512 | 86.54 s | 23.11 | 0.31 GiB |
| 2,000 | 1024 | 69.21 s | 28.90 | 0.59 GiB |
| 2,000 | 2048 | 61.44 s | 32.55 | 1.12 GiB |
| 10,000 | 2048 | 307.78 s | 32.49 | 1.14 GiB |

At batch cap 2048, the three 2k generation repetitions took 61.40-61.44 sec. Strict sanitization, connectedness, and canonicalization took about 0.52-0.53 sec for 2k and 2.66 sec for 10k. Serialization took approximately 0.15 sec for 2k and 1.11 sec for 10k. Set-based deduplication and novelty membership took milliseconds or less.

Including generation, strict metrics, reference-set creation, and serialization, a 2k curve evaluation averaged **62.10 sec** and a 10k final evaluation took **311.56 sec**. The legacy evaluator adds 2.89 sec for 2k or 14.12 sec for 10k; it is a separate relaxed/corrected pipeline, not the same validity policy. These CPU stages still consumed GPU allocation wall time in this pilot.

The two resumed epochs plus two 2k evaluations and checkpoint writes took 165.14 sec. Summing recorded components gives 163.44 sec, leaving 1.70 sec (1.0%) of framework/transition overhead. Resume completed exactly epochs 21 and 22.

## Quality snapshot, not a fully trained result

All standalone evaluations below use the **20-epoch workers=4 checkpoint** (1920 optimizer steps). The raw generation event reports `checkpoint_age=0` because it reads Lightning's property on a freshly constructed model after loading weights directly. That metadata field is misleading; checkpoint-load and training events establish the actual 20-epoch age. Do not interpret the snapshot as an untrained model.

| 10,000 attempts | V | U among valid | N among unique valid | VUN |
| --- | --- | --- | --- | --- |
| Strict connected, no correction | 84.83% | 97.95% | 62.13% | 51.62% |
| Legacy relaxed / largest-fragment | 93.67% | 97.41% | 62.75% | 57.25% |

Strict counts: 8,483 valid connected molecules, 8,309 unique, 5,162 novel; 801 sanitization failures and 716 disconnected graphs. The legacy evaluator's uncorrected sanitization-only validity was 91.99%, which includes disconnected graphs; never multiply that value by relaxed U/N.

Across three 2k sampling repetitions, strict V/U/N/VUN were 84.10 +/- 0.59%, 99.74 +/- 0.07%, 62.38 +/- 2.06%, and 52.32 +/- 1.40% respectively (sample standard deviation, one trained checkpoint). The 10k uniqueness/VUN differ partly because deduplication depends on sample count, not because the model changed.

| Completed epoch | Evaluation attempts | Strict V | Strict U | Strict N | Strict VUN |
| --- | --- | --- | --- | --- | --- |
| 20 | 2k, mean of 3 repeats | 84.10% | 99.74% | 62.38% | 52.32% |
| 21 | 2k, one repeat | 84.90% | 99.29% | 61.09% | 51.50% |
| 22 | 2k, one repeat | 88.10% | 99.32% | 52.74% | 46.15% |

These are sparse early diagnostic points with different sampling seeds, not a complete learning curve or evidence of a persistent trend. No 1000-epoch model has been trained.

## Projected full-study budget on this A100

Assumptions: 1000 epochs, four workers, 12 online curve evaluations of 2k attempts, three fresh final evaluations of 10k, identical model/sampler settings, and no visualization/FCD. These are extrapolations, not measured full-run costs.

| Component per training seed | Projected time |
| --- | --- |
| Training, median epoch timing | 5.19 h |
| 12 curve evaluations | 12.42 min |
| Three 10k final evaluations | 15.58 min |
| Imports/cached setup and checkpoint writes | About 0.5 min, excluding extra process restarts |
| Total central projection | **5.67 GPU-hours** |
| Three training seeds | **17.00 GPU-hours** |

Using epoch p90 instead of median gives 5.88 GPU-hours per seed. Adding a 25% margin gives 7.08-7.36 hours; an **8-hour single-A100 reservation per full seed** is a practical initial request. Three such requests reserve 24 GPU-hours while the central consumed-time estimate is about 17 GPU-hours.

Recommendation: train batch 1024, train.num_workers=4, inference batch cap 2048 (best tested), 8 CPU cores, and initially 32 GiB host RAM rather than 64 GiB. Larger inference batches may be possible but were not benchmarked. No new jobs were submitted during this analysis.

## Limits and artifacts

- One GPU and training seed; confirm throughput on the first full run. Extrapolation does not establish convergence or future VUN.
- QM9 only. Profile ZINC/MOSES separately; graph sizes and dataset lengths change costs substantially.
- Unused full-training-set reference-metric calculation, FCD, visualization, repeated cold installation and queue delays are excluded.
- CPU chemistry cost can change as validity improves; remeasure at later checkpoints.
- Original model and prior are unchanged. Model input/output work is the vanilla full network; instrumentation adds timing and separate metrics.
- Raw events, summary, GPU monitor, strict metrics, resolved configs, setup timings, and Slurm accounting are saved locally under `raw/` and `setup/`. Derived values are in `analysis.json`.
- Remote originals: `/scratch/xinyan14/motiflow-benchmarks/qm9-a100-retry-20261026T042900Z/`.
