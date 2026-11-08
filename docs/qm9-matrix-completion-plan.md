# QM9 matrix completion proposal

Updated 2026-11-08. Planning only: no new implementations or submissions.

Standing user instruction: each new experiment batch has a 50-A100-GPU-hour
budget unless the user explicitly changes it. This does not retrospectively
erase historical allocation charges or replenish an already-running batch.

## Core matrix: QM9 without explicit H

The original full reference matrix is nine configurations by training seeds
0,1,2, all at exactly 1000 epochs. This is a proposed definition of completion,
not a launch authorization or a requirement to perform every optional study.
There are ten completed cells and seventeen missing cells. Seed0 is the
exploratory screen. Failed allocations count toward cost, not completed cells.

| ID | Meaning | Seed0 | Seed1 | Seed2 | Additional runs |
| --- | --- | --- | --- | --- | ---: |
| B0 | Original global node/edge marginal | Complete | Complete | Complete | 0 |
| B1 | Size-dependent edge existence q(N) | Complete | Missing | Missing | 2 |
| D0 | Clean target-degree conditioning, global edge prior | Unimplemented | Unimplemented | Unimplemented | 3 |
| D1 | D0 plus Chung-Lu edge prior | Unimplemented | Unimplemented | Unimplemented | 3 |
| A0 | Fixed actual atoms, global edge prior | Complete | Complete | Missing | 1 |
| A1 | Fixed actual atoms, atom-based Chung-Lu edge prior | Complete | Missing | Missing | 2 |
| A2 | A1 plus endpoint-atom-specific existing-bond type prior | Unimplemented | Unimplemented | Unimplemented | 3 |
| M0 | Persistent reference atoms, mutable actual atoms, global edges | Complete | Complete | Failed before training | 1 |
| M1 | M0 plus reference-based Chung-Lu edges | Complete | Failed before training | Failed before training | 2 |

Matched scientific contrasts: B1/B0 (size-specific density); D1/D0 (degree
prior beyond identical degree conditioning); A1/A0 (atom-based edge prior);
A2/A1 (conditional bond-type prior); M1/M0 (structural prior with mutable atoms).
D0/B0, A0/B0 and M0/B0 are system-level comparisons; tasks, conditioning or
node paths differ. D0/D1 target degrees are conditions, not guaranteed output
degrees. Train with clean aligned degrees, infer with train-only empirical
degree profiles conditional on N, randomly assigned to vertices. Keep D0/D1
conditioning identical. A2 uses final fixed atoms and training-only smoothed
bond-type statistics. Existing-edge type probabilities must normalize; the
no-edge probability stays the A1 rule. Do not condition M variants on clean
target atoms or update the reference edge prior during sampling.

## Suggested sequence: three fresh batches

| Batch | Runs | Count | Reservation at 7 hours/run |
| --- | --- | ---: | ---: |
| Next | Retry M0 seed2, M1 seeds1,2; add A0 seed2, A1 seeds1,2, B1 seeds1,2 | 8 | 56 hours |
| Profiling prerequisite | D0/D1/A2 at full model/data settings, short training and representative evaluation | 3 profiles | Provisional total cap 3 hours |
| Following, after profiling | D0/D1/A2 each seeds0,1 | 6 | 42 hours if profiling supports 7 hours/run |
| Final core fill | D0/D1/A2 each seed2 | 3 | 21 hours |

User revision: the immediate next batch includes B1 seed2 and is explicitly
allowed to exceed 50 hours because of unused allocation from failed runs in
the previous batch. Its eight seven-hour limits sum to 56 hours; expected
allocation at existing runtimes is approximately 49–53 hours. This is a
specific exception, not a change to the default 50-hour budget for later batches.
Profiling can be charged to the following batch (provisionally 3 + 42 = 45 hours)
if timings support that plan. Full D0/D1/A2 training must wait for profiling
results; no automatic launch of full new variants based only on CPU checks.

Current completed runs took roughly 6.1–6.6 allocated A100-hours each.
Seventeen more baseline-equivalent runs suggest approximately 104–112 hours,
with 119 hours of full-run requested limits at seven hours/run, plus profiling.
New D/A2 timings are unmeasured: implement and verify their correctness on CPU,
then profile each on an A100 before reserving full training. Measure warm-up
separately from steady-state epochs, training/peak memory, condition/prior
construction, 500-step generation of 2k and 10k pools, strict/legacy evaluation,
and checkpoint save/resume overhead. A provisional profile can use 20–50 epochs
at the full model, batch size, dataset and loader settings, plus representative
sampling; revise the provisional cap if required. Project the exact 1000-epoch
evaluation schedule and add margin before choosing full-run time limits.
If needed, reduce the
number queued in a batch rather than silently exceeding 50 hours. Queue each
authorized batch's tasks together, without an artificial concurrency throttle.
Exclude ccc0388 and ccc0390 for these retries unless subsequently verified
healthy. Preserve existing completed runs, failed-run evidence, and budget
lineage; partial retries require a targeted submission path, not resubmission
of all seven previously queued tasks.

Both failures were the same observed failure class: CUDA unavailable while the
allocated GPU reported a reset requirement. The first was on ccc0388; the retry
was on ccc0390, a different GPU. Their underlying hardware/driver trigger was
not established from saved application logs. Retain the CUDA tensor precheck,
GPU identity/health logs and stop-on-failure behavior. Exclusions reduce repeat
exposure but cannot ensure that all remaining GPUs are healthy; cluster staff
must repair or remove unhealthy devices from scheduling. No staff message has
been sent and no further jobs have been submitted by this planning update.

## Shared reporting

Use fixed final epoch1000 validity as primary, plus the validity curve and
early validity AUC. Record strict V/U/N/VUN, raw counts, cause-specific failures,
composition/degree/bond statistics, losses and allocation/training time.
Use the current batch2 evaluation and checkpoint protocol: scheduled early
2k pools, late 10k pools with a saved 2k subset, three fresh 10k final repeats,
and no best-VUN checkpoint selection. Average repetitions within each training
seed; report each seed and matched differences before pooled summaries.

Use the saved 2k subsets for cross-batch U/N/VUN curves; keep final comparisons
at 10k. Existing seed0 checkpoints need no retraining for that comparison. An
optional evaluation-only pass can generate uniform late 10k curves from saved
seed0 snapshots; budget it separately and never present pool-size effects as
learning differences. Validity alone is not distribution fit: compare size,
composition, degree, bond type and property distributions against held-out
validation data without retuning to the test set.

## Previously proposed secondary experiments (not core completion)

Run on a chosen matched pair after the core comparisons, initially one seed,
then replicate a promising effect. Do not build a full Cartesian product.

1. Atom-degree weights mu[a] versus mu[a,N].
2. Deterministic weights versus sampled rho(w | atom type,N), retaining sampled
   weights as persistent known conditions in all prior-dependent terms.
3. Chung-Lu versus calibrated expected-degree probabilities; fit/cache beta
   parameters outside minibatch training, test density/degree fidelity and cost.
4. Empirical versus broader learned composition sampling, only if diagnostics
   establish a composition-support limitation. Distinguish fixed A support from
   mutable M outputs; test generalization to new conditions.
5. Existing structural features/RRWP on/off for B0 and the selected variant,
   with matched settings and a new baseline if preprocessing changes.
6. Optional sampler robustness: same predeclared validation-only tuning budget
   per selected model; keep the main matrix sampler fixed.

Endpoint-conditioned bond types are already A2 in the core matrix, not another
additional augmentation. Motif insertion, hard valence constraints, and evolving
Q(X_t) were explicitly deferred; they are separate methodological changes.

QM9 with explicit H was proposed as a targeted representation transfer test,
not a second full matrix. Start with B0/A0/A1 to test whether degree-based atom
priors behave differently when H is represented; optionally add D0/D1 if degree
conditioning is the relevant finding. Use consistent underlying molecule splits,
compare within each representation, and profile cost first: the no-H timing
estimate does not apply to larger explicit-H graphs. ZINC/MOSES were separate
future datasets, outside completing the QM9 matrix.
