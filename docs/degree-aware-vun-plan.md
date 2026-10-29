# Degree-aware DeFoG: evaluation and implementation memory

Recorded: 2026-10-25. Source: the user's discussion and requested evaluation plan.

## Objective and current status

Investigate how graph structural information in the initial distribution and
conditioning affects molecular validity (V), uniqueness (U), novelty (N), and
their combined yield (VUN). Measure both final results and learning curves.

User priority clarification (2026-10-28): final strict validity and the validity
learning curve/learning efficiency are primary. VUN is a secondary overall
performance diagnostic at matched checkpoints, not a checkpoint-selection or
variant-ranking objective by itself. See `docs/qm9-batch2-validity-plan.md` for
the reassessment and proposed confirmation study; it supersedes the suggested
fixed-epoch-25 VUN follow-up. No follow-up jobs are authorized or submitted.

Further user update: batch 2 has a fresh 50-A100-hour budget and should train
new seeds until both validity and VUN mostly stabilize. Current proposal is
`docs/qm9-batch2-convergence-plan.md`, superseding the 100-epoch follow-up:
B0/M0/M1 seeds 1,2 and A0 seed 1, with late stability reviews and budgeted
extensions. It is not a submission authorization.

Latest instruction: implement, synchronize and submit all seven runs together,
each for exactly 1000 epochs. No drift checks or adaptive extensions. No
artificial concurrency cap. Current protocol: `docs/qm9-batch2-launch.md`.
Batch2 uses a fresh 50-hour ledger and a 49-hour reservation (seven 7-hour tasks).

The user previously developed Motiflow on top of DeFoG using motif and other
structural information. This study starts from the `year2` DeFoG baseline,
without those older motif extensions. The baseline is `year2` at `a35327f`.
Implementation is now on `experiments/qm9-structural-priors-batch1` locally and
on Campus Cluster, synchronized through Git. Recheck working-tree state before
future synchronization.

B0/B1/A0/A1/M0/M1 are implemented in `src/experiments/batch1`, with source configs,
mathematical tests, checkpoint recovery, strict/legacy evaluation, offline W&B,
and guarded Slurm scripts. Original model modules remain unchanged. D0/D1/A2
are not implemented. See `docs/qm9-batch1-plan.md` and
`docs/qm9-batch1-launch.md`; the current scope is six 1000-epoch runs with one
seed and a roughly 50-GPU-hour envelope, superseding the execution order below.
After preparation, the user authorized launch: CPU job 11846895, dependent
A100 array 11846896 (B0/B1/A0/A1/M0/M1, tasks 0-5, concurrency 2), and W&B sync
job 11846897 were submitted. Check state before retrying; do not duplicate.
See `docs/submissions/qm9-batch1-11846896.json`. Executed source: `cd90286`.
All six runs subsequently completed 1000 epochs and 96,000 optimizer steps;
preparation/training/W&B sync completed with exit 0:0. Assessment:
`docs/results/qm9-batch1-11846896/report.md`. Actual allocation: 37.85 A100-hours,
38.37 including pilot. All VUN-selected epochs are 10-25. Epoch-1000 strict
10k VUN means: B0 30.07%, B1 30.57%, A0 29.03%, A1 28.87%, M0 31.65%,
M1 32.19%; selected-checkpoint fresh VUN means span 51.94%-55.79%.
Validity rises toward 99% while novelty falls; this alone does not prove
overfitting. Edge-prior effects are small/inconsistent within matched pairs.
One training seed; repeated sampling is not training-seed replication.
No follow-up experiments were submitted during assessment.
A vanilla benchmark harness and strict evaluator exist in
`benchmarks/`. CPU preparation job 11821406 and dependent A100 job 11821407 were
submitted on 2026-10-26 UTC. Setup failed for missing pip and the GPU dependency
was canceled without GPU use. After installer bootstrapping was repaired, CPU
job 11822422 and dependent A100 job 11822489 completed successfully on retry.
The measured report is
`benchmarks/results/qm9-a100-retry-20261026T042900Z/report.md`: 18.69 s/epoch with
4 workers, 61.4 s/2k and 307.8 s/10k sampling at 500 steps on A100-SXM4-80GB.
Estimated full 1000-epoch run plus planned evaluation is 5.67 GPU-hours/seed;
request 8 hours initially with margin. Strict 20-epoch/10k VUN is 51.62%; this
is an early snapshot, not a final trained result. See
`docs/vanilla-budget-benchmark.md` and the submission JSON under
`benchmarks/submissions/` before continuing. The budgets and optional extensions
below are proposals to refine after baseline profiling.

## Background and modeling constraints

Read the local paper `../papers/DeFoG, Qin et al.pdf`, especially Section 3.1,
Section 3.3, and Appendices B.5, C.1, D.2.1. Online version:
https://arxiv.org/html/2410.04263v3

Original DeFoG has shared categorical node and edge priors, independent
coordinates conditional on graph size, and the probability path

    p_t(z | z_1) = t * delta(z, z_1) + (1-t) * p_0(z).

The prior determines training corruption, initial sampling, and conditional
sampling rates. Training/sampling decoupling does not authorize an arbitrary
replacement of the prior of a trained model. Full support is not required:
masking and one-hot priors are examples in the paper.

A proposed extension uses a fixed auxiliary condition c (size, degrees, or
reference atom identities) and pair-specific categorical priors:

    pi_ij(0 | c) = 1 - q_ij(c)
    pi_ij(k | c) = q_ij(c) * r_k(c), k > 0
    p_t(e_ij=k | e_1,c) = t * delta(k,e_1_ij) + (1-t) * pi_ij(k|c)
    derivative = delta(k,e_1_ij) - pi_ij(k|c).

Here 0 means no edge; r is normalized over existing-edge types. It can be global
or depend on the two endpoint atom types. Edges remain independent conditional
on c. Marginalizing a random c generally introduces edge dependencies.

All prior-dependent components must use the same probabilities: training
corruption, initial sampling, path derivatives, and sampling-rate terms.
Recomputing Q from evolving atom identities or degrees is a different process;
it is not justified by substituting into the fixed-prior interpolation above.

### Degrees, valence, and prior construction

- Degree counts neighbors in the represented graph; valence involves bond
  orders, hydrogens, charge, and representation conventions. QM9 without H uses
  heavy-atom degree. Do not set carbon's degree target to 4 merely from valence.
- Estimate atom weights from training data, initially
  `mu[a,N] = E[represented degree | atom type a, size N]`. Consider global mu[a]
  or smoothing when size-specific data are sparse.
- Simple proposed Chung-Lu-style rule, for distinct vertices:
  `q_ij = min(1, w_i*w_j/sum(w))`; set all q to zero if the sum is zero.
- This loop-free, clipped variant only approximately matches expected degrees.
  Without clipping, expected degree is `w_i*(1-w_i/sum(w))`.
- Independent edges control expected degree, not exact realized degree or hard
  valence budgets. Do not claim guaranteed chemical validity.
- A calibrated beta-model alternative uses
  `q_ij = sigmoid(beta_i+beta_j)`, fitting row sums to desired expected degrees.
  Boundary/infeasible targets need treatment; fit/cache parameters in
  preprocessing rather than running iterative optimization every minibatch.

### Symmetry

Heterogeneous pair probabilities are compatible with permutation symmetry when
conditions move with vertices:

    Q(P*c) = P*Q(c)*P^T
    p_0(P*A*P^T | P*c) = p_0(A | c).

Require symmetric probabilities and undirected edges, mask padding and the
diagonal, and jointly permute graphs and their conditions. Randomly assign
sampled degree profiles or atom multisets to vertices for an exchangeable
unconditional generator. Node-order changes do not create novel molecules.

### Conditioning, composition, and likelihood distinctions

- Fixed atoms means each vertex retains its atom identity through one
  trajectory. Different trajectories receive different sampled atoms. This is
  stronger than fixing only atom counts.
- In hydrogen-suppressed graphs, composition refers to represented atom counts,
  not necessarily a complete molecular formula including implicit H.
- Sampling empirical compositions restricts final composition support in a
  fixed-atom model, but does not inherently lower molecular novelty/uniqueness.
  Familiar compositions can yield unseen structures. Do not assume a major
  diversity penalty without evidence.
- Allowing atom updates alone does not establish composition revision. If the
  clean target atoms are supplied as permanent conditions, training can simply
  reconstruct them. To study genuine final composition changes, training pairs
  must allow different initial/reference and target atom assignments.
- Keeping a reference condition fixed is distinct from keeping actual evolving
  atom labels fixed. The M variants below retain the former, not the latter.
- Conditional prior probability is a product over edges AND nonedges. An
  unconditional latent-degree prior requires averaging that product over the
  degree-vector distribution and assignments. Substituting observed graph
  degrees is not that marginal probability.
- Training conditional flows does not require evaluating this expensive
  marginal likelihood or enumerating degree assignments. One known condition
  per example and direct noising at a sampled time suffice.
- With deterministic weights from observed fixed atom types, the hierarchical
  prior is `p_N(N)*rho_N(atom_types)*product_ij pi_ij(edge|atom_types)`.
  Initial-prior probabilities are not final DeFoG graph likelihoods.

## Dataset priorities

| Dataset | Role |
| --- | --- |
| QM9 without hydrogens | Full initial ablation study |
| ZINC250k without hydrogens | Transfer to larger molecular graphs |
| MOSES | Larger-scale confirmation and diversity evaluation |
| QM9 with hydrogens | Targeted degree-versus-valence diagnostic |
| GuacaMol | Optional final confirmation |

Use existing dataset splits and matched preprocessing within each dataset.
Compute degree statistics, bond marginals, and composition samplers from the
training split only. Smooth sparse conditional estimates using training data.

## Main experiment matrix

### Current budget preference (2026-10-26)

Latest user request supersedes the earlier screen recommendation below:
include B0/B1/A0/A1/M0/M1, one training seed each, QM9 without H, through 1000
epochs within approximately 50 GPU-hours. See `docs/qm9-batch1-plan.md` for the
new proposed execution plan, 17 evaluation points, checkpoints and storage,
W&B mirroring, Git management, and roughly 36-hour central/45-hour contingency
estimate. This planning update did not implement variants or submit jobs.

### Earlier lean-screen recommendation (superseded by six-model request)

The user does not consider these ideas ready for a roughly 200-GPU-hour study
and requests a less rigorous prioritization. The nine-configuration/three-seed
matrix below remains a reference, not a commitment to execute it.

Recommendation, not yet an executed or confirmed reduced matrix: begin with
B0, B1, A0, A1 on QM9 without H, one training seed, initially 200 epochs. Keep
A0 as the matched control for A1; otherwise conditioning on final atom types is
confounded with changing the edge prior. Defer D0/D1 (explicit target-degree
conditioning), A2 (additional bond-type prior), and M0/M1 (joint composition
revision). If revisable compositions become a requirement, promote M0/M1 as a
pair. An absolute minimum atom-aware study is B0/A0/A1, deferring B1 too.

Three training seeds are for confirmation of promising comparisons, not initial
feasibility. Repeated sampling of one checkpoint measures sampling uncertainty
but cannot replace independent training runs. Prefer one training seed and
matched sampling conditions for screening; add a second training seed only to
the winner and its direct matched control. A third seed remains an optional
confirmation step, especially for small or inconsistent differences. Do not
claim statistically established superiority from a one-seed screen.

Baseline-equivalent costs: four 200-epoch screens about 4.7 GPU-hours (allow
about 6 with initial margin); three screens about 3.6. Completing all four to
1000 epochs with one seed would be about 23-24 GPU-hours, including screening
if resumed. Adding a second full training seed for one matched pair would bring
that optional campaign to about 34-36 GPU-hours. No new jobs were submitted.

Do not automatically reject a model solely for an ambiguous 200-epoch result;
compare trajectories and consider extending a promising matched pair. Keep
sample count, evaluation policy, and sampler settings matched. Our baseline
20-epoch 2k sampling repetitions had VUN standard deviation of 1.40 percentage
points; this is an example of sampling noise, not an estimate of training-seed
variance or a universal threshold.

### Original full reference matrix

Budget estimate for completing all nine configurations on QM9 with three seeds:
`docs/main-ablation-budget.md`. Baseline-equivalent cost is 153-160 A100
GPU-hours; about 200 hours with 25% contingency, or 216 hours summed across
27 eight-hour reservation caps. The initial 200-epoch screen is about 11 hours
and is included if its checkpoints are continued rather than discarded.

| ID | Configuration | Intended comparison |
| --- | --- | --- |
| B0 | Original shared node/edge marginal priors; joint atom/bond generation | Baseline |
| B1 | B0 with edge-existence probability q(N); global bond-type marginals conditional on existence | B1 vs B0: size-dependent density |
| D0 | Degree-target conditioning; shared edge marginal prior; joint atom/bond generation | Control for degree information alone |
| D1 | D0 with Chung-Lu-style pair probabilities | D1 vs D0: degree-aware prior beyond conditioning |
| A0 | Sample and fix atom identities; shared edge marginal prior; generate bonds | Control for knowing final atoms |
| A1 | A0 with weights from atom-type-dependent expected degree and pair-specific q | A1 vs A0: atom-aware connection prior |
| A2 | A1 plus endpoint-atom-type-dependent bond-type marginals | A2 vs A1: bond-type prior |
| M0 | Sample reference atoms; actual atoms can change; shared edge marginal prior | Control for composition-changing construction |
| M1 | M0 with edge prior from the fixed reference atom identities | M1 vs M0: structural initialization with changing atoms |

### Training and sampling semantics

**B variants:** retain ordinary node generation. For B1, estimate q(N) from
training data; N is known in both training and sampling. The probability of a
bond type is q(N) times its conditional existing-edge frequency.

**D variants:** during training use the clean graph's degree vector as a fixed
node condition. At inference sample training degree profiles conditional on N
and randomly assign them to vertices. Keep each condition aligned with its
target vertex in training; do not permute degree targets independently of the
clean graph. D0/D1 must have identical conditioning/features.

**A variants:** use clean atom identities as fixed training conditions. Sample
compositions from the same training-based sampler and assignments for A0/A1/A2
at inference. Only bonds evolve; atom-type denoising is unnecessary. Initially
use identical composition draws across these variants.

**M variants:** for each clean target G_1, independently sample reference atom
identities from the training distribution conditional on the same N. Keep this
reference available as an auxiliary node condition while the actual atom labels
evolve. A consistent node path starts from a point mass on the reference label
and ends at the clean target label. M0/M1 must share this node path, input
conditions, and reference sampler. M1 computes Q from the reference once and
keeps it fixed; it does not recompute Q from noisy atoms. The reference controls
initialization, not the desired final composition. Record actual C_1 vs C_0.

A vs M comparisons change the modeling task and reference/target coupling;
they are system-level comparisons, not an isolated edge-prior ablation.

## Secondary augmentations

Apply these to promising matched models, not as a full factorial search:

1. Atom weights mu[a] versus mu[a,N].
2. Deterministic weights versus sampled weights rho(w | atom type,N). Retain
   sampled weights as known fixed conditions if the prior depends on them.
3. Chung-Lu-style versus calibrated expected-degree probabilities.
4. Global versus endpoint-atom-pair-specific bond-type marginals (A1/A2).
5. Empirical versus broader learned composition sampler, only if composition
   support is shown to limit diversity. Validate new-condition generalization.
6. Existing structural features/RRWP on versus off for B0 and the strongest
   variant. Keep them enabled in the main study.

Defer motif insertion, hard valence enforcement, and adaptive Q(X_t) until the
simpler changes are understood. No claim is made that broader composition
sampling, stronger degree matching, or any proposed prior improves VUN.

## Evaluation contract

For M generated attempts, let n_v be accepted valid molecules including
duplicates, n_u be their distinct canonical molecules, and n_n be the distinct
accepted molecules absent from the canonical training reference.

    V = n_v / M
    U = n_u / n_v
    N = n_n / n_u
    VUN = V * U * N = n_n / M

Store all counts. Handle empty valid/unique sets explicitly; VUN is zero when
there are no novel valid molecules. Percentage displays must not change the
fraction-based calculation. Use consistent identity/stereochemistry/charge
conventions for generated molecules and the training reference.

**Existing evaluator issue:** `BasicMolecularMetrics.evaluate()` in
`src/analysis/rdkit_functions.py` reports strict validity but computes U/N from
`compute_relaxed_validity()`, including partial-charge handling and largest
fragment extraction. Do not multiply the existing strict V by those U and N.

Proposed primary protocol: strict sanitization with an explicitly documented
connectivity policy, consistent canonicalization, and no fragment/correction
transformation silently changing molecule identity. For connected-molecule
datasets, use strict connected validity as the main result. Preserve legacy
relaxed V/U/N/VUN as a separate, clearly labeled compatibility result. Confirm
the exact evaluation contract before interpreting benchmark comparisons.

Novelty is against training, not held-out test molecules. All reference-driven
checkpoint/sampler selection uses validation; reserve held-out test reference
metrics and fresh reporting samples for final assessment.

Additional diagnostics:

- Degrees and bond-order sums by atom type; degree/valence error where relevant.
- Connectivity and causes of sanitization/valence failures.
- Composition retention/change before any fragment extraction, composition
  frequencies, and uniqueness within common compositions.
- Training node/edge CE, epoch, optimizer step, training GPU time, peak GPU
  memory, and sampling time/cost.
- Optional scaffold diversity/distribution metrics for finalists; current
  `src/metrics/molecular_metrics.py` explicitly disables FCD despite some
  dataset configuration flags, so do not assume it is being evaluated.

## Learning curves and final reporting

Starting full budgets, from current presets:

| Dataset | Epochs | Proposed evaluation epochs |
| --- | --- | --- |
| QM9 without H | 1000 | 0,1,5,10,25,50,100,200,400,600,800,1000 |
| ZINC250k | 300 | 0,1,5,10,25,50,100,150,200,250,300 |
| MOSES | 300 | 0,1,5,10,25,50,100,150,200,250,300 |

These are proposed budgets, not demonstrated convergence points. Profile B0
first and revise if necessary. Treat epoch 0 as a separately labeled untrained
snapshot; record completed epochs and optimizer steps to avoid off-by-one
confusion with Lightning's zero-based epoch index.

- Curve evaluation: 2,000 generated attempts per scheduled checkpoint.
- Final evaluation: 10,000 attempts per independent sampling repetition; also
  report the matched 2,000-attempt score/subset for curve comparability.
- Uniqueness and VUN depend on batch size. Never compare raw U/VUN at 2k and 10k
  as if they had the same evaluation budget. Name the sample count in results.
- Use matched sampling/condition seeds where meaningful across configurations;
  use fresh sampling seeds for final reporting.
- Report mean and standard deviation across three training seeds for completed
  comparisons. Distinguish sampling repetitions from independent training runs.
- Plot V, U, N, VUN against epoch, optimizer step, and training GPU time.
- Report both the final-epoch checkpoint and the checkpoint selected using the
  validation sampling protocol. Do not choose the best checkpoint on final
  reporting samples or test metrics.
- Keep canonical samples, raw generated graphs, counts, checkpoint IDs, resolved
  configs, git revision, environment details, seed values, and timing records.

Hold model capacity, optimizer, effective batch size, preprocessing, and existing
features fixed within matched comparisons where possible. Use a common sampler
within each dataset: identical step count, time distortion, stochasticity eta,
and guidance omega. Exact common sampler values are not yet selected; existing
dataset defaults differ. Disable sample.search during learning curves.

For finalists, optionally perform a separate equal-budget validation search of
sampler settings. Report fixed-sampler and tuned-sampler results separately.
Use benchmark-specific supplemental protocols if needed for published-number
comparison; do not silently mix them with the common internal 10k protocol.

## Implementation map and remaining work

Relevant existing code:

- `src/graph_discrete_flow_model.py`: training_step, apply_noise, forward,
  sample_batch, sample_p_zs_given_zt, on_validation_epoch_end,
  on_test_epoch_end, evaluate_samples.
- `src/flow_matching/noise_distribution.py`: global categorical priors.
- `src/flow_matching/utils.py`: interpolation and time derivative.
- `src/flow_matching/flow_matching_utils.py`: initial/corrupted sampling,
  symmetry, masking.
- `src/flow_matching/rate_matrix.py`: all prior-dependent rate terms.
- `src/datasets/abstract_dataset.py`: input-feature dimension inference.
- `src/models/extra_features_molecular.py`: existing molecular features.
- `src/analysis/rdkit_functions.py`: acceptance, canonicalization, U/N.
- `src/metrics/molecular_metrics.py`: references, logging, optional metrics.
- `src/main.py`: Trainer/checkpoint callbacks.
- `configs/experiment`, `configs/model`, `configs/sample`: current presets.

Implementation requirements:

1. Preprocess/cache train-only statistics and profile distributions, documenting
   smoothing, size bins, representation, and reference/target pairing.
2. Represent conditions explicitly in batched data; preserve them through
   sampling steps and jointly permute/mask with vertices.
3. Support per-graph/per-pair priors instead of assuming a shared [K] vector;
   audit category counting, broadcasting, derivatives, and all rate terms.
4. Add permanent node conditions (degree/reference atom features) and adjust
   input dimension inference. Keep architectural changes matched in controls.
5. Preserve direct noising at a sampled t and ordinary CE. No trajectory
   simulation, marginal-likelihood integration, or extra graph enumeration is
   required in training.
6. Keep Q on device during a sampling trajectory. Simple construction costs
   O(B*N^2) extra work/storage; existing dense edge tensors already scale with
   N^2. Actual overhead remains unmeasured. Cache expensive calibration fits.
7. Implement the consistent evaluator, structured per-checkpoint records, and
   explicit evaluation/checkpoint schedule. Existing periodic validation and
   checkpoint intervals alone do not implement the proposed irregular schedule.
8. Audit fold handling before using `general.num_sample_fold`; clearly separate
   per-fold evaluation, pooled evaluation, sampling repeats, and training seeds.
9. Validate probability normalization, t=0/t=1 endpoints, masking, symmetry,
   reduction to the shared-prior baseline, condition persistence, and metric
   counts when implementing these consequential changes.
10. Keep experiment IDs/configuration fields in the separate batch1 entry point.
    B0/B1/A0/A1/M0/M1 are implemented there; D0/D1/A2 remain future work.

## Execution order

1. Screen B0/B1/D0/D1/A0/A1/A2/M0/M1 on QM9 without H using one seed to 200
   epochs, with learning-curve samples.
2. Complete the main QM9 comparisons with three training seeds, retaining
   matched controls, to the profiled full budget (initially 1000 epochs).
3. Transfer B0 and strongest matched comparisons to ZINC250k.
4. Confirm B0 and strongest two or three alternatives on MOSES.
5. Add explicit-H, calibration, broader-composition, or feature diagnostics in
   response to measured limitations.

Final table: model ID, last-epoch V/U/N/VUN, validation-selected checkpoint
V/U/N/VUN, training GPU hours, sampling cost, and uncertainty across seeds.

## Campus Cluster context (snapshot, not a current availability guarantee)

SSH alias `xinyan14-cc` was verified as user `xinyan14`. Remote repo:
`/u/xinyan14/motiflow`. Scratch: `/scratch/xinyan14`; reported quota was 10 TB,
with 100 GB home quota. No environment installation or jobs were performed as
part of this planning discussion.

Slurm test-only requests accepted these account/partition combinations:

- `26sp-cs525s-eng` / `eng-instruction`: A10, 12-hour partition limit.
- `xinyan14-ic` / `IllinoisComputes-GPU`: A100 request accepted; partition also
  lists H200 nodes, 72-hour partition limit.
- `xinyan14-ic` / `ic-express`: H100 20 GB MIG slices, 8-hour partition limit.

Secondary offers multiple GPU types with a 4-hour limit; scavenger offers idle
GPUs with a 24-hour limit and preemption. Recheck Slurm configuration and access
before scheduling. Start with single-GPU jobs; baseline configuration says
multi-GPU is unsupported. Slurm limits and installed environments may change.
