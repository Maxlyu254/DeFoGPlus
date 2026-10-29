# First molecular ablation batch: six configurations, approximately 50 GPU-hours

Updated 2026-10-26 following the user's request. The six variants and guarded
launch scripts are now implemented on `experiments/qm9-structural-priors-batch1`.
See [`qm9-batch1-launch.md`](qm9-batch1-launch.md) for current launch/storage
instructions. Following the user's launch authorization, CPU preparation
11846895, A100 array 11846896 and W&B sync 11846897 were submitted using
immutable source commit `cd90286`. Check these jobs before any retry.
All runs subsequently completed successfully through 1000 epochs. Results and
assessment: [`report`](results/qm9-batch1-11846896/report.md). Actual allocation
including the pilot was 38.37 A100-hours; early checkpoints yield higher VUN
than epoch 1000. No follow-up batch is authorized by this assessment.
Preparation checks and the staged batch path are recorded in
[`qm9-batch1-readiness.md`](qm9-batch1-readiness.md).
It supersedes the earlier proposed four-model/200-epoch screen as the current
scope. The nine-model/three-seed study remains a reference only.

## Scope

QM9 without explicit hydrogens; B0, B1, A0, A1, M0, M1, each to 1000 completed
epochs, with one training seed (0). Keep the existing 9-layer backbone,
float32, batch 1024, four loader workers, RRWP/molecular features, optimizer,
learning rate, and train-time distortion matched wherever applicable.

| ID | Atoms during generation | Edge-existence prior |
| --- | --- | --- |
| B0 | Jointly generated using original DeFoG | Original global marginal |
| B1 | Jointly generated as in B0 | Size-conditioned q(N) |
| A0 | Sampled first and fixed | Original global marginal |
| A1 | Same fixed-atom construction as A0 | Atom-type/size degree weights and Chung-Lu-style Q |
| M0 | May differ from the sampled reference atoms | Original global marginal |
| M1 | Same node/reference construction as M0 | Q computed once from reference atoms |

Use global existing-bond-type marginals in all six; pair-specific bond types
(A2), explicit degree-target conditions (D0/D1), broader composition samplers,
and sampler hyperparameter searches are outside this batch.

### Reference/target pairing

A0/A1 train conditional on the clean target's atom identities. At generation,
sample empirical training compositions conditional on N and uniformly assign
their atoms to vertices; keep the atom identities fixed. Share the composition
sampler and evaluation draws between A0/A1.

M0/M1 train with reference atom identities sampled independently of the target
graph conditional on N. Retain those reference identities as auxiliary node
conditions while the actual node variables evolve. Use a point-mass initial
node distribution at each reference atom, interpolated toward its clean target
label. At inference reference identities also supply the initial atom labels.
Share all this between M0/M1; only their edge priors differ. Q is fixed during a
trajectory, not recomputed from evolving atoms. Validate zero-support prior
handling and rates explicitly before long runs.

Reference atoms in M are initialization information, not the desired final
composition. A-vs-M comparisons change the task/reference-target coupling;
their results cannot be attributed solely to the edge prior.

### Priors and conditioning statistics

Fit new statistics using only processed training graphs in the actual
hydrogen-suppressed representation:

- Edge-existence rates q(N), with a recorded sparse-bin fallback.
- Expected represented neighbor count mu[atom_type,N], smoothed toward the
  global atom-type mean; freeze the smoothing rule before comparing models.
- Existing-bond-type frequencies, normalized after excluding no-edge.
- Empirical atom-count profiles conditional on N and a sampler with random
  assignment to vertices.
- Canonical training SMILES set and atom-composition set for novelty checks.

For A1/M1 use the same degree-weight mapping and
q_ij=min(1,w_i*w_j/sum(w)); zero sum gives no edges. This is an approximate
expected-degree construction, not a hard degree or valence constraint. Keep B0
faithful to the original supplied priors. Hash the dataset/split, references,
statistics, smoothing configuration, and generated prior-cache files.

## GPU budget and execution

Reference measurements: 18.69 s/epoch on A100-SXM4-80GB; a 2k evaluation takes
62.10 s including strict metrics and serialization, and a 10k evaluation 311.56
s. Six models are projected at baseline-equivalent throughput until measured.

| Component | A100 GPU-hours, approximately |
| --- | --- |
| Six 1000-epoch training runs | 31.15 |
| Seventeen 2k curve evaluations per run | 1.76 |
| Two checkpoints/run, three 10k final repetitions/checkpoint | 3.12 |
| Startup and rolling checkpoint I/O | <0.1 |
| Central total | 36.1 |
| Central total plus 25% contingency | 45.1 |

Count the existing 0.516-GPU-hour pilot conservatively against the user's
approximately 50-hour envelope. Six full-run reservation caps of eight hours
sum to 48 hours; the previous pilot then leaves about 1.48 hours of additional
GPU verification capacity even if all six jobs consume their caps. Expected
actual consumption is lower. CPU preparation/statistics work is logged
separately and does not occupy an A100.

Maintain an allocation ledger containing every experiment/debug/retry job ID,
state, elapsed GPU allocation and projected remaining work. Reconcile with sacct
before retries/continuations; no automatic seed expansion or extra variants.
Save state before a time limit. Do not exceed the batch envelope silently.

Implement/test on CPU first, then a short full-model GPU smoke for each mode.
Profile the first epochs of each new variant and audit progress at epoch 200,
but continue every valid configuration to 1000; this is not a performance-based
early-elimination screen. Preserve checkpoint state across any job split.

## Statistics and their purpose

### Every epoch

Record completed epoch and optimizer step, node CE when atoms are learned,
edge CE, learning rate, training seconds, graphs/steps per second, CPU/GPU
memory, resource utilization, and cumulative allocated GPU-hours.

Node loss is not applicable to fixed-atom A0/A1; do not report its absence as a
node-prediction improvement. Losses under different corruption paths are
diagnostics, not the shared model-selection criterion.

Use a fixed held-out validation subset and saved noise seeds for a small
validation reconstruction-loss diagnostic; log the subset fingerprint and
time bins. Never fit prior statistics on it. Generated validation VUN is the
shared model-selection signal.

### Generated evaluation schedule

Use completed epochs:

    0,1,5,10,20,25,50,100,200,300,400,500,600,700,800,900,1000

Each uses 2000 generation attempts, 500 steps, polydec, eta=omega=0, inference
batch cap 2048, and fixed documented validation-generation seeds. This adds
early epoch 20 and fuller late coverage relative to the original 12-point
proposal, at roughly 0.52 extra GPU-hours across all six models. Epoch 0 is an
untrained diagnostic and is excluded from best-checkpoint selection.

Record raw attempted/valid/unique/novel counts and strict connected V/U/N/VUN,
using one consistent sanitization/canonicalization/reference policy:

    V=n_valid/n_attempted
    U=n_unique/n_valid
    N=n_novel/n_unique
    VUN=n_novel/n_attempted.

Keep legacy relaxed scores separately labeled. All generated graphs are saved
before any correction/fragment extraction so alternate evaluators remain
possible. Use identical graph-size draws across configurations; share initial
composition draws within matched A/M pairs. Do not replace B0's original node
prior merely to force it to have those same compositions.

Log connectedness, sanitization/valence failures, degree distributions and
bond-order sums by atom type, initial/final composition histograms, and VUN
within sufficiently populated composition groups. These diagnose validity
changes, concentration within compositions, and actual M0/M1 atom revision.
Record atom-identity changes and normalized atom-count-vector changes relative
to the initial reference before any fragment extraction. A0/A1 should show
zero atom/composition changes by construction.

For a small fixed subset of an evaluation batch, record aggregate diagnostics
at t=0,.25,.5,.75,1 without additional denoiser evaluations; do not retain every
full trajectory or generate videos by default. Profile diagnostic overhead.

### Final evaluation

Retain final epoch 1000 and the trained checkpoint with highest strict 2k
validation-generation VUN (earliest epoch breaks exact ties). Use three fresh
10k sampling seeds per checkpoint. If selected and final are identical,
evaluate once. Final samples do not influence checkpoint or sampler selection.

Report individual repeats and their mean/sample SD, separately from epoch-curve
2k scores. This measures sampling uncertainty for one trained model, not
training-seed uncertainty. Fix sampler settings; no test-set-driven tuning.

## Checkpoint retention and space

- Immutable snapshots at all 17 scheduled evaluation epochs, labeled by
  completed epoch; the initial snapshot has no trained optimizer state.
- Update a rolling latest recovery checkpoint at least every 25 completed
  epochs, on clean exit, and at a safe boundary after scheduler warning.
- Keep one previous rolling recovery version until a replacement is verified.
  Use atomic writes/rename; never overwrite an evaluation snapshot.
- best/final are manifest pointers or links to retained snapshots, not extra
  physical copies.
- Full resumable snapshots contain weights, optimizer (and scheduler if used),
  completed epoch/step, resolved config, code/env/data/statistics fingerprints,
  Python/NumPy/Torch CPU/CUDA RNG states, and loader/sampler generator state.
  Test epoch-boundary resumption; do not promise bitwise replay without testing.

Measured ordinary checkpoint size was 122.1 MiB; use 120-130 MiB for planning.
At most 17 snapshots plus two rolling files gives about 2.3 GiB/run, 13.6 GiB
across six runs at the measured size. Initial snapshots and links can reduce
actual use. Reserve 20 GiB for checkpoints and 30-40 GiB total working space
including raw samples/logs/statistics, excluding the shared environment/dataset.

At baseline serialization sizes, all curve samples and up to six final 10k sets
per model total about 0.7 GiB across six models, before any format change.

Scratch holds complete working runs. After evaluations, back up statistics,
manifests, and the latest/best/final distinct checkpoints to a home-directory
archive. Mirror small artifacts to the local workspace. Full checkpoint
downloads are optional; raw snapshots need not be copied to the laptop.
Scratch alone is not the only retained copy of selected model results.

## Statistics storage and W&B

Canonical records are append-only local JSONL plus CSV exports and a final
summary JSON. Evaluation rows include logical run ID, config/seed, completed
epoch, checkpoint fingerprint, sampler seed, sample count, strict/legacy policy,
counts, metrics and timings. These remain usable without a W&B account/network.

Proposed cluster layout:

    /scratch/xinyan14/motiflow-experiments/qm9-batch1/<config>/seed-0/<run-id>/
      manifest.json
      resolved-config.yaml
      checkpoints/
      metrics/train.jsonl
      metrics/eval.jsonl
      metrics/summary.json
      samples/
      logs/
      wandb/

Plan to mirror scalar statistics, sanitized configs, provenance and final
aggregate tables to W&B. Proposed project name: `defog-qm9-batch1`; authenticated
default entity: `maxlyu254-university-of-illinois-urbana-champaign`. Read-only API
authentication with SDK 0.20.1 was verified after sourcing
`/u/xinyan14/.config/defog/wandb-env.sh`; the credential file was corrected to
0600 permissions. Scripts must source it explicitly without logging its
contents. Project creation/write access has not been tested. Record offline
during training and sync the named
run directories afterward from a network-enabled host. Account authentication
is required for cloud sync; no run or metric uploads were created by the
authentication check. Credentials stay outside this repository.

Define custom axes for completed epoch, optimizer step and cumulative training
time; keep validation-generated selection metrics distinct from final-report
metrics. Use a logical run ID per configuration/seed, and segment IDs/parent
links for interrupted offline sessions. Do not assume online-style resume
semantics for offline SDK runs. Local logs stitch segments without duplicates.

Upload explicit, small allowlisted artifacts, not wildcard txt files or all
working files. Do not upload model checkpoints, datasets, raw molecule pools,
or full trajectories to W&B by default. Keep W&B a dashboard mirror; local
JSONL/CSV and checkpoint archives are authoritative. Check installed SDK behavior
with a small test before relying on sync/resume features.

## Code management and reproducibility

Develop locally on a new branch from year2, proposed
`experiments/qm9-structural-priors-batch1` (now created). Preserve year2 as the baseline and
leave the older Motiflow branches intact. Commit tested implementation before
launching long jobs. Source belongs in origin (the user's DeFoGPlus fork), not
the upstream DeFoG repository. Implementation is committed locally and
synchronized to Campus Cluster and origin with Git; no cloud metrics or Slurm
jobs are created by staging code.

Version in Git:

- Model/prior/conditioning changes in src, matching Hydra configs, training and
  evaluation entry points, Slurm scripts, preprocessing/statistics-generation
  code, meaningful tests, environment locks and documented compatibility fixes.
- Plans, documentation, small curated aggregate result tables/reports, and
  sanitized run-index/provenance manifests (including job IDs and artifact paths).
- Benchmark scripts/docs and curated aggregate results are included in the
  experiment branch; raw benchmark output remains ignored.

Keep outside Git:

- Checkpoint binaries, downloaded/processed data, generated prior-statistic
  tensor caches, raw samples/trajectories, full logs and W&B directories, large
  result artifacts, environments/package caches, and authentication material.
- A generated resolved config lives with each run; its source/default configs
  and small provenance report are versioned. Dataset split/statistic hashes
  are versioned metadata, not a reason to commit the full data files.

Extend .gitignore for experiment artifacts and checkpoints before launching.
Keep curated reports separate from ignored raw data, including existing
`benchmarks/results/**/raw` and `setup` directories. Do not indiscriminately add
all currently untracked workspace files or the user's editor settings.

Transfer a clean commit archive to an immutable per-run source directory on the
cluster, verify hashes, and record commit + lock + split + prior-statistics
fingerprints in the manifest. Avoid live mutable checkouts in running jobs and
avoid cluster-only code edits. Fix locally, commit a new revision, then launch a
new attempt snapshot linked to the original logical run. Runtime manifests and
job accounting document retries without overwriting previous attempts.
