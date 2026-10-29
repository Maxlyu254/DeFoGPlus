# Project context for future sessions

Read `docs/degree-aware-vun-plan.md` before implementing molecular prior changes,
experiment runners, or evaluation. It records the user's research objective,
agreed experiment matrix, proposed protocols, and implementation caveats from
the October 25, 2026 discussion. Update its status when work is completed or the
user changes the plan.

The vanilla Campus Cluster budget benchmark is implemented in `benchmarks/` and
documented in `docs/vanilla-budget-benchmark.md`. Original setup job 11821406
failed because the Pixi environment did not contain pip; A100 job 11821407 was
canceled without GPU use. Installer bootstrapping was repaired. Latest retry:
CPU job 11822422 and A100 job 11822489 both COMPLETED with exit 0:0. Read
`benchmarks/submissions/qm9-a100-retry-20261026T042900Z.json` before checking
results or resubmitting. Do not duplicate these jobs without checking state.

Measured results and raw timing/metric artifacts are saved in
`benchmarks/results/qm9-a100-retry-20261026T042900Z/report.md`. A100-SXM4-80GB:
18.69 s/epoch with 4 loader workers versus 22.49 s with 0; 500-step generation
takes about 61.4 s/2k or 307.8 s/10k. Projected 1000 epochs + evaluation: about
5.67 GPU-hours/seed; initially request 8 hours with margin. These are QM9-only
timings and extrapolations, not measured full 1000-epoch costs.

Current requested study: B0/B1/A0/A1/M0/M1, QM9 without H, one training seed each,
up to 1000 epochs, approximately 50 GPU-hours. The proposed execution/logging/
checkpoint/Git plan is `docs/qm9-batch1-plan.md`; read it before implementing.
Expected baseline-equivalent cost is about 36 hours, 45 with 25% margin; the
budget conservatively includes the existing 0.516-hour pilot. No new variants
or jobs were executed by that planning turn. The original nine-model,
three-seed matrix and the earlier four-model screen are superseded for now.

W&B read-only authentication was verified with SDK 0.20.1 after explicitly
sourcing `/u/xinyan14/.config/defog/wandb-env.sh`. The file is owner-readable/
writable (0600); do not print, copy into Git, or log its contents. Default entity:
`maxlyu254-university-of-illinois-urbana-champaign`. New SSH sessions/jobs must
source the file explicitly. No cloud upload was created by the authentication check.

Implementation is on `experiments/qm9-structural-priors-batch1`, based on `year2`.
See `docs/qm9-batch1-launch.md`, `scripts/batch1.py`,
`scripts/batch1_cluster.py`, and `scripts/slurm/batch1-*.sbatch`.
Preparation verification and staged paths are in `docs/qm9-batch1-readiness.md`:
10 CPU unit tests and all-six full-model smoke/resume checks passed. GPU checks
are deferred to the first step of each authorized A100 allocation.
The helper stages immutable Git archives and defaults to dry-run submission;
only `submit --submit` submits jobs. The user subsequently authorized launch:
CPU preparation 11846895, A100 array 11846896 (tasks 0-5, concurrency 2),
and dependent W&B sync 11846897 were submitted. Check their state before any
retry; do not duplicate them. The executed immutable source commit is
`cd902863f32a9a42b32f75fd95b5034e5e1eb9fc`. Submission metadata is in
`docs/submissions/qm9-batch1-11846896.json`.
All six tasks completed 1000 epochs/96,000 steps; preparation and W&B sync also
completed with exit 0:0. Assessment and plots:
`docs/results/qm9-batch1-11846896/report.md`. Total allocation was 37.85 A100-hours,
38.37 including the pilot. All VUN-selected checkpoints are at epochs 10-25;
novelty/VUN decline toward epoch 1000 despite rising validity. Best final VUN:
M1 32.19%, versus M0 31.65% and B0 30.07%. One training seed only. Do not infer
training-seed significance or launch follow-ups without authorization.
The user subsequently clarified priorities: **final validity and validity
learning speed are primary; VUN is secondary overall performance**. Do not
rank methods by best VUN. Read `docs/qm9-batch2-validity-plan.md`: A0/A1 learn
validity faster early; M0/M1 have a small unconfirmed final-validity advantage;
isolated edge-prior changes show no clear validity benefit. The proposed
100-epoch multi-seed follow-up is not submitted or authorized for launch.
The user then allocated a fresh 50-hour batch budget and requested training new
seeds until validity and VUN mostly stabilize. Read
`docs/qm9-batch2-convergence-plan.md` instead of the superseded 100-epoch scope.
Current proposal: B0/M0/M1 seeds 1,2 plus A0 seed 1; no new jobs submitted.
Latest user instruction authorizes implementation and submission of those seven
runs, all queued together, for exactly 1000 epochs, without drift checks.
Read `docs/qm9-batch2-launch.md` and use `scripts/batch2_cluster.py`. Batch2 has
its own fresh budget and uses `--array=0-6` with no throttle. Check submission
state before retrying; never duplicate the jobs.
Batch2 is now submitted: CPU preparation 11899422, unthrottled A100 array
11899423 (tasks 0-6), W&B sync 11899424. Executed source `3bd423f`.
Read `docs/submissions/qm9-batch2-11899423.json` before any retry. Do not duplicate.
Artifacts/caches stay outside Git under `/scratch/xinyan14/motiflow-experiments/`;
selected checkpoint/metric backups use `/u/xinyan14/motiflow-artifacts/`.
Code is edited locally and transferred with Git, including future fixes.

- The objective is to measure final validity, uniqueness, novelty, and their
  combined yield (VUN), as well as their evolution through training epochs.
- `year2` is the DeFoG baseline branch for this work. The older Motiflow research
  branch contains motif extensions; do not confuse it with this baseline.
- Local paper: `../papers/DeFoG, Qin et al.pdf`.
- Campus Cluster access: `ssh xinyan14-cc`; remote checkout:
  `/u/xinyan14/motiflow`. Check remote branch and working-tree state before
  synchronizing. Do not overwrite the older research branch with this checkout.
- The current checkout includes `pixi.toml`, `pixi.lock`, and `pyproject.toml`.
  Earlier discussion of a Python 3.9-only setup predates inspection of this
  branch's updated environment; inspect current environment files first.
- B0/B1/A0/A1/M0/M1 are implemented in `src/experiments/batch1`; original model
  modules remain unchanged. D0/D1/A2 remain unimplemented. Batch 1 is submitted
  under the job IDs above. The benchmark retains strict and legacy evaluators.
  Pilot scores are from 20-22 epochs, not a fully trained model. At 20 epochs and
  10k attempts, strict connected V/U/N/VUN = 84.83/97.95/62.13/51.62 percent;
  legacy relaxed scores use a different acceptance policy. Raw standalone
  checkpoint_age=0 is a logging artifact; provenance establishes epoch 20.
