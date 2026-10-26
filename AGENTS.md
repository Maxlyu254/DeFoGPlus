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
source the file explicitly. No W&B run or project write was created by the check.

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
- Research model variants remain unimplemented. The benchmark adds a separate
  strict evaluator, with legacy metrics retained; the model source is unchanged.
  Pilot scores are from 20-22 epochs, not a fully trained model. At 20 epochs and
  10k attempts, strict connected V/U/N/VUN = 84.83/97.95/62.13/51.62 percent;
  legacy relaxed scores use a different acceptance policy. Raw standalone
  checkpoint_age=0 is a logging artifact; provenance establishes epoch 20.

