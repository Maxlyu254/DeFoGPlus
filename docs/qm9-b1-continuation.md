# Authorized B1 seed1 continuation

User instruction on 2026-11-12: submit continuation to finish exactly1000
epochs, without a GPU-hour budget cap, but with a generous wall-time safeguard;
email xinyan14@illinois.edu when finished.

Plan: resume the full epoch825 checkpoint (optimizer, callback and RNG state)
from the original immutable source19e1288. The original trajectory logged848
epochs, so archive all original metrics and preserve a checkpoint copy before
trimming active train.jsonl to epochs1–825. The replayed epochs826–848 will then
appear once in active history. Original offline W&B history remains preserved;
the new attempt gets its own offline session in the same project.

`scripts/continue_b1.py` stages immutable operational source, verifies checkpoint
SHA256/signature/global step, archives evidence, and requires explicit --submit.
It refuses duplicate submissions and existing summaries. The GPU job uses
the original batch3 config and data/cache paths, seed1, target1000 epochs,
and the same curve/checkpoint/final-three-10k reporting protocol.

Resources: one A100, eight CPU cores,32 GiB; ccc0388 and ccc0390 excluded.
Expected remaining work: roughly1.5–2 hours at measured rates. Safeguard:
**12 hours**, below the live partition72-hour maximum. There is no allocation
budget check against the prior batch's remaining4.23 hours. Actual continuation
allocation is recorded separately and can be added to historical cost.

GPU completion triggers an afterany CPU sync/notification job. It uploads only
new B1 offline sessions, checks the epoch1000 summary and GPU completion state,
and emails the standard Slurm END/FAIL notification. The continuation's own
summary.json records outcome and cost. Original batch-summary.json remains a
historical report of the original7/8 outcome; the new continuation record is
the authoritative follow-up. No other model is restarted.

Operational source is edited locally and synced through Git. Training source
and prior immutable batch are not edited. Runtime backups, checkpoints and
reconciliation records stay outside Git under the continuation root.
Submission IDs will be recorded after submission; check them before retrying.
