# Batch2 execution assessment

Checked live Slurm accounting and saved allocation logs on 2026-10-29.

**No research results were produced.** All seven array tasks failed during the
initial CUDA smoke check, before the full training command. There are no run
summary.json files, training histories, or model checkpoints in the run folders.
Final validity, validity learning speed, and VUN cannot be compared across the
new seeds. Batch1 findings remain unconfirmed by this batch.

| Job | Outcome | Allocated elapsed time |
| --- | --- | --- |
| Preparation 11899422 | COMPLETED, exit 0:0 | 2m28s CPU |
| Array 11899423_0: B0 seed1 | FAILED, exit 1:0 | 39s A100 |
| Array 11899423_1: B0 seed2 | FAILED, exit 1:0 | 24s A100 |
| Array 11899423_2: A0 seed1 | FAILED, exit 1:0 | 28s A100 |
| Array 11899423_3: M0 seed1 | FAILED, exit 1:0 | 23s A100 |
| Array 11899423_4: M0 seed2 | FAILED, exit 1:0 | 29s A100 |
| Array 11899423_5: M1 seed1 | FAILED, exit 1:0 | 30s A100 |
| Array 11899423_6: M1 seed2 | FAILED, exit 1:0 | 31s A100 |
| Sync 11899424 | COMPLETED, exit 0:0 | 12s CPU |

All tasks ran on ccc0388 and failed in torch.cuda.get_device_name(), called
before model training, with `RuntimeError: No CUDA GPUs are available`.
The allocation's nvidia-smi logs report `GPU requires reset` and
`GPU Recovery Action: Reset`. This is direct evidence of an unhealthy allocated
GPU, rather than a failure of any model configuration. A successful nvidia-smi
exit alone did not establish that CUDA could initialize the device.

The dependent sync uses afterany: its successful exit does not imply successful
training. No completed training results were available to synchronize.

Total allocated A100 time: 204 seconds = 0.05667 GPU-hours; approximately
49.9433 hours remain from the fresh 50-hour budget. These are allocation times,
not productive GPU-compute time.

Evidence root:
`/scratch/xinyan14/motiflow-experiments/qm9-batch2/batch2-fixed1000-20261028/`.
See `logs/defog-batch2-11899423_*.out`,
`runs/*/logs/gpu-*.txt`, and `budget-ledger.json`.

Recommended recovery: preserve this failed batch and its budget charge, stage
a new immutable retry, exclude ccc0388 for that retry, and verify actual CUDA
tensor allocation before proceeding with training. Keep the same seven-run,
1000-epoch protocol. No replacement jobs were submitted during this assessment.
