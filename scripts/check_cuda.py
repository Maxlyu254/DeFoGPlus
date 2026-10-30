"""Check allocated GPU identity and actual CUDA execution before model setup."""
import json
import os
import socket

import torch

print(json.dumps({
    "hostname": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
}), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("Allocated GPU failed CUDA availability check; inspect saved nvidia-smi logs")
name = torch.cuda.get_device_name(0)
if "A100" not in name:
    raise RuntimeError("Expected A100, received "+name)
value = (torch.ones(16, device="cuda") * 2).sum()
torch.cuda.synchronize()
if value.item() != 32:
    raise RuntimeError("CUDA tensor check returned an incorrect result")
print(json.dumps({"cuda_check": "passed", "device": name}), flush=True)
