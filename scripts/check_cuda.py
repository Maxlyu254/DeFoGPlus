"""Check allocated GPU identity and actual CUDA execution before model setup."""
import json
import os
from pathlib import Path
import re
import socket
import sys

import torch

print(json.dumps({
    "hostname": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
}), flush=True)
try:
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated GPU failed CUDA availability check")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        raise ValueError("Expected A100, received "+name)
    value = (torch.ones(16, device="cuda") * 2).sum()
    torch.cuda.synchronize()
    if value.item() != 32:
        raise ValueError("CUDA tensor check returned an incorrect result")
except RuntimeError as error:
    # Only explicit reset evidence plus failed CUDA initialization is eligible.
    # Missing libraries, incorrect GPU type, OOMs or training errors are not.
    health_path = os.environ.get("BATCH1_GPU_HEALTH_LOG")
    health = Path(health_path).read_text() if health_path and Path(health_path).is_file() else ''
    reset = bool(re.search(r'GPU Recovery Action\s*:\s*Reset\b|GPU requires reset', health, re.I))
    marker = os.environ.get('BATCH1_GPU_FAILURE_MARKER')
    if marker and reset:
        path = Path(marker)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(
            reason='gpu_reset_required', phase='cuda_precheck',
            hostname=socket.gethostname().split('.')[0],
            array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
            array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            error=str(error)))+'\n')
        temporary.replace(path)
        print('GPU reset required; recorded a retry-eligible startup failure', flush=True)
        sys.exit(86)
    raise
print(json.dumps({"cuda_check": "passed", "device": name}), flush=True)
