#!/usr/bin/env python3
"""Fixed-1000 batch 2: seven simultaneously eligible A100 tasks, fresh budget."""
import argparse
import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from batch1_cluster import command, write_json, PYTHON, DATA

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("B0", 1), ("B0", 2), ("A0", 1), ("M0", 1), ("M0", 2), ("M1", 1), ("M1", 2)]


def stage(args):
    if command(["git", "-C", str(ROOT), "status", "--porcelain"]):
        raise RuntimeError("Stage only a clean committed checkout")
    root = Path(args.batch_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    commit = command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    source = root/"source"
    source.mkdir()
    archive = root/"source.tar"
    subprocess.run(["git", "-C", str(ROOT), "archive", "-o", str(archive), commit], check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(source)
    archive.unlink()
    (source/"SOURCE_COMMIT").write_text(commit+"\n")
    for name in ["logs", "runs"]:
        (root/name).mkdir()
    (root/"tasks.tsv").write_text("".join(f"{variant}\t{seed}\n" for variant, seed in TASKS))
    settings = dict(BATCH1_ROOT=str(root), BATCH1_SOURCE=str(source), BATCH1_PYTHON=PYTHON,
                    BATCH1_DATA=DATA,
                    BATCH1_CACHE="/scratch/xinyan14/motiflow-experiments/qm9-batch1/prior-cache",
                    BATCH1_ARCHIVE="/u/xinyan14/motiflow-artifacts/qm9-batch2/"+root.name,
                    BATCH1_PROTOCOL="batch2", BATCH1_TASK_FILE=str(root/"tasks.tsv"),
                    BATCH1_MANAGER=str(source/"scripts/batch2_cluster.py"))
    (root/"environment.sh").write_text("".join("export "+k+"="+shlex.quote(v)+"\n" for k,v in settings.items()))
    plan = dict(schema=1, protocol="batch2", git_commit=commit, settings=settings,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                tasks=[dict(array_task=i, configuration=v, seed=s) for i,(v,s) in enumerate(TASKS)],
                target_epochs=1000, array_expression="0-6", concurrency_cap=None,
                gpu_task_hours=7, reservation_gpu_hours=49, budget_gpu_hours=50,
                previous_batch_charge_gpu_hours=0, jobs={})
    write_json(root/"launch.json", plan)
    print(json.dumps(plan, indent=2))


def accounting(root, plan):
    records = []
    if plan["jobs"].get("array"):
        raw = command(["sacct", "-X", "-n", "-P", "-j", plan["jobs"]["array"],
                       "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"])
        for line in raw.splitlines():
            f = line.split("|")
            if len(f)<4 or not re.fullmatch(r"\d+(?:_\d+)?", f[0]):
                continue
            gpu = re.search(r"(?:^|,)gres/gpu=(\d+)", f[3])
            records.append(dict(job_id=f[0], state=f[1], seconds=int(f[2]),
                                allocated_gpu_hours=int(f[2])*(int(gpu.group(1)) if gpu else 0)/3600))
    ledger = dict(budget_gpu_hours=50, reservation_gpu_hours=49,
                  consumed_gpu_hours=sum(r["allocated_gpu_hours"] for r in records),
                  previous_batch_charge_gpu_hours=0, jobs=records)
    write_json(root/"budget-ledger.json", ledger)
    return ledger


def submit(args, root, plan):
    if plan["jobs"]:
        raise RuntimeError("Already submitted (possibly partially); inspect before retry")
    for other in root.parent.glob("*/launch.json"):
        if other.parent != root and json.loads(other.read_text()).get("jobs", {}).get("array"):
            raise RuntimeError("Another batch-2 submission exists; reconcile before submitting")
    assert plan["reservation_gpu_hours"] <= plan["budget_gpu_hours"]
    if not args.submit:
        print("Dry run: CPU checks -> seven A100 tasks (0-6, no throttle, 7h/task) -> W&B sync. No submission.")
        return
    if not Path(PYTHON).is_file() or not (Path.home()/".config/defog/wandb-env.sh").is_file():
        raise RuntimeError("Existing environment or private W&B file unavailable")
    common = ["sbatch", "--parsable", "--export=ALL,BATCH1_ROOT="+str(root),
              "--output="+str(root/"logs/%x-%A_%a.out")]
    def job(key, extra, script):
        jobid = command(common+extra+[str(root/"source/scripts/slurm"/script)]).split(";")[0]
        plan["jobs"][key] = jobid
        write_json(root/"launch.json", plan)
        print(key+": "+jobid, flush=True)
        return jobid
    cpu = job("prepare", ["--job-name=defog-batch2-prepare"], "batch1-prepare.sbatch")
    gpu = job("array", ["--job-name=defog-batch2", "--time=07:00:00", "--array=0-6",
                         "--dependency=afterok:"+cpu, "--kill-on-invalid-dep=yes"], "batch1-a100.sbatch")
    job("sync", ["--job-name=defog-batch2-sync", "--dependency=afterany:"+gpu,
                 "--kill-on-invalid-dep=yes"], "batch1-sync.sbatch")
    accounting(root, plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["stage", "submit", "status", "sync"])
    p.add_argument("--batch-root", required=True)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    if args.mode == "stage":
        stage(args)
        return
    root = Path(args.batch_root).resolve()
    plan = json.loads((root/"launch.json").read_text())
    if args.mode == "submit":
        submit(args, root, plan)
    elif args.mode == "sync":
        for session in sorted((root/"runs").glob("*/wandb/offline-run-*")):
            subprocess.run([PYTHON,"-m","wandb","sync",str(session)], check=True)
        print(json.dumps(accounting(root, plan), indent=2))
    else:
        print(json.dumps(accounting(root, plan), indent=2))
        if plan["jobs"]:
            subprocess.run(["squeue","-j",",".join(plan["jobs"].values()),
                            "-o","%.20i %.28j %.10T %.12M %R"], check=True)


if __name__ == "__main__":
    main()
