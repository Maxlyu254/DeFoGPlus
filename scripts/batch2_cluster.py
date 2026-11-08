#!/usr/bin/env python3
"""Fixed-1000 batch 2: seven simultaneously eligible A100 tasks, fresh budget."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

from batch1_cluster import command, write_json, PYTHON, DATA
from gpu_retry import reconcile, waves

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("B0", 1), ("B0", 2), ("A0", 1), ("M0", 1), ("M0", 2), ("M1", 1), ("M1", 2)]


def stage(args):
    if command(["git", "-C", str(ROOT), "status", "--porcelain"]):
        raise RuntimeError("Stage only a clean committed checkout")
    root = Path(args.batch_root).resolve()
    retry_of = str(Path(args.retry_of).resolve()) if args.retry_of else None
    previous_charge = 0
    if retry_of:
        previous = Path(retry_of)
        previous_plan = json.loads((previous/"launch.json").read_text())
        previous_ledger = accounting(previous, previous_plan)
        if len(previous_ledger["jobs"]) != len(TASKS) or any(
                r["state"] != "FAILED" for r in previous_ledger["jobs"]):
            raise RuntimeError("Retry requires all seven previous tasks to be FAILED")
        if list((previous/"runs").glob("*/metrics/summary.json")):
            raise RuntimeError("Previous results exist; reconcile before retry")
        previous_charge = previous_ledger["consumed_gpu_hours"]
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
                previous_batch_charge_gpu_hours=0, retry_of=retry_of,
                excluded_nodes=args.exclude_nodes,
                previous_attempt_charge_gpu_hours=previous_charge, jobs={})
    plan['max_gpu_retries'] = args.max_gpu_retries
    plan['mail_user'] = args.mail_user
    write_json(root/"launch.json", plan)
    print(json.dumps(plan, indent=2))


def accounting(root, plan):
    records = []
    for wave in waves(plan):
        raw = command(["sacct", "-X", "-n", "-P", "-j", wave['array'],
                       "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"])
        for line in raw.splitlines():
            f = line.split("|")
            if len(f)<4 or not re.fullmatch(r"\d+(?:_\d+)?", f[0]):
                continue
            gpu = re.search(r"(?:^|,)gres/gpu=(\d+)", f[3])
            records.append(dict(job_id=f[0], state=f[1], seconds=int(f[2]),
                                allocated_gpu_hours=int(f[2])*(int(gpu.group(1)) if gpu else 0)/3600))
    attempt_hours = sum(r["allocated_gpu_hours"] for r in records)
    prior_hours = plan.get("previous_attempt_charge_gpu_hours", 0)
    ledger = dict(budget_gpu_hours=plan['budget_gpu_hours'], reservation_gpu_hours=plan['reservation_gpu_hours'],
                  consumed_gpu_hours=prior_hours+attempt_hours,
                  current_attempt_gpu_hours=attempt_hours,
                  previous_attempt_charge_gpu_hours=prior_hours,
                  previous_batch_charge_gpu_hours=0, jobs=records)
    write_json(root/"budget-ledger.json", ledger)
    return ledger


def submit(args, root, plan):
    if plan["jobs"]:
        raise RuntimeError("Already submitted (possibly partially); inspect before retry")
    for other in root.parent.glob("*/launch.json"):
        if other.parent != root and json.loads(other.read_text()).get("jobs", {}).get("array"):
            if str(other.parent.resolve()) == plan.get("retry_of"):
                previous_plan = json.loads(other.read_text())
                previous_ledger = accounting(other.parent, previous_plan)
                if len(previous_ledger["jobs"]) == len(TASKS) and all(
                        r["state"] == "FAILED" for r in previous_ledger["jobs"]):
                    if not list((other.parent/"runs").glob("*/metrics/summary.json")):
                        continue
            raise RuntimeError("Another batch-2 submission exists; reconcile before submitting")
    assert plan["reservation_gpu_hours"] + plan.get("previous_attempt_charge_gpu_hours", 0) <= plan["budget_gpu_hours"]
    if not args.submit:
        print("Dry run: CPU checks -> seven A100 tasks (0-6, no throttle, 7h/task) -> W&B sync. No submission.")
        return
    if not Path(PYTHON).is_file() or not (Path.home()/".config/defog/wandb-env.sh").is_file():
        raise RuntimeError("Existing environment or private W&B file unavailable")
    common = ["sbatch", "--parsable", "--mail-type=NONE", "--export=ALL,BATCH1_ROOT="+str(root),
              "--output="+str(root/"logs/%x-%A_%a.out")]
    def job(key, extra, script):
        jobid = command(common+extra+[str(root/"source/scripts/slurm"/script)]).split(";")[0]
        plan["jobs"][key] = jobid
        write_json(root/"launch.json", plan)
        print(key+": "+jobid, flush=True)
        return jobid
    cpu = job("prepare", ["--job-name=defog-batch2-prepare"], "batch1-prepare.sbatch")
    gpu_options = ["--job-name=defog-batch2", "--time=07:00:00", "--array=0-6",
                   "--dependency=afterok:"+cpu, "--kill-on-invalid-dep=yes"]
    if plan.get("excluded_nodes"):
        gpu_options.append("--exclude="+plan["excluded_nodes"])
    gpu = job("array", gpu_options, "batch1-a100.sbatch")
    sync_options = ["--job-name=defog-batch2-sync", "--dependency=afterany:"+gpu,
                    "--kill-on-invalid-dep=yes"]
    if plan.get('mail_user'):
        sync_options.append('--hold')
    sync = job("sync", sync_options, "batch1-sync.sbatch")
    if plan.get('mail_user'):
        notify = job('notify', ['--job-name=defog-batch-finished', '--hold',
                     '--dependency=afterany:'+sync, '--mail-type=END,FAIL',
                     '--mail-user='+plan['mail_user']], 'batch-notify.sbatch')
        command(['scontrol','release',sync])
        command(['scontrol','release',notify])
    accounting(root, plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["stage", "submit", "status", "sync"])
    p.add_argument("--batch-root", required=True)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--retry-of", help="Stage a replacement for a fully failed batch root")
    p.add_argument("--exclude-nodes", default="ccc0388,ccc0390", help="Slurm node list to exclude from GPU tasks")
    p.add_argument("--max-gpu-retries", type=int, default=2,
                   help="Automatic reset-failure retries per task in a newly staged batch; 0 disables")
    p.add_argument('--mail-user', default='xinyan14@illinois.edu',
                   help='Final batch notification recipient; empty string disables')
    args = p.parse_args()
    if args.max_gpu_retries < 0 or args.max_gpu_retries > 2:
        p.error('GPU retry count must be between 0 and 2')
    if args.mode == "stage":
        stage(args)
        return
    root = Path(args.batch_root).resolve()
    plan = json.loads((root/"launch.json").read_text())
    if args.mode == "submit":
        submit(args, root, plan)
    elif args.mode == "sync":
        # The CPU afterany job is the sole controller. Lock and reread prevent
        # duplicate submissions from concurrent/manual controller invocations.
        with (root/'retry.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            plan = json.loads((root/'launch.json').read_text())
            pending = reconcile(root, plan)
            print(json.dumps(accounting(root, plan), indent=2))
            if pending:
                return
            for session in sorted((root/"runs").glob("*/wandb/offline-run-*")):
                subprocess.run([PYTHON,"-m","wandb","sync",str(session)], check=True)
            if plan.get('retry_stopped'):
                raise RuntimeError('Batch has unresolved failures; see retry_stopped in launch.json')
    else:
        print(json.dumps(accounting(root, plan), indent=2))
        if plan["jobs"]:
            ids = list(plan['jobs'].values())
            for wave in plan.get('retry_waves', []):
                ids.extend([wave['array'],wave['sync']])
            subprocess.run(["squeue","-j",",".join(i for i in ids if i),
                            "-o","%.20i %.28j %.10T %.12M %R"], check=True)


if __name__ == "__main__":
    main()
