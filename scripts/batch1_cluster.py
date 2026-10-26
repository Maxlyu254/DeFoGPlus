#!/usr/bin/env python3
"""Stage immutable Git sources, inspect budget, or explicitly submit batch 1.

Run on Campus Cluster. `stage` and `status` never call sbatch. `submit` additionally
requires --submit and refuses a repeated/partial submission. No automatic retries.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]
PYTHON = "/scratch/xinyan14/motiflow-benchmarks/qm9-a100-20261026T041904Z/environment/bin/python"
DATA = "/scratch/xinyan14/motiflow-benchmarks/data/qm9"
PILOT_HOURS = 1859 / 3600
VARIANTS = ["B0", "B1", "A0", "A1", "M0", "M1"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(args):
    return subprocess.check_output(args, text=True).strip()


def stage(args):
    if command(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("Commit tracked changes before staging")
    root = Path(args.batch_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    commit = command(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    (root / "source").mkdir()
    archive = root / "source.tar"
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "-o", str(archive), commit], check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(root / "source")  # Trusted archive of this exact Git commit.
    archive.unlink()
    (root / "source" / "SOURCE_COMMIT").write_text(commit + "\n")
    for name in ["logs", "runs"]:
        (root / name).mkdir()
    settings = dict(BATCH1_ROOT=str(root), BATCH1_SOURCE=str(root / "source"),
                    BATCH1_PYTHON=args.python, BATCH1_DATA=args.data_dir,
                    BATCH1_CACHE=str(root.parent / "prior-cache"),
                    BATCH1_ARCHIVE=str(Path(args.archive_root).resolve() / root.name))
    (root / "environment.sh").write_text("\n".join(
        "export " + key + "=" + shlex.quote(value) for key, value in settings.items()) + "\n")
    plan = dict(schema=1, git_commit=commit, configurations=VARIANTS, seed=0,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                settings=settings, pilot_gpu_hours=PILOT_HOURS, gpu_cap_hours=50,
                gpu_array_max_hours=48, jobs={})
    write_json(root / "launch.json", plan)
    print(json.dumps(plan, indent=2))
    print("Staged only. No jobs submitted.")


def accounting(root, plan):
    records = []
    array_id = plan.get("jobs", {}).get("array")
    if array_id:
        raw = command(["sacct", "-X", "-n", "-P", "-j", array_id,
                       "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"])
        for line in raw.splitlines():
            fields = line.split("|")
            if len(fields) < 4 or not re.fullmatch(r"\d+(?:_\d+)?", fields[0]):
                continue
            match = re.search(r"(?:^|,)gres/gpu=(\d+)", fields[3])
            gpus = int(match.group(1)) if match else 0
            records.append(dict(job_id=fields[0], state=fields[1], seconds=int(fields[2]),
                                allocated_gpu_hours=int(fields[2]) * gpus / 3600))
    ledger = dict(pilot_gpu_hours=PILOT_HOURS, jobs=records,
                  consumed_gpu_hours=PILOT_HOURS + sum(row["allocated_gpu_hours"] for row in records),
                  initial_reservation_cap_gpu_hours=PILOT_HOURS + 48,
                  limit_gpu_hours=50,
                  note="No retry or continuation is submitted automatically. Reconcile all attempts before adding GPU jobs.")
    write_json(root / "budget-ledger.json", ledger)
    return ledger


def submit(args, root, plan):
    if plan["jobs"]:
        raise RuntimeError("This batch already has submitted jobs; inspect status, do not duplicate")
    if PILOT_HOURS + 48 > plan["gpu_cap_hours"]:
        raise RuntimeError("GPU reservation exceeds budget")
    scripts = root / "source" / "scripts" / "slurm"
    common = ["sbatch", "--parsable", "--export=ALL,BATCH1_ROOT=" + str(root),
              "--output=" + str(root / "logs" / "%x-%A_%a.out")]
    if not args.submit:
        print("Dry run: CPU prepare -> A100 array 0-5%" + str(args.concurrency) + " (8h each) -> CPU W&B sync.")
        print("Reservation including the previous pilot: %.3f / 50 GPU-hours." % (PILOT_HOURS + 48))
        print("Pass --submit to actually submit. No jobs submitted.")
        return
    if not Path(plan["settings"]["BATCH1_PYTHON"]).is_file():
        raise FileNotFoundError("Prepared Python environment is unavailable")
    if not (Path.home() / ".config/defog/wandb-env.sh").is_file():
        raise FileNotFoundError("Private W&B environment file is unavailable")
    def job(name, extra, script):
        job_id = command(common + extra + [str(scripts / script)]).split(";")[0]
        plan["jobs"][name] = job_id
        write_json(root / "launch.json", plan)  # Preserve partial submissions too.
        print(name + ": " + job_id)
        return job_id
    prepare = job("prepare", [], "batch1-prepare.sbatch")
    array = job("array", ["--dependency=afterok:" + prepare,
                         "--kill-on-invalid-dep=yes", "--array=0-5%" + str(args.concurrency)], "batch1-a100.sbatch")
    job("sync", ["--dependency=afterany:" + array, "--kill-on-invalid-dep=yes"], "batch1-sync.sbatch")
    accounting(root, plan)


def sync(root, plan):
    python = plan["settings"]["BATCH1_PYTHON"]
    for session in sorted((root / "runs").glob("*/wandb/offline-run-*")):
        subprocess.run([python, "-m", "wandb", "sync", str(session)], check=True)
    accounting(root, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage", "submit", "status", "sync"])
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--data-dir", default=DATA)
    parser.add_argument("--archive-root", default="/u/xinyan14/motiflow-artifacts/qm9-batch1")
    parser.add_argument("--concurrency", type=int, choices=range(1, 7), default=2)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.mode == "stage":
        stage(args)
        return
    root = Path(args.batch_root).resolve()
    plan = json.loads((root / "launch.json").read_text())
    if args.mode == "submit":
        submit(args, root, plan)
    elif args.mode == "sync":
        sync(root, plan)
    else:
        print(json.dumps(accounting(root, plan), indent=2))
        ids = ",".join(plan["jobs"].values())
        if ids:
            subprocess.run(["squeue", "-j", ids, "-o", "%.20i %.28j %.10T %.12M %R"], check=True)


if __name__ == "__main__":
    main()
