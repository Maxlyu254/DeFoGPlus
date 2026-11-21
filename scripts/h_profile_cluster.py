#!/usr/bin/env python3
"""Stage/submit only the authorized B0/M1 explicit-H profiling allocation."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

from batch1_cluster import command,write_json,PYTHON,DATA
from batch2_cluster import submit,accounting
from gpu_retry import reconcile
from wandb_sync import sync_session

ROOT=Path(__file__).resolve().parents[1]


def stage(root):
    if command(['git','-C',str(ROOT),'status','--porcelain']):
        raise RuntimeError('Commit before staging')
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source';source.mkdir()
    commit=command(['git','-C',str(ROOT),'rev-parse','HEAD'])
    archive=root/'source.tar'
    subprocess.run(['git','-C',str(ROOT),'archive','-o',str(archive),commit],check=True)
    with tarfile.open(archive) as h:
        h.extractall(source)
    archive.unlink();(source/'SOURCE_COMMIT').write_text(commit+'\n')
    for n in ['logs','runs','profiles']:
        (root/n).mkdir()
    settings=dict(BATCH1_ROOT=str(root),BATCH1_SOURCE=str(source),BATCH1_PYTHON=PYTHON,
        BATCH1_DATA=DATA,BATCH1_CACHE=str(root.parent/'prior-cache'),
        BATCH1_ARCHIVE='/u/xinyan14/motiflow-artifacts/qm9-h-profiling/'+root.name,
        BATCH1_PROTOCOL='h_profile',BATCH1_MANAGER=str(source/'scripts/h_profile_cluster.py'))
    (root/'environment.sh').write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in settings.items()))
    plan=dict(schema=1,protocol='h_profile',git_commit=commit,settings=settings,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tasks=[dict(array_task=0,configuration='Hprofile',seed=0)],array_expression='0',
        profiling_variants=['B0','M1'],target_epochs=8,concurrency_cap=None,
        gpu_task_hours=4,reservation_gpu_hours=4,budget_gpu_hours=50,
        previous_attempt_charge_gpu_hours=0,previous_batch_charge_gpu_hours=0,
        excluded_nodes='ccc0388,ccc0390',max_gpu_retries=2,mail_user='xinyan14@illinois.edu',
        prepare_script='qm9-h-prepare.sbatch',gpu_script='qm9-h-profile.sbatch',jobs={})
    write_json(root/'launch.json',plan)
    print(json.dumps(plan,indent=2))


def sync(root):
    with (root/'retry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan=json.loads((root/'launch.json').read_text())
        pending=reconcile(root,plan)
        ledger=accounting(root,plan)
        if pending:
            return
        for session in (root/'profiles').glob('*/wandb/offline-run-*'):
            sync_session(session, PYTHON)
        summaries={}
        for variant in ['B0','M1']:
            path=root/'profiles'/variant/'summary.json'
            if path.is_file():
                summaries[variant]=json.loads(path.read_text())
        ok=len(summaries)==2 and all(s.get('status')=='completed' for s in summaries.values()) and not plan.get('retry_stopped')
        write_json(root/'summary.json',dict(status='completed' if ok else 'incomplete',profiles=summaries,budget=ledger))
        # One logical profiling task encompasses both sequential model profiles.
        (root/'runs/Hprofile-seed0/metrics').mkdir(parents=True,exist_ok=True)
        write_json(root/'runs/Hprofile-seed0/metrics/summary.json',
                   dict(status='completed' if ok else 'incomplete',completed_epochs=8 if ok else 0,
                        note='Profiling task: each B0/M1 has seven timed/warm-up epochs plus one resume-check epoch.'))
        archive=Path(plan['settings']['BATCH1_ARCHIVE'])
        archive.mkdir(parents=True,exist_ok=True)
        for filename in ['data-audit.json','selected-batches.json','summary.json','budget-ledger.json','launch.json']:
            if (root/filename).exists(): shutil.copy2(root/filename,archive/filename)
        for variant in summaries:
            src=root/'profiles'/variant
            dest=archive/'profiles'/variant
            dest.mkdir(parents=True,exist_ok=True)
            for filename in ['summary.json','manifest.json','resolved-config.yaml','resume-check.ckpt']:
                if (src/filename).exists(): shutil.copy2(src/filename,dest/filename)
            if (src/'metrics').exists(): shutil.copytree(src/'metrics',dest/'metrics',dirs_exist_ok=True)
        if not ok:
            raise RuntimeError('Profiling incomplete; see logs and summary.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['stage','submit','sync','status'])
    p.add_argument('--batch-root',required=True,type=Path)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args();root=args.batch_root.resolve()
    if args.mode=='stage':
        stage(root)
    elif args.mode=='sync':
        sync(root)
    else:
        plan=json.loads((root/'launch.json').read_text())
        if args.mode=='submit':
            submit(args,root,plan)
        else:
            print(json.dumps(accounting(root,plan),indent=2))
            subprocess.run(['squeue','-j',','.join(plan['jobs'].values())],check=True)
