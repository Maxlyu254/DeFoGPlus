#!/usr/bin/env python3
"""Prepare and explicitly submit the authorized B1 seed1 continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from batch1_cluster import command, write_json
from gpu_retry import task_records

REPO=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def reconcile_train_log(run, backup, epoch):
    """Preserve original metrics, then remove post-checkpoint rows from active log."""
    shutil.copytree(run/'metrics',backup/'metrics')
    path=run/'metrics/train.jsonl'
    lines=path.read_text().splitlines(keepends=True)
    rows=[json.loads(line) for line in lines]
    assert [r['completed_epoch'] for r in rows]==list(range(1,len(rows)+1))
    kept=[line for line,row in zip(lines,rows) if row['completed_epoch']<=epoch]
    assert len(kept)==epoch
    before=digest(path)
    tmp=path.with_suffix('.reconcile.tmp')
    tmp.write_text(''.join(kept))
    tmp.replace(path)
    return dict(original_epochs=len(rows),active_epochs=len(kept),archived_tail_epochs=len(rows)-len(kept),
                original_sha256=before,active_sha256=digest(path))


def stage(root,batch):
    if command(['git','-C',str(REPO),'status','--porcelain']):
        raise RuntimeError('Commit source before staging')
    original=json.loads((batch/'launch.json').read_text())
    rec=task_records(original['jobs']['array'])
    if rec[0]['state']!='TIMEOUT' or any(r['state']!='COMPLETED' for i,r in rec.items() if i):
        raise RuntimeError('Unexpected batch state; reconcile before continuing')
    run=batch/'runs/B1-seed1'
    if (run/'metrics/summary.json').is_file():
        raise RuntimeError('Run already has a summary; inspect before continuing')
    index=json.loads((run/'checkpoint-index.json').read_text())
    checkpoint=run/'checkpoints'/index['latest']
    expected=index['sha256'][checkpoint.name]
    if digest(checkpoint)!=expected:
        raise RuntimeError('Recovery checksum mismatch')
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    epoch=state['batch1']['completed_epochs']
    manifest=json.loads((run/'manifest.json').read_text())
    assert epoch==825 and state['global_step']==825*96
    assert state['batch1']['signature']==manifest['model_signature']
    assert state['optimizer_states'] and state['batch1']['rng']
    del state
    root.mkdir(parents=True,exist_ok=False)
    for directory in ['logs','backup','source']:
        (root/directory).mkdir()
    commit=command(['git','-C',str(REPO),'rev-parse','HEAD'])
    archive=root/'source.tar'
    subprocess.run(['git','-C',str(REPO),'archive','-o',str(archive),commit],check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(root/'source')
    archive.unlink()
    shutil.copy2(checkpoint,root/'backup'/checkpoint.name)
    for name in ['manifest.json','checkpoint-index.json','resolved-config.yaml']:
        shutil.copy2(run/name,root/'backup'/name)
    reconciliation=reconcile_train_log(run,root/'backup',epoch)
    plan=dict(schema=1,batch_root=str(batch),run_root=str(run),operations_commit=commit,
              training_source=original['settings']['BATCH1_SOURCE'],training_commit=original['git_commit'],
              python=original['settings']['BATCH1_PYTHON'],checkpoint=str(root/'backup'/checkpoint.name),
              checkpoint_sha256=expected,resume_epoch=epoch,target_epochs=1000,
              configuration='B1',seed=1,wall_safeguard_hours=12,budget_limit_gpu_hours=None,
              excluded_nodes='ccc0388,ccc0390',mail_user='xinyan14@illinois.edu',
              existing_wandb_sessions=[p.name for p in (run/'wandb').glob('offline-run-*')],
              reconciliation=reconciliation,jobs={})
    write_json(root/'continuation.json',plan)
    print(json.dumps(plan,indent=2))


def verify(root,plan):
    if digest(plan['checkpoint'])!=plan['checkpoint_sha256']:
        raise RuntimeError('Continuation checkpoint checksum mismatch')
    if (Path(plan['run_root'])/'metrics/summary.json').is_file():
        raise RuntimeError('Run now has a summary; refusing duplicate continuation')


def submit(root,plan):
    if plan['jobs']:
        raise RuntimeError('Continuation already submitted; do not duplicate')
    for other in root.parent.glob('*/continuation.json'):
        old=json.loads(other.read_text())
        if other.parent!=root and old.get('run_root')==plan['run_root'] and old.get('jobs'):
            raise RuntimeError('Another continuation exists for this run')
    verify(root,plan)
    common=['sbatch','--parsable','--export=ALL,CONTINUATION_ROOT='+str(root),
            '--output='+str(root/'logs/%x-%j.out')]
    def job(role,args,script):
        jobid=command(common+args+[str(root/'source/scripts/slurm'/script)]).split(';')[0]
        if not jobid.isdigit():
            raise RuntimeError('Unknown submission response; inspect scheduler before retry')
        plan['jobs'][role]=jobid
        write_json(root/'continuation.json',plan)
        print(role+': '+jobid,flush=True)
        return jobid
    gpu=job('gpu',['--exclude='+plan['excluded_nodes'],'--mail-type=NONE'],'continue-b1.sbatch')
    job('sync_notify',['--dependency=afterany:'+gpu,'--mail-user='+plan['mail_user'],
                      '--mail-type=END,FAIL'],'continue-b1-sync.sbatch')


def sync(root,plan):
    run=Path(plan['run_root'])
    sessions=[p for p in (run/'wandb').glob('offline-run-*') if p.name not in plan['existing_wandb_sessions']]
    raw=command(['sacct','-X','-n','-P','-j',plan['jobs']['gpu'],
                 '--format=JobID,State,ElapsedRaw'])
    fields=next(line.split('|') for line in raw.splitlines() if line.split('|')[0].strip()==plan['jobs']['gpu'])
    result=dict(gpu_state=fields[1],allocated_gpu_hours=int(fields[2])/3600,budget_limit_gpu_hours=None)
    try:
        for session in sessions:
            subprocess.run([plan['python'],'-m','wandb','sync',str(session)],check=True)
        summary=json.loads((run/'metrics/summary.json').read_text()) if (run/'metrics/summary.json').is_file() else {}
        result.update(completed_epochs=summary.get('completed_epochs'),status=summary.get('status','incomplete'))
        if fields[1]!='COMPLETED' or summary.get('status')!='completed' or summary.get('completed_epochs')!=1000:
            raise RuntimeError('Continuation incomplete; see summary and Slurm output')
    except BaseException:
        result['status']='incomplete'
        raise
    finally:
        write_json(root/'summary.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['stage','verify','submit','sync'])
    p.add_argument('--continuation-root',required=True,type=Path)
    p.add_argument('--batch-root',type=Path)
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    root=args.continuation_root.resolve()
    if args.mode=='stage':
        if not args.batch_root:
            p.error('--batch-root required to stage')
        stage(root,args.batch_root.resolve())
    else:
        plan=json.loads((root/'continuation.json').read_text())
        if args.mode=='submit':
            if args.submit:
                submit(root,plan)
            else:
                print('Dry run: 12-hour GPU safeguard, then CPU sync and email. Pass --submit to submit.')
        elif args.mode=='verify':
            verify(root,plan)
        else:
            sync(root,plan)
