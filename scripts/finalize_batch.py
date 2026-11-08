#!/usr/bin/env python3
"""Save batch completion counts and label the one native Slurm notification."""
import argparse
import json
import os
from pathlib import Path

from batch1_cluster import command, write_json


def summary(root, plan, controller_state):
    completed, incomplete = [], []
    for task in plan['tasks']:
        name = f"{task['configuration']}-seed{task['seed']}"
        path = root/'runs'/name/'metrics/summary.json'
        result = json.loads(path.read_text()) if path.is_file() else {}
        if result.get('status') == 'completed' and result.get('completed_epochs') == plan['target_epochs']:
            completed.append(name)
        else:
            incomplete.append(name)
    return dict(batch_root=str(root), source_commit=plan['git_commit'],
                completed=completed, incomplete=incomplete, total=len(plan['tasks']),
                final_controller_state=controller_state,
                status='COMPLETE' if not incomplete and controller_state=='COMPLETED' else 'INCOMPLETE',
                budget=json.loads((root/'budget-ledger.json').read_text())
                       if (root/'budget-ledger.json').is_file() else None,
                retry_stopped=plan.get('retry_stopped',{}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-root',required=True,type=Path)
    args=p.parse_args()
    root=args.batch_root.resolve()
    plan=json.loads((root/'launch.json').read_text())
    controller=(plan.get('retry_waves') or [{}])[-1].get('sync') or plan['jobs']['sync']
    raw=command(['sacct','-X','-n','-P','-j',controller,'--format=JobID%40,State'])
    states=[f[1].strip().split()[0] for line in raw.splitlines()
            if len(f:=line.split('|'))>=2 and f[0].strip()==controller]
    result=summary(root,plan,states[0] if states else 'UNKNOWN')
    write_json(root/'batch-summary.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if os.environ.get('SLURM_JOB_ID'):
        # Slurm's standard email contains the job name. This encodes outcome
        # and completed-run count; full details stay in batch-summary.json.
        name=f"defog-{result['status']}-{len(result['completed'])}of{result['total']}"
        command(['scontrol','update','JobId='+os.environ['SLURM_JOB_ID'],'JobName='+name])


if __name__=='__main__':
    main()
