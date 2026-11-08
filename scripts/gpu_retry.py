"""Bounded startup-health retries, run by the afterany CPU sync job.

Caller must hold retry.lock while modifying launch.json. Never called by a GPU
task. Only explicit reset markers from this exact failed allocation are retried.
"""
import json
import re

from batch1_cluster import command, write_json

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def waves(plan):
    initial = plan.get('jobs', {}).get('array')
    return ([dict(array=initial, tasks=[t['array_task'] for t in plan['tasks']])]
            if initial else []) + plan.get('retry_waves', [])


def task_records(array, runner=command):
    raw = runner(['sacct', '-X', '-n', '-P', '-j', array,
                  '--format=JobID%40,State,ExitCode,ElapsedRaw,AllocTRES,NodeList%100'])
    result = {}
    for line in raw.splitlines():
        f = [s.strip() for s in line.split('|')]
        if len(f) < 6 or not re.fullmatch(re.escape(array)+r'_\d+', f[0]):
            continue
        gpu = re.search(r'(?:^|,)gres/gpu=(\d+)', f[4])
        if not gpu:
            raise RuntimeError('Missing GPU accounting: '+line)
        task = int(f[0].split('_')[1])
        if task in result:
            raise RuntimeError('Duplicate accounting row: '+f[0])
        result[task] = dict(job_id=f[0], state=f[1].split()[0].rstrip('+'),
                            exit_code=f[2], seconds=int(f[3]), node=f[5],
                            allocated_gpu_hours=int(f[3])*int(gpu.group(1))/3600)
    return result


def eligible(root, task, array, record):
    run = root/'runs'/f"{task['configuration']}-seed{task['seed']}"
    # A startup retry never restarts partially trained or completed models.
    if any((run/name).exists() for name in ['manifest.json','metrics/train.jsonl','metrics/summary.json']):
        return None
    if list(run.rglob('*.ckpt')):
        return None
    marker = run/'logs'/f"gpu-health-{array}_{task['array_task']}.json"
    if record['state'] != 'FAILED' or record['exit_code'] != '86:0' or not marker.is_file():
        return None
    data = json.loads(marker.read_text())
    if (data.get('reason') != 'gpu_reset_required' or data.get('phase') != 'cuda_precheck'
            or data.get('array_job_id') != array
            or data.get('array_task_id') != str(task['array_task'])
            or data.get('hostname') != record['node']):
        return None
    node = data['hostname']
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', node):
        raise RuntimeError('Invalid node in health marker')
    return node


def reconcile(root, plan, runner=command):
    """Return True if another wave is pending; otherwise allow final W&B sync."""
    if not plan.get('max_gpu_retries', 0):
        return False
    if plan.get('retry_intent'):
        raise RuntimeError('An interrupted retry submission needs reconciliation; refusing duplicate sbatch')
    history = waves(plan)
    if not history:
        return False
    records, consumed = [], plan.get('previous_attempt_charge_gpu_hours', 0)
    for wave in history:
        rec = task_records(wave['array'], runner)
        if set(rec) != set(wave['tasks']):
            raise RuntimeError('Accounting incomplete; no retry submitted')
        if any(r['state'] not in TERMINAL for r in rec.values()):
            # A duplicate controller must not submit while another wave is active.
            return True
        consumed += sum(r['allocated_gpu_hours'] for r in rec.values())
        records.append(rec)
    latest = {}
    attempts = {}
    for wave, rec in zip(history, records):
        for task in wave['tasks']:
            latest[task] = (wave['array'], rec[task])
            attempts[task] = attempts.get(task, 0)+1
    excluded = set(filter(None, plan.get('excluded_nodes', '').split(',')))
    retry, stopped = [], {}
    for task in plan['tasks']:
        index = task['array_task']
        array, record = latest[index]
        if record['state'] == 'COMPLETED':
            continue
        node = eligible(root, task, array, record)
        if node:
            excluded.add(node)
        if node and attempts[index] <= plan['max_gpu_retries']:
            retry.append(index)
        else:
            stopped[str(index)] = 'retry_limit' if node else 'not_a_reset_startup_failure'
    plan['retry_stopped'] = stopped
    plan['excluded_nodes'] = ','.join(sorted(excluded))
    plan['retry_accounted_gpu_hours'] = consumed
    # The full time limits of proposed replacements must fit the batch ceiling.
    if consumed + len(retry)*plan['gpu_task_hours'] > plan['budget_gpu_hours']:
        plan['retry_stopped'].update({str(i):'budget_limit' for i in retry})
        retry = []
    write_json(root/'launch.json', plan)
    if not retry:
        return False
    # Persist intent before sbatch; if response/persistence is interrupted, stop
    # for reconciliation rather than submitting an untracked duplicate.
    plan['retry_intent'] = dict(tasks=retry, excluded_nodes=plan['excluded_nodes'])
    write_json(root/'launch.json', plan)
    common = ['sbatch','--parsable','--export=ALL,BATCH1_ROOT='+str(root),
              '--output='+str(root/'logs/%x-%A_%a.out')]
    scripts = root/'source/scripts/slurm'
    seconds = int(plan['gpu_task_hours']*3600)
    array = runner(common+['--job-name=defog-gpu-retry', '--time='+str(seconds//60),
                   '--array='+','.join(map(str,retry)), '--exclude='+plan['excluded_nodes'],
                   str(scripts/'batch1-a100.sbatch')]).split(';')[0]
    if not array.isdigit():
        raise RuntimeError('Unrecognized sbatch response; reconcile retry_intent')
    wave = dict(array=array, tasks=retry, sync=None)
    plan.setdefault('retry_waves', []).append(wave)
    write_json(root/'launch.json', plan)
    sync = runner(common+['--job-name=defog-retry-sync','--dependency=afterany:'+array,
                  '--kill-on-invalid-dep=yes',str(scripts/'batch1-sync.sbatch')]).split(';')[0]
    if not sync.isdigit():
        raise RuntimeError('Unrecognized sync sbatch response; reconcile retry_intent')
    wave['sync'] = sync
    del plan['retry_intent']
    write_json(root/'launch.json', plan)
    print(json.dumps(dict(retry_array=array, tasks=retry, sync=sync,
                          excluded_nodes=plan['excluded_nodes']),indent=2), flush=True)
    return True
