#!/usr/bin/env python3
"""Sync with a separate SDK and verify every saved scalar history row remotely.

The training interpreter only decodes the old offline format. Uploads and cloud
reads use the pinned sidecar SDK; a CLI success or .synced marker is insufficient.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SYNC_VERSION = '0.31.2'
SYNC_PYTHON = '/scratch/xinyan14/motiflow-tools/wandb-sync/'+SYNC_VERSION+'/bin/python'
READER_PYTHON = '/scratch/xinyan14/motiflow-benchmarks/qm9-a100-20261026T041904Z/environment/bin/python'


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def inspect_session(session):
    # These are local-record imports only. This interpreter never contacts W&B.
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb
    files = list(Path(session).glob('*.wandb'))
    if len(files) != 1:
        raise ValueError('Expected one offline record file: '+str(session))
    path = files[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    store = DataStore()
    store.open_for_scan(str(path))
    identity, rows, has_exit, truncated = None, [], False, False
    try:
        while True:
            try:
                data = store.scan_data()
            except AssertionError:
                if not store.in_last_block():
                    raise
                truncated = True
                break
            if data is None:
                break
            record = pb.Record()
            record.ParseFromString(data)
            kind = record.WhichOneof('record_type')
            if kind == 'run':
                current = dict(entity=record.run.entity, project=record.run.project,
                               run_id=record.run.run_id)
                if identity is not None and identity != current:
                    raise ValueError('Multiple run identities in one session')
                identity = current
            elif kind == 'exit':
                has_exit = True
            elif kind == 'history':
                values = {}
                for item in record.history.item:
                    key = item.key or '.'.join(item.nested_key)
                    value = json.loads(item.value_json)
                    if key == '_step' or (not key.startswith('_') and
                            isinstance(value, (int, float, bool, str)) and
                            not (isinstance(value, float) and not math.isfinite(value))):
                        values[key] = value
                if '_step' not in values:
                    raise ValueError('History row missing its original step')
                rows.append(values)
    finally:
        store.close()
    if not identity or not rows or not all(identity.values()):
        raise ValueError('Missing run identity or scalar history: '+str(session))
    return dict(**identity, session=str(Path(session).resolve()), file_sha256=digest,
                history=rows, has_exit=has_exit, truncated_last_block=truncated)


def same_scalar(expected, actual):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-10)
    return type(expected) is type(actual) and expected == actual


def compare_history(expected, actual):
    cloud = {int(row['_step']): row for row in actual}
    matched = values = 0
    for row in expected:
        step = int(row['_step'])
        if step not in cloud:
            raise ValueError('Cloud history missing step '+str(step))
        for key, value in row.items():
            if key not in cloud[step] or not same_scalar(value, cloud[step][key]):
                raise ValueError('Cloud history mismatch at step '+str(step)+', key '+key)
            values += 1
        matched += 1
    counts = {namespace: sum(any(k.startswith(namespace+'/') for k in row)
                             for row in expected)
              for namespace in ['train','validation','curve','final']}
    train_epochs = [row['train/completed_epoch'] for row in expected if 'train/completed_epoch' in row]
    return dict(matched_history_rows=matched, matched_scalar_values=values,
                cloud_history_rows=len(actual), namespace_rows=counts,
                last_training_epoch=max(train_epochs) if train_epochs else None)


def project_info(api, spec):
    query = '''query($entity:String!,$project:String!){
        project(entityName:$entity,name:$project){name access}}'''
    return api._service_api.execute_graphql(query, dict(entity=spec['entity'],
                                                       project=spec['project']))['project']


def ensure_project(api, spec):
    existing = project_info(api, spec)
    if existing is None:
        query = '''mutation($input:UpsertModelInput!){upsertModel(input:$input){model{name access}}}'''
        api._service_api.execute_graphql(query, dict(input=dict(
            entityName=spec['entity'], name=spec['project'], access='PRIVATE')))
        existing = project_info(api, spec)
        if not existing or existing['access'] != 'PRIVATE':
            raise RuntimeError('New synchronization project was not created privately')
    return existing


def verify_cloud(spec):
    import wandb
    last_error = None
    path = '/'.join(spec[k] for k in ['entity','project','run_id'])
    for attempt in range(7):
        try:
            # Refresh run metadata and bypass stale history caches during ingestion.
            api = wandb.Api(timeout=30)
            run = api.run(path)
            rows = list(run.scan_history(page_size=1000, use_cache=False))
            result = compare_history(spec['history'], rows)
            return dict(**result, url=run.url, state=run.state,
                        project_access=project_info(api, spec)['access'])
        except (ValueError, RuntimeError) as error:
            last_error = error
            if attempt < 6:
                time.sleep(5)
    raise RuntimeError('Cloud verification failed for '+path+': '+str(last_error))


def worker(session, reader_python):
    import wandb
    if wandb.__version__ != SYNC_VERSION:
        raise RuntimeError('Expected pinned synchronization SDK '+SYNC_VERSION)
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY missing; explicitly source the private wandb-env.sh')
    session = Path(session).resolve()
    evidence = session/'verified-sync.json'
    with (session/'verified-sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = subprocess.check_output([reader_python, str(Path(__file__).resolve()),
                                       'inspect', str(session)], text=True)
        spec = json.loads(raw)
        metadata = {k:v for k,v in spec.items() if k != 'history'}
        metadata.update(sdk_version=wandb.__version__,
                        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            # Never trust a marker from the legacy CLI. A prior receipt is also
            # rechecked remotely, so deleted/missing cloud data triggers recovery.
            if evidence.exists() and json.loads(evidence.read_text()).get('file_sha256') == spec['file_sha256']:
                try:
                    verified = verify_cloud(spec)
                    result = dict(metadata, status='verified', **verified)
                    atomic_json(evidence, result)
                    print(json.dumps(result), flush=True)
                    return
                except RuntimeError:
                    pass
            ensure_project(wandb.Api(timeout=30), spec)
            env = dict(os.environ, WANDB_MODE='online')
            subprocess.run([sys.executable, '-m', 'wandb', 'sync', '--yes', '--include-synced',
                            '--entity', spec['entity'], '--project', spec['project'],
                            str(session)], env=env, check=True, timeout=1800)
            result = dict(metadata, status='verified', **verify_cloud(spec))
            atomic_json(evidence, result)
            print(json.dumps(result), flush=True)
        except BaseException as error:
            # Retain previous success receipts, but save this failed check separately.
            message = str(error).replace(os.environ['WANDB_API_KEY'], '[REDACTED]')
            atomic_json(session/'verified-sync-failure.json', dict(metadata, status='failed', error=message))
            raise


def sync_session(session, reader_python=None):
    python = os.environ.get('BATCH1_SYNC_PYTHON', SYNC_PYTHON)
    if not Path(python).is_file():
        raise FileNotFoundError('Install the isolated W&B sync environment: '+python)
    subprocess.run([python, str(Path(__file__).resolve()), 'sync', str(session),
                    '--reader-python', reader_python or sys.executable], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect','sync','recover'])
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--reader-python', default=READER_PYTHON)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.mode == 'inspect':
        if len(args.paths) != 1:
            parser.error('inspect requires exactly one offline session')
        print(json.dumps(inspect_session(args.paths[0]), allow_nan=False))
    elif args.mode == 'sync':
        if len(args.paths) != 1:
            parser.error('sync requires exactly one offline session')
        worker(args.paths[0], args.reader_python)
    else:
        if not args.report:
            parser.error('recover requires --report')
        sessions = sorted({p.resolve() for root in args.paths for p in root.glob('*/wandb/offline-run-*')})
        if not sessions:
            raise RuntimeError('No offline sessions found; refusing an empty success')
        report = dict(status='running', sdk_version=SYNC_VERSION, sessions=[], failures=[])
        atomic_json(args.report, report)
        for session in sessions:
            try:
                sync_session(session, args.reader_python)
                report['sessions'].append(json.loads((session/'verified-sync.json').read_text()))
            except Exception as error:
                report['failures'].append(dict(session=str(session), error=str(error)))
            atomic_json(args.report, report)
        report['status'] = 'verified' if not report['failures'] else 'incomplete'
        atomic_json(args.report, report)
        if report['failures']:
            raise SystemExit(1)
