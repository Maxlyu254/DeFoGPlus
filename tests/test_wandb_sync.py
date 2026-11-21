"""A zero CLI status or sync marker must never substitute for cloud history."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wandb_sync import compare_history, sync_session, ensure_project, worker


class VerifiedSyncTest(unittest.TestCase):
    def setUp(self):
        self.expected=[{'_step':0,'completed_epoch':1,'train/completed_epoch':1,'train/node_ce':.5},
                       {'_step':1,'completed_epoch':1,'curve/validity':.9,'curve/novelty':.3}]

    def test_all_values_and_curve_rows_checked(self):
        result=compare_history(self.expected,[dict(r,_runtime=2) for r in self.expected])
        self.assertEqual(result['matched_history_rows'],2)
        self.assertEqual(result['namespace_rows']['curve'],1)
        self.assertEqual(result['last_training_epoch'],1)

    def test_missing_cloud_run_history_fails(self):
        with self.assertRaisesRegex(ValueError,'row count'):
            compare_history(self.expected,[])

    def test_duplicate_replay_rows_rejected(self):
        duplicate=[dict(row,_step=row['_step']+2) for row in self.expected]
        with self.assertRaisesRegex(ValueError,'row count'):
            compare_history(self.expected,self.expected+duplicate)

    def test_missing_metric_and_wrong_value_fail(self):
        for rows in [[self.expected[0],{'_step':1}],
                     [self.expected[0],dict(self.expected[1],**{'curve/novelty':.8})]]:
            with self.assertRaisesRegex(ValueError,'mismatch'):
                compare_history(self.expected,rows)

    def test_small_roundoff_allowed_but_boolean_not_numeric(self):
        compare_history([{'_step':0,'x':.3}],[{'_step':0,'x':.30000000000001}])
        with self.assertRaises(ValueError):
            compare_history([{'_step':0,'x':True}],[{'_step':0,'x':1}])

    def test_new_project_private(self):
        calls=[]
        class Service:
            def execute_graphql(self,query,variables):
                calls.append((query,variables))
                return {'project':None if len(calls)==1 else {'name':'p','access':'PRIVATE'}}
        api=type('Api',(),{'_service_api':Service()})()
        ensure_project(api,{'entity':'e','project':'p'})
        self.assertEqual(calls[1][1]['input']['access'],'PRIVATE')

    def test_existing_project_visibility_preserved(self):
        calls=[]
        class Service:
            def execute_graphql(self,query,variables):
                calls.append(query)
                return {'project':{'name':'p','access':'PRIVATE'}}
        ensure_project(type('Api',(),{'_service_api':Service()})(),{'entity':'e','project':'p'})
        self.assertEqual(len(calls),1)

    def test_delegate_uses_sidecar_and_explicit_reader(self):
        with patch('wandb_sync.Path.is_file',return_value=True),patch('wandb_sync.subprocess.run') as run:
            sync_session('/tmp/session','/training/python')
        args=run.call_args[0][0]
        self.assertIn('/wandb-sync/0.31.2/bin/python',args[0])
        self.assertEqual(args[-2:],['--reader-python','/training/python'])
        self.assertTrue(run.call_args[1]['check'])

    def worker_case(self, existing, verification):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        session=Path(temp.name)
        spec=dict(entity='e',project='p',run_id='r',file_sha256='hash',history=self.expected)
        api=types.SimpleNamespace(runs=lambda *a,**k:iter([object()] if existing else []))
        fake=types.SimpleNamespace(__version__='0.31.2',Api=lambda **k:api)
        patches=[patch.dict(sys.modules,wandb=fake),
                 patch.dict('os.environ',WANDB_API_KEY='test-only'),
                 patch('wandb_sync.subprocess.check_output',return_value=json.dumps(spec)),
                 patch('wandb_sync.ensure_project'),
                 patch('wandb_sync.verify_cloud',side_effect=verification),
                 patch('wandb_sync.subprocess.run')]
        import contextlib
        with contextlib.ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            run=patches[-1].target.run
            if isinstance(verification,Exception):
                with self.assertRaises(RuntimeError): worker(session,'/reader')
                self.assertFalse((session/'verified-sync.json').exists())
                self.assertTrue((session/'verified-sync-failure.json').exists())
            else:
                worker(session,'/reader')
                self.assertEqual(json.loads((session/'verified-sync.json').read_text())['status'],'verified')
            return run.call_count

    def test_existing_cloud_data_not_replayed_without_receipt(self):
        self.assertEqual(self.worker_case(True,lambda spec:dict(matched_history_rows=2)),0)

    def test_zero_cli_exit_with_missing_cloud_data_is_failure(self):
        self.assertEqual(self.worker_case(False,RuntimeError('missing remote history')),1)

    def test_partial_existing_cloud_run_is_not_blindly_appended(self):
        self.assertEqual(self.worker_case(True,RuntimeError('incomplete remote history')),0)


if __name__=='__main__': unittest.main()
