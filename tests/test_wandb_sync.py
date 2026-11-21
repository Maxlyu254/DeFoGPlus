"""A zero CLI status or sync marker must never substitute for cloud history."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wandb_sync import compare_history, sync_session, ensure_project


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


if __name__=='__main__': unittest.main()
