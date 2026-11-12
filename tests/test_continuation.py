"""Continuation bookkeeping tests; no scheduler or GPU calls."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import continue_b1 as c


class ContinuationTest(unittest.TestCase):
    def test_archives_uncheckpointed_rows_and_keeps_unique_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';backup=root/'backup'
            (run/'metrics').mkdir(parents=True);backup.mkdir()
            path=run/'metrics/train.jsonl'
            original=''.join(json.dumps(dict(completed_epoch=i))+'\n' for i in range(1,5))
            path.write_text(original)
            result=c.reconcile_train_log(run,backup,2)
            self.assertEqual(result['original_epochs'],4)
            self.assertEqual(result['archived_tail_epochs'],2)
            self.assertEqual((backup/'metrics/train.jsonl').read_text(),original)
            self.assertEqual([json.loads(s)['completed_epoch'] for s in path.read_text().splitlines()],[1,2])

    def test_bad_history_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';backup=root/'backup'
            (run/'metrics').mkdir(parents=True);backup.mkdir()
            path=run/'metrics/train.jsonl'
            original='{"completed_epoch":1}\n{"completed_epoch":1}\n'
            path.write_text(original)
            with self.assertRaises(AssertionError):
                c.reconcile_train_log(run,backup,1)
            self.assertEqual(path.read_text(),original)

    def test_explicit_submit_and_job_ids_are_recorded_without_budget_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);calls=[]
            plan=dict(jobs={},run_root=str(root/'run'),excluded_nodes='ccc0388,ccc0390',
                      mail_user='xinyan14@illinois.edu',budget_limit_gpu_hours=None)
            def scheduler(args):
                calls.append(args)
                return str(100+len(calls))
            with patch.object(c,'verify'),patch.object(c,'command',scheduler):
                c.submit(root,plan)
            self.assertEqual(plan['jobs'],{'gpu':'101','sync_notify':'102'})
            self.assertIn('--exclude=ccc0388,ccc0390',calls[0])
            self.assertIn('--dependency=afterany:101',calls[1])
            self.assertIn('--mail-user=xinyan14@illinois.edu',calls[1])
            with self.assertRaises(RuntimeError):
                c.submit(root,plan)

    def test_modified_checkpoint_refuses_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'checkpoint';path.write_bytes(b'original')
            plan=dict(checkpoint=str(path),checkpoint_sha256=c.digest(path),run_root=str(root/'run'))
            c.verify(root,plan)
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                c.verify(root,plan)


if __name__=='__main__':
    unittest.main()
