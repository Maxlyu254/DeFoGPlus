"""Controller tests with simulated scheduler responses; never submit real jobs."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import tarfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import gpu_retry
import batch2_cluster
from batch3_cluster import TASKS as BATCH3_TASKS
from finalize_batch import summary as batch_summary


class RetryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(jobs={'array':'100'}, tasks=[
            dict(array_task=i, configuration=v, seed=1) for i,v in enumerate(['B0','M0','M1'])],
            max_gpu_retries=2, excluded_nodes='ccc0388,ccc0390',
            gpu_task_hours=7, budget_gpu_hours=50, reservation_gpu_hours=21)
        self.rows = {'100': ['100_0|COMPLETED|0:0|21600|gres/gpu=1|ccc0389',
                             '100_1|FAILED|86:0|20|gres/gpu=1|ccc0391',
                             '100_2|FAILED|1:0|200|gres/gpu=1|ccc0387']}
        self.calls = []
        self.marker('100',1,'ccc0391')

    def marker(self, array, index, node):
        task=self.plan['tasks'][index]
        path=self.root/'runs'/f"{task['configuration']}-seed1"/'logs'/f'gpu-health-{array}_{index}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(reason='gpu_reset_required',phase='cuda_precheck',
            array_job_id=array,array_task_id=str(index),hostname=node)))
        return path

    def runner(self, args):
        self.calls.append(args)
        if args[0]=='sacct':
            return '\n'.join(self.rows[args[args.index('-j')+1]])
        return '200' if any(s.startswith('--array=') for s in args) else '201'

    def reconcile(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return gpu_retry.reconcile(self.root,self.plan,self.runner)

    def test_only_reset_startup_task_retried_and_excluded(self):
        self.assertTrue(self.reconcile())
        submissions=[a for a in self.calls if a[0]=='sbatch']
        self.assertEqual(len(submissions),2)
        self.assertIn('--array=1',submissions[0])
        self.assertIn('--exclude=ccc0388,ccc0390,ccc0391',submissions[0])
        self.assertIn('--dependency=afterany:200',submissions[1])
        self.assertEqual(self.plan['retry_stopped'], {'2':'not_a_reset_startup_failure'})
        self.assertNotIn('retry_intent',self.plan)

    def test_active_replacement_not_duplicated(self):
        self.reconcile()
        self.calls=[]
        self.rows['200']=['200_1|RUNNING|0:0|20|gres/gpu=1|ccc0392']
        self.assertTrue(self.reconcile())
        self.assertFalse(any(a[0]=='sbatch' for a in self.calls))

    def test_completed_replacement_never_restarted(self):
        self.reconcile()
        self.calls=[]
        self.rows['200']=['200_1|COMPLETED|0:0|21600|gres/gpu=1|ccc0392']
        self.assertFalse(self.reconcile())
        self.assertFalse(any(a[0]=='sbatch' for a in self.calls))

    def test_prior_hours_and_full_reservation_enforce_budget(self):
        self.plan['budget_gpu_hours']=10
        self.plan['previous_attempt_charge_gpu_hours']=1
        self.assertFalse(self.reconcile())
        self.assertEqual(self.plan['retry_stopped']['1'],'budget_limit')
        self.assertFalse(any(a[0]=='sbatch' for a in self.calls))

    def test_two_retries_then_stop(self):
        self.plan['retry_waves']=[dict(array='200',tasks=[1],sync='201'),
                                  dict(array='300',tasks=[1],sync='301')]
        self.rows['200']=['200_1|FAILED|86:0|20|gres/gpu=1|ccc0392']
        self.rows['300']=['300_1|FAILED|86:0|20|gres/gpu=1|ccc0393']
        self.marker('300',1,'ccc0393')
        self.assertFalse(self.reconcile())
        self.assertEqual(self.plan['retry_stopped']['1'],'retry_limit')
        self.assertFalse(any(a[0]=='sbatch' for a in self.calls))

    def test_incomplete_accounting_stops_submission(self):
        self.rows['100']=self.rows['100'][:1]
        with self.assertRaisesRegex(RuntimeError,'Accounting incomplete'):
            self.reconcile()
        self.assertFalse(any(a[0]=='sbatch' for a in self.calls))

    def test_unknown_submission_outcome_stops_duplicate(self):
        def broken(args):
            if args[0]=='sbatch':
                raise RuntimeError('Connection interrupted after submission')
            return self.runner(args)
        with self.assertRaises(RuntimeError):
            gpu_retry.reconcile(self.root,self.plan,broken)
        self.assertIn('retry_intent',json.loads((self.root/'launch.json').read_text()))
        with self.assertRaisesRegex(RuntimeError,'interrupted retry submission'):
            self.reconcile()

    def test_stale_marker_and_training_artifacts_disqualify(self):
        path=self.marker('100',1,'wrongnode')
        self.assertFalse(self.reconcile())
        self.marker('100',1,'ccc0391')
        (path.parent.parent/'manifest.json').write_text('{}')
        self.assertFalse(self.reconcile())

    def test_disabled_policy_does_not_query_scheduler(self):
        self.plan.pop('max_gpu_retries')
        self.assertFalse(self.reconcile())
        self.assertEqual(self.calls,[])

    def test_ledger_carries_all_waves_and_prior_charge(self):
        self.plan['retry_waves']=[dict(array='200',tasks=[1],sync='201')]
        self.plan['previous_attempt_charge_gpu_hours']=.1
        def accounting(args):
            array=args[args.index('-j')+1]
            return (array+'|FAILED|36|gres/gpu=1')
        with patch.object(batch2_cluster,'command',accounting):
            ledger=batch2_cluster.accounting(self.root,self.plan)
        self.assertAlmostEqual(ledger['consumed_gpu_hours'],.12)
        self.assertEqual(len(ledger['jobs']),2)

    def test_notification_dependency_follows_replacement_controller(self):
        self.plan['jobs']['notify']='150'
        self.assertTrue(self.reconcile())
        hold=['scontrol','hold','150']
        update=['scontrol','update','JobId=150','Dependency=afterany:201']
        release=['scontrol','release','150']
        self.assertLess(self.calls.index(hold),next(i for i,a in enumerate(self.calls) if a[0]=='sbatch'))
        self.assertLess(self.calls.index(update),self.calls.index(release))
        gpu=[a for a in self.calls if a[0]=='sbatch'][0]
        self.assertIn('--mail-type=NONE',gpu)

    def test_notification_stays_held_if_dependency_update_is_interrupted(self):
        self.plan['jobs']['notify']='150'
        def failed_update(args):
            if args[:2]==['scontrol','update']:
                raise RuntimeError('Connection interrupted')
            return self.runner(args)
        with self.assertRaises(RuntimeError):
            gpu_retry.reconcile(self.root,self.plan,failed_update)
        self.assertIn('retry_intent',self.plan)
        self.assertNotIn(['scontrol','release','150'],self.calls)

    def test_initial_mail_job_is_single_barrier_after_sync(self):
        root=self.root/'batch'
        root.mkdir()
        plan=dict(self.plan,jobs={},mail_user='xinyan14@illinois.edu')
        calls=[]
        def scheduler(args):
            calls.append(args)
            return str(400+sum(a[0]=='sbatch' for a in calls)) if args[0]=='sbatch' else ''
        with patch.object(batch2_cluster,'command',scheduler), \
                patch.object(batch2_cluster,'accounting',return_value={}), \
                patch.object(Path,'is_file',return_value=True),contextlib.redirect_stdout(io.StringIO()):
            batch2_cluster.submit(types.SimpleNamespace(submit=True),root,plan)
        jobs=[a for a in calls if a[0]=='sbatch']
        self.assertEqual(len(jobs),4)
        self.assertIn('--hold',jobs[2])
        self.assertIn('--dependency=afterany:403',jobs[3])
        self.assertIn('--mail-user=xinyan14@illinois.edu',jobs[3])
        self.assertIn('--mail-type=END,FAIL',jobs[3])
        self.assertNotIn('--mail-type=END,FAIL',jobs[0])
        self.assertEqual(calls[-2:],[['scontrol','release','403'],['scontrol','release','404']])

    def test_final_summary_separates_run_success_and_controller_failure(self):
        self.plan.update(target_epochs=1000,git_commit='test')
        for task in self.plan['tasks']:
            path=self.root/'runs'/f"{task['configuration']}-seed1"/'metrics/summary.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(status='completed',completed_epochs=1000)))
        result=batch_summary(self.root,self.plan,'COMPLETED')
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(len(result['completed']),3)
        self.assertEqual(batch_summary(self.root,self.plan,'FAILED')['status'],'INCOMPLETE')
        path.unlink()
        result=batch_summary(self.root,self.plan,'COMPLETED')
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['incomplete'],['M1-seed1'])

    def test_eight_run_staging_and_submission(self):
        root=self.root/'batch3'
        args=types.SimpleNamespace(batch_root=str(root),retry_of=None,
            exclude_nodes='ccc0388,ccc0390',max_gpu_retries=2,mail_user='xinyan14@illinois.edu')
        def git(args):
            return 'a'*40 if 'rev-parse' in args else ''
        def archive(args,**kwargs):
            with tarfile.open(args[args.index('-o')+1],'w'):
                pass
        with patch.object(batch2_cluster,'command',git), \
                patch.object(batch2_cluster.subprocess,'run',archive),contextlib.redirect_stdout(io.StringIO()):
            batch2_cluster.stage(args,tasks=BATCH3_TASKS,batch_name='batch3',budget_hours=56)
        plan=json.loads((root/'launch.json').read_text())
        self.assertEqual(plan['budget_gpu_hours'],56)
        self.assertEqual(plan['reservation_gpu_hours'],56)
        self.assertEqual(plan['array_expression'],'0-7')
        self.assertEqual(plan['settings']['BATCH1_PROTOCOL'],'batch3')
        self.assertTrue(plan['settings']['BATCH1_MANAGER'].endswith('scripts/batch3_cluster.py'))
        self.assertEqual([(t['configuration'],t['seed']) for t in plan['tasks']],BATCH3_TASKS)
        calls=[]
        def scheduler(args):
            calls.append(args)
            return str(500+sum(a[0]=='sbatch' for a in calls)) if args[0]=='sbatch' else ''
        with patch.object(batch2_cluster,'command',scheduler), \
                patch.object(batch2_cluster,'accounting',return_value={}), \
                patch.object(Path,'is_file',return_value=True),contextlib.redirect_stdout(io.StringIO()):
            batch2_cluster.submit(types.SimpleNamespace(submit=True),root,plan)
        gpu=[a for a in calls if a[0]=='sbatch'][1]
        self.assertIn('--array=0-7',gpu)
        self.assertIn('--exclude=ccc0388,ccc0390',gpu)
        self.assertIn('--time=420',gpu)


class CudaMarkerTest(unittest.TestCase):
    def test_reset_failure_marker_and_unclassified_failure(self):
        # Execute the real startup script with a CPU-only fake torch module.
        fake=types.SimpleNamespace(__version__='test',version=types.SimpleNamespace(cuda='test'),
            cuda=types.SimpleNamespace(is_available=lambda:False))
        script=Path(__file__).resolve().parents[1]/'scripts/check_cuda.py'
        with tempfile.TemporaryDirectory() as d:
            health,marker=Path(d)/'gpu.txt',Path(d)/'marker.json'
            env=dict(BATCH1_GPU_HEALTH_LOG=str(health),BATCH1_GPU_FAILURE_MARKER=str(marker),
                     SLURM_ARRAY_JOB_ID='100',SLURM_ARRAY_TASK_ID='1',SLURM_JOB_ID='101')
            with patch.dict(sys.modules,torch=fake),patch.dict(os.environ,env),contextlib.redirect_stdout(io.StringIO()):
                health.write_text('GPU Recovery Action : Reset\n')
                with self.assertRaises(SystemExit) as error:
                    runpy.run_path(str(script))
                self.assertEqual(error.exception.code,86)
                self.assertEqual(json.loads(marker.read_text())['reason'],'gpu_reset_required')
                marker.unlink()
                health.write_text('GPU Recovery Action : None\n')
                with self.assertRaises(RuntimeError):
                    runpy.run_path(str(script))
                self.assertFalse(marker.exists())


if __name__=='__main__':
    unittest.main()
