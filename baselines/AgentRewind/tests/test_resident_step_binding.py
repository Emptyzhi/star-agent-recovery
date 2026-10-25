"""Runner and resident steps are distinct, explicit scheduler identities."""
from types import SimpleNamespace
from pathlib import Path
import io
import json
import os
import sys
import unittest
from unittest.mock import Mock,patch

from agentrewind_bench.models import resident_binding_step,prepare_resident
import run_agentrewind as cli


class ResidentStepBindingTests(unittest.TestCase):
    def args(self,extra=()):
        return cli.parser().parse_args(['--dataset','game24','--model','test',
            '--endpoint','http://127.0.0.1:41581','--task-ids','0','--output-dir','unused',
            '--run-namespace','test','--step-id','900',*extra])

    def test_existing_same_step_command_retains_its_binding(self):
        args=self.args()
        self.assertEqual(resident_binding_step(args),'900')
        self.assertIsNone(args.resident_step_id)

    def test_existing_service_binding_does_not_rewrite_runner_identity(self):
        args=self.args(('--resident-step-id','602'))
        self.assertEqual(resident_binding_step(args),'602')
        self.assertEqual(args.step_id,'900')

    def test_invalid_service_identity_is_not_inferred_or_repaired(self):
        for step in ('','batch','602.1',602):
            with self.subTest(step=step),self.assertRaises(ValueError):
                resident_binding_step(SimpleNamespace(step_id='900',resident_step_id=step))

    def prepared(self,service_step,runner_step='900'):
        args=self.args(('--resident-step-id','602','--parent-job-id','4000417','--node','gpu16'))
        args.native_adapter_root=Path('explicit-test-adapter')
        args.dtype='bfloat16'
        settings=SimpleNamespace(model='test',max_output_tokens=1200)
        health=dict(status='READY',fatal_error=None,slurm_step_id=service_step,
            max_num_seqs=40,max_new_tokens=1200,model_context_tokens=32768,
            max_prompt_tokens=32000,decoding=dict(eos_token_id=[2]))
        backend=SimpleNamespace(load_model_bundle=Mock(return_value=({'model_type':'test'},None,(2,))),
            check_model_health=Mock())
        opener=Mock();opener.open.return_value=io.StringIO(json.dumps(health))
        with patch.dict(sys.modules,{'model_backend':backend}), \
             patch.dict(os.environ,SLURM_JOB_ID='4000417',SLURM_STEP_ID=runner_step), \
             patch('socket.gethostname',return_value='gpu16'), \
             patch('agentrewind_bench.models.build_opener',return_value=opener):
            return prepare_resident(args,settings)

    def test_actual_runner_and_existing_service_steps_are_both_checked(self):
        self.assertEqual(self.prepared('602')[2]['slurm_step_id'],'602')
        with self.assertRaisesRegex(RuntimeError,'model-service step'):
            self.prepared('601')
        with self.assertRaisesRegex(RuntimeError,'workload step'):
            self.prepared('602',runner_step='901')


if __name__=='__main__':unittest.main()
