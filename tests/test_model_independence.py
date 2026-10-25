"""Offline model-switch and baseline regression tests; no weights or services."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'source'), str(PACKAGE/'vllm_service')]

from model_backend import load_model_bundle, check_model_health, encode_prompt
from experiment_settings import Sampling
from resident_settings import engine_options, build_parser as service_parser
from scoped_recovery_batch import build_parser as worker_parser
from dispatch_webshop import worker_command
from bound_resident_client import BoundResidentClient
import scoped_recovery_batch as batch
from test_web_v16 import make_controller


class Tokenizer:
    def __init__(self, eos, pad, prefix):
        self.eos_token_id, self.pad_token_id, self.prefix = eos, pad, prefix

    def __call__(self, prompt, add_special_tokens, **kwargs):
        raw = [ord(c) for c in prompt]
        return {'input_ids': (self.prefix if add_special_tokens else []) + raw}


class ModelIndependenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, family, dtype='bfloat16'):
        if family == 'qwen2':
            architecture, eos, pad, prefix = 'Qwen2ForCausalLM', [151645,151643], 151643, []
        elif family == 'llama':
            architecture, eos, pad, prefix = 'LlamaForCausalLM', [128001,128008,128009], None, [128000]
        else:
            architecture, eos, pad, prefix = 'OtherForCausalLM', [7], 0, []
        path = self.root/family
        path.mkdir(exist_ok=True)
        (path/'config.json').write_text(json.dumps(dict(model_type=family,
            architectures=[architecture], eos_token_id=eos)), encoding='utf-8')
        (path/'generation_config.json').write_text(json.dumps(dict(eos_token_id=eos)), encoding='utf-8')
        tokenizer = Tokenizer(eos[0], pad, prefix)
        loader = Mock(return_value=tokenizer)
        model, loaded, ids = load_model_bundle(path, dtype, tokenizer_loader=loader)
        loader.assert_called_once_with(str(path.resolve()), local_files_only=True, trust_remote_code=False)
        return model, loaded, ids

    def test_path_selects_all_model_properties_and_engine_tokenizer(self):
        for family in ('qwen2', 'llama', 'other'):
            with self.subTest(family=family):
                model, tokenizer, eos = self.bundle(family, 'float16')
                options = engine_options(model, context_limit=32768, max_num_seqs=40, seed=123)
                self.assertEqual(options['model'], model['model_path'])
                self.assertEqual(options['tokenizer'], model['model_path'])
                self.assertEqual(options['dtype'], 'float16')
                self.assertEqual(model['model_type'], family)
                self.assertIn(tokenizer.eos_token_id, eos)
                self.assertEqual(encode_prompt(tokenizer,'Action:')['input_ids'], tokenizer.prefix+[ord(c) for c in 'Action:'])

    def test_padding_preserves_native_or_uses_declared_eos_policy(self):
        qwen, qt, _ = self.bundle('qwen2')
        llama, lt, _ = self.bundle('llama')
        self.assertEqual((qwen['padding_policy'], qt.pad_token_id), ('native',151643))
        self.assertEqual((llama['padding_policy'], lt.pad_token_id), ('eos',lt.eos_token_id))
        self.assertEqual(lt.padding_side,'left')

    def test_model_config_eos_used_when_generation_config_absent(self):
        model, tokenizer, expected = self.bundle('llama')
        (Path(model['model_path'])/'generation_config.json').unlink()
        _, _, actual = load_model_bundle(model['model_path'],tokenizer_loader=Mock(return_value=tokenizer))
        self.assertEqual(actual, expected)

    def test_foreign_eos_rejected_instead_of_guessed(self):
        model, tokenizer, _ = self.bundle('qwen2')
        (Path(model['model_path'])/'generation_config.json').write_text('{"eos_token_id":128009}')
        with self.assertRaisesRegex(ValueError,'EOS'):
            load_model_bundle(model['model_path'],tokenizer_loader=Mock(return_value=tokenizer))

    def test_no_model_family_whitelist_or_stale_package_config(self):
        model, _, _ = self.bundle('other')
        self.assertEqual(model['model_class'],'OtherForCausalLM')
        self.assertFalse((PACKAGE/'model_config.json').exists())

    def test_runner_and_service_use_same_model_arguments(self):
        worker = ['--website','http://127.0.0.1:3001','--endpoints','http://127.0.0.1:41581',
                  'http://127.0.0.1:41582','--output-dir','out','--parent-job-id','123',
                  '--node','gpu13','--run-namespace','run','--task-ids','0,1','--gpu-index','0']
        service = ['--state','state','--parent-job-id','123','--node','gpu13','--gpu-index','0','--port','41581']
        for flag in ('--local_llm_path','--model-path'):
            w = worker_parser().parse_args(worker+[flag,'/models/selected','--dtype','float16'])
            s = service_parser().parse_args(service+[flag,'/models/selected','--dtype','float16'])
            self.assertEqual((w.model_path,w.dtype),(s.model_path,s.dtype))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            worker_parser().parse_args(worker+['--model-path','/models/x','--model-config','ignored.json'])

    def test_queue_propagates_model_and_dtype_to_both_workers(self):
        args = SimpleNamespace(model_path='/models/selected',dtype='float16',parent_job_id='123',node='gpu13')
        for gpu in (0,1):
            command = worker_command(PACKAGE,args,'run',[gpu,gpu+2],gpu,self.root)
            parsed = worker_parser().parse_args(command[2:])
            self.assertEqual(parsed.model_path,args.model_path)
            self.assertEqual(parsed.dtype,args.dtype)
            self.assertEqual(parsed.gpu_index,gpu)
            self.assertEqual(parsed.workers_per_gpu,40)

    def test_resident_identity_validation_accepts_both_and_rejects_cross_model(self):
        for family in ('qwen2','llama'):
            model, _, _ = self.bundle(family)
            health = dict(model,model_backend=model)
            check_model_health(health,model)
            health['model_class'] = 'WrongModel'
            with self.assertRaisesRegex(ValueError,'model_class'):
                check_model_health(health,model)

    def test_bound_client_accepts_correct_non_llama_resident(self):
        model, _, _ = self.bundle('qwen2')
        binding = dict(parent_job_id='123',step_id='4',hostname='node',gpu=0,
                       gpu_uuid='GPU-test',port=41581,pid=456,instance='instance')
        health = dict(model,model_backend=model,parent_job_id='123',slurm_step_id='4',
            hostname='node',cuda_visible_devices='0',gpu_uuid='GPU-test',gpu_name='NVIDIA A100-PCIE-40GB',
            status='READY',engine='vllm',engine_version='0.6.6',loaded_once=True,max_num_seqs=40,
            chat_template_applied=False,prompt_truncation=False,service_pid=456,
            service_instance_id='instance',service_start_time=1,stop_modes=['first_line','first_nonempty_line'],
            sampling_run=dict(experiment_id='run',request_seed=None,per_task_reseed=False,per_request_reseed=False),
            decoding=dict(do_sample=True,temperature=.1,top_p=1.,top_k=20,repetition_penalty=1.05,request_seed=None))
        client = BoundResidentClient(binding,'run',model)
        self.addCleanup(client.close)
        client.http = Mock()
        client.http.get.return_value.json.return_value = health
        self.assertEqual(client.health()['model_type'],'qwen2')

    def test_existing_llama_health_schema_reuses_canonical_backend_identity(self):
        model, _, _ = self.bundle('llama')
        health = dict(model_backend=model, model_path=model['model_path'],
                      model_class=model['model_class'],torch_dtype=model['torch_dtype'])
        check_model_health(health,model)
        health['model_path'] = '/models/wrong'
        with self.assertRaisesRegex(ValueError,'model_path'):
            check_model_health(health,model)

    def test_actual_batch_entry_delivers_selected_model_to_each_task(self):
        for family in ('qwen2','llama'):
            model, tokenizer, eos = self.bundle(family,'float16')
            health = dict(model,model_backend=model,max_num_seqs=40,
                model_context_tokens=32768,max_prompt_tokens=32000,max_new_tokens=256,
                decoding=dict(**Sampling().as_dict(),eos_token_id=list(eos)),gpu_uuid='GPU-test',
                service_pid=456,service_instance_id='instance')
            backend = dict(parent_job_id='123',step_id='4',hostname='node',
                interface='table1_webshop_attributes_v1',initialization_seed=1,show_attributes=True)
            responses = [Mock(),Mock()]
            responses[0].json.return_value=health
            responses[1].json.return_value=backend
            output = self.root/('run_'+family)
            argv = ['run_webshop.py','--local_llm_path',model['model_path'],'--dtype','float16',
                '--website','http://127.0.0.1:3001','--endpoints','http://127.0.0.1:41581',
                'http://127.0.0.1:41582','--output-dir',str(output),'--parent-job-id','123',
                '--node','node','--run-namespace','run','--task-ids','0,2','--gpu-index','0']
            def run_case(task,*args,**kwargs):
                return dict(identity=dict(task_id=task,controller_version='baseline'),
                            success=True,final_public_reward=1.,benchmark_reward=1.)
            with (patch.object(sys,'argv',argv), patch.dict(os.environ,dict(
                    SLURM_JOB_ID='123',SLURM_STEP_ID='4',CUDA_VISIBLE_DEVICES='0')),
                  patch.object(batch.socket,'gethostname',return_value='node'),
                  patch.object(batch,'load_model_bundle',return_value=(model,tokenizer,eos)) as resolve,
                  patch('requests.get',side_effect=responses),
                  patch.object(batch,'run_case',side_effect=run_case) as execute,
                  redirect_stdout(StringIO())):
                batch.main()
            resolve.assert_called_once_with(model['model_path'],'float16')
            self.assertEqual(execute.call_count,2)
            for call in execute.call_args_list:
                self.assertIs(call.args[3],tokenizer)
                self.assertEqual(call.args[4],model)
                self.assertEqual(call.kwargs['prompt_limit'],32000)
            record = json.loads((output/'run_config.json').read_text())
            self.assertEqual(record['model'],model)

    def test_public_entry_points_import_without_a_loaded_model_or_cluster_environment(self):
        for entry in ('run_webshop.py','serve_model.py','dispatch_webshop.py'):
            result = subprocess.run([sys.executable,str(PACKAGE/entry),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--local_llm_path',result.stdout)

    def test_invalid_navigation_gets_real_receipt_not_new_early_termination(self):
        c, _, events = make_controller(['click[Next >]', 'Use the search form.',
            'search[cup]','click[A]','click[red]','click[Buy Now]'])
        self.assertEqual(c.run()['status'],'SUCCESS')
        first = c.occurrences[0]
        self.assertTrue(first.receipt['invalid_action'])
        self.assertEqual(first.action,'click[Next >]')
        self.assertFalse(any(e['event']=='model_protocol_failure' for e in events))

    @unittest.skipUnless(os.environ.get('STAR_BASELINE_ROOT'),'set STAR_BASELINE_ROOT for archived source parity')
    def test_controller_prompts_parser_and_framing_equal_llama_baseline(self):
        baseline = Path(os.environ['STAR_BASELINE_ROOT'])
        names = ['alf_simple_v64.py',
                 'web_v16_evidence.py','tree_history.py','webshop_semantics.py',
                 'webshop_public_env.py','icr_webshop.py','icr_output_framing.py',
                 'scoped_recovery.py']
        for name in names:
            with self.subTest(source=name):
                self.assertEqual((PACKAGE/'source'/name).read_bytes(),(baseline/'source'/name).read_bytes())
        for folder in ('prompts','vendor'):
            for file in (baseline/folder).rglob('*.txt'):
                self.assertEqual((PACKAGE/file.relative_to(baseline)).read_bytes(),file.read_bytes())


if __name__ == '__main__':
    unittest.main()
