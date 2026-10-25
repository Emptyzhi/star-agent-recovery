"""Exercise nondefault CLI settings through queues, requests and the controller."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PACKAGE),str(PACKAGE/'source'),str(PACKAGE/'vllm_service')]
from experiment_settings import Experiment,Sampling,settings_flags,settings_from_args,validate_runtime_settings
from scoped_recovery_batch import build_parser as worker_parser
from dispatch_webshop import build_parser as dispatch_parser,resolve_run_settings,worker_command
from resident_launch_config import build_parser as startup_parser
import resident_launch_config
from resident_settings import engine_options
from star_vllm_protocol import validate_request,ProtocolViolation
from bound_resident_client import BoundResidentClient
from icr_core import GenerationRequest,ProtocolError,BudgetExhausted,Ledger
from web_v16_run import verify_backend
from test_web_v16 import make_controller


class ConfigurableSettingsTests(unittest.TestCase):
    def test_webshop_has_no_request_or_derived_token_count_cap(self):
        limits=Experiment().ledger_limits(24000)
        self.assertEqual((limits.environment_calls,limits.model_requests,limits.output_tokens,limits.input_tokens),
                         (32,None,None,None))

    def test_queue_cli_roundtrip_constructs_real_ledger_with_integer_task_timeout(self):
        args=SimpleNamespace(model_path='/models/x',dtype='bfloat16',parent_job_id='123',node='node')
        command=worker_command(PACKAGE,args,'run',[0,2],0,Path('out'),
                               experiment=Experiment(task_timeout=90,request_timeout=.5))
        parsed=worker_parser().parse_args(command[2:])
        settings=settings_from_args(parsed,Experiment)
        ledger=Ledger(settings.ledger_limits(parsed.prompt_limit))
        self.assertEqual(ledger.limits.elapsed_seconds,90)
        self.assertIs(type(ledger.limits.elapsed_seconds),int)
        self.assertEqual(settings.request_timeout,.5)

    def test_nondefault_windows_and_red_tokens_reach_execution(self):
        cfg=Experiment(initial_actions=1,recovery_actions=4,actor_output_tokens=77,
                       red_reflection_tokens=222,red_selection_tokens=33)
        c,requests,_=make_controller(['search[cup]','Select red before buying.','ACTION 0',
            'search[cup]','click[A]','click[red]','click[Buy Now]'],cfg)
        state=c.run()
        self.assertEqual(state['status'],'SUCCESS')
        self.assertEqual([r.output_limit for r in requests],[77,222,33,77,77,77,77])
        self.assertEqual(state['forward_trials'][0]['forward_actions_granted'],4)
        self.assertEqual(state['first_window_actions'],1)

    def test_yellow_token_limits_are_independent_from_green(self):
        cfg=Experiment(actor_output_tokens=70,yellow_diagnosis_tokens=45,yellow_output_tokens=55)
        c,requests,_=make_controller(['search[cup','Use the search form.','search[cup]',
                                     'click[A]','click[red]','click[Buy Now]'],cfg)
        self.assertEqual(c.run()['status'],'SUCCESS')
        self.assertEqual([r.output_limit for r in requests],[70,45,55,70,70,70])

    def test_yellow_action_budget_uses_feedback_and_returns_on_progress(self):
        cfg=Experiment(yellow_actions=2)
        c,_,_=make_controller(['search[cup]','click[Next >]','Return to search and refine.',
            'click[Back to Search]','search[red cup]','click[A]','click[red]','click[Buy Now]'],cfg)
        self.assertEqual(c.run()['status'],'SUCCESS')
        self.assertEqual(c.local_outcomes[0]['actions'],2)

    def test_yellow_cannot_outlive_shared_window(self):
        cfg=Experiment(initial_actions=3,yellow_actions=8)
        c,requests,_=make_controller(['search[cup]','click[Next >]','Return to search.',
                                     'click[Back to Search]'],cfg)
        c._run_window()
        self.assertEqual(c._forward_window()['new_actions_remaining'],0)
        self.assertEqual(c.local_outcomes[0]['actions'],1)
        self.assertEqual(len(requests),4)

    def test_yellow_entry_limit_zero_is_respected(self):
        c,_,events=make_controller(['click[Next >]','search[cup]','click[A]',
                                   'click[red]','click[Buy Now]'],Experiment(yellow_entries=0))
        self.assertEqual(c.run()['status'],'SUCCESS')
        self.assertFalse(any(e['event']=='yellow_local_diagnosis' for e in events))

    def test_explicit_model_request_cap_is_enforced(self):
        c,requests,_=make_controller(['search[cup]','click[A]'],Experiment(model_request_limit=1))
        with self.assertRaises(BudgetExhausted):c.run()
        self.assertEqual(len(requests),1)

    def test_sampling_to_vllm_is_exact_and_greedy_is_explicit(self):
        sample=Sampling(temperature=.7,top_p=.85,top_k=-1,repetition_penalty=1.2)
        self.assertEqual(sample.engine_parameters(),dict(temperature=.7,top_p=.85,top_k=-1,repetition_penalty=1.2))
        self.assertEqual(Sampling(do_sample=False).engine_parameters()['temperature'],0)
        with self.assertRaises(ValueError):Sampling(temperature=0)
        with self.assertRaises(ValueError):Sampling(top_k=0)
        with self.assertRaises(ValueError):Sampling(temperature=float('nan'))

    def test_request_protocol_accepts_configured_output_and_sampling(self):
        request=dict(route='normal',prompt='Action:',experiment_id='task',task_seed=None,
                     call_index=0,max_new_tokens=512,sampling=Sampling(top_k=-1).as_dict())
        validate_request(request,max_output_tokens=1024)
        with self.assertRaises(ProtocolViolation):validate_request(request,max_output_tokens=256)
        request['sampling']['unknown']=1
        with self.assertRaises(ProtocolViolation):validate_request(request,max_output_tokens=1024)

    def test_actual_client_payload_carries_sampling_budget_and_timeout(self):
        binding=dict(parent_job_id='123',step_id='4',hostname='node',gpu=0,
                     gpu_uuid='GPU-test',port=41581,pid=456,instance='instance')
        sampling=Sampling(temperature=.7,top_p=.8,top_k=-1,repetition_penalty=1.2)
        client=BoundResidentClient(binding,'run',{},sampling=sampling,request_timeout=7)
        client.active=True
        client.identity=dict(service_pid=456,service_instance_id='instance',service_start_time=1,gpu_uuid='GPU-test')
        client.health_record=dict(max_new_tokens=512,model_context_tokens=32768,
                                  request_sampling_supported=True,flexible_output_limits=True)
        client.http=Mock()
        raw=dict(gpu_identity=dict(parent_job_id='123',slurm_step_id='4',hostname='node',
            cuda_visible_devices='0',gpu_uuid='GPU-test',gpu_name='NVIDIA A100-PCIE-40GB'),
            runtime=client.identity,decoding=dict(**sampling.as_dict(),request_seed=None),
            route='normal',task_seed=None,call_index=0,effective_seed=None,output_token_limit=128,
            input_tokens=20,output_tokens=1,output_token_ids=[1],text='search[cup]',raw_output='search[cup]',
            queue_time_seconds=0,model_active_step_seconds=.1,budget_seconds=None)
        client.http.post.return_value.json.return_value=raw
        request=GenerationRequest('r','Action:',1,'normal',(),128)
        result=client.generate(request)
        payload=client.http.post.call_args.kwargs['json']
        validate_request(payload,max_output_tokens=512)
        self.assertEqual(payload['sampling'],sampling.as_dict())
        self.assertEqual(payload['max_new_tokens'],128)
        self.assertIsNone(payload['task_seed'])
        self.assertEqual(client.http.post.call_args.kwargs['timeout'],7)
        self.assertEqual(result.request.output_limit,128)

    def test_queue_cli_and_resident_defaults_have_explicit_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'defaults.json'
            path.write_text(json.dumps(dict(experiment={'environment_seed':99,'initial_actions':12},
                sampling={'temperature':.3,'top_k':30},workers_per_gpu=7)))
            args=dispatch_parser().parse_args(['--model-path','/models/x','--request','r.json',
                '--results-root',temp,'--parent-job-id','123','--node','node',
                '--defaults',str(path),'--top-k','40','--recovery-actions','9'])
            experiment,sampling,capacity=resolve_run_settings(args,dict(sampling={'top_p':.8},workers_per_gpu=6))
            self.assertEqual((experiment.environment_seed,experiment.initial_actions,experiment.recovery_actions),(99,12,9))
            self.assertEqual((sampling.temperature,sampling.top_k,sampling.top_p),(.3,40,.8))
            command=worker_command(PACKAGE,args,'run',[0,2],0,Path(temp),
                experiment=experiment,sampling=sampling,**capacity)
            parsed=worker_parser().parse_args(command[2:])
            self.assertEqual((parsed.environment_seed,parsed.workers_per_gpu,parsed.top_k),(99,6,40))
            self.assertEqual(parsed.recovery_actions,9)
            with self.assertRaises(ValueError):resolve_run_settings(args,dict(environment_seed=1,experiment={'environment_seed':2}))

    def test_startup_cli_propagates_seed_sampling_and_engine_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'settings.json'
            argv=['launch','--model-path','/models/x','--output',str(path),'--seed','99',
                '--max-num-seqs','12','--workers-per-gpu','7','--temperature','.4',
                '--top-k','-1','--actor-output-tokens','150','--max-output-tokens','512',
                '--prompt-limit','31000','--gpu-memory-utilization','.9','--max-num-batched-tokens','4096']
            model=dict(model_path='/models/x',torch_dtype='bfloat16')
            with patch.object(sys,'argv',argv),patch.object(resident_launch_config,'load_model_bundle',return_value=(model,None,())),redirect_stdout(StringIO()) as stream:
                resident_launch_config.main()
            record=json.loads(path.read_text())
            self.assertEqual(record['experiment']['environment_seed'],99)
            self.assertEqual(record['sampling']['temperature'],.4)
            self.assertEqual(record['max_num_seqs'],12)
            self.assertEqual(stream.getvalue().splitlines()[2],'99')
            options=engine_options(model,context_limit=32768,max_num_seqs=12,seed=123,
                                   gpu_memory_utilization=.9,max_num_batched_tokens=4096)
            self.assertEqual((options['gpu_memory_utilization'],options['max_num_batched_tokens']),(.9,4096))

    def test_environment_seed_requires_actual_backend_identity(self):
        binding=dict(parent_job_id='123',step_id='4',hostname='node')
        backend=dict(binding,interface='table1_webshop_attributes_v1',initialization_seed=99,show_attributes=True)
        verify_backend(backend,binding,99)
        with self.assertRaisesRegex(ProtocolError,'restart the website'):verify_backend(backend,binding,1)

    def test_legacy_service_reuse_is_not_silent_reconfiguration(self):
        health=dict(max_new_tokens=256,max_prompt_tokens=32000,model_context_tokens=32768,
                    decoding=Sampling().as_dict())
        validate_runtime_settings(health,Experiment(),Sampling(),32000)
        with self.assertRaisesRegex(ValueError,'sampling'):validate_runtime_settings(health,Experiment(),Sampling(top_k=5),32000)
        with self.assertRaisesRegex(ValueError,'100-token'):validate_runtime_settings(health,Experiment(actor_output_tokens=50),Sampling(),32000)


if __name__=='__main__':unittest.main()
