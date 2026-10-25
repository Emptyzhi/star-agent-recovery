import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_agentrewind as cli
from agentrewind_bench.config import resolve
from agentrewind_bench.controller import Controller
from agentrewind_bench.models import ModelDeadlineExpired, Reply, ResidentModel
from test_adaptation import CounterEnvironment,NATIVE,commit


class DeadlineTests(unittest.TestCase):
    def settings(self, extra=()):
        args = cli.parser().parse_args(['--dataset','game24','--model','fixture',
            '--backend','openai','--endpoint','http://127.0.0.1:8000/v1',
            '--task-ids','0','--csv','fixture.csv','--output-dir','fixture-output',
            '--run-namespace','fixture','--native-adapter-root',str(NATIVE),*extra])
        return resolve(args)[0]

    def test_cli_receipt_grace_is_explicit_and_validated(self):
        self.assertEqual(self.settings().response_grace_seconds, 60)
        self.assertEqual(self.settings(['--response-grace-seconds','17']).as_dict()['response_grace_seconds'], 17)
        with self.assertRaisesRegex(ValueError, 'response_grace_seconds'):
            self.settings(['--response-grace-seconds','0'])

    def model(self):
        sys.path.insert(0, str(NATIVE/'source'))
        model = ResidentModel.__new__(ResidentModel)
        model.model_type = 'qwen2'
        model.history_format = 'flat'
        model.prompt_format = 'native-chat'
        model.candidate_history = 'full'
        model.interface_format = 'native'
        model.recovery_prompt_policy = 'full'
        model.text_output_boundary = 'eos'
        model.text_history_format = 'sectioned'
        model.tokenizer = SimpleNamespace(apply_chat_template=lambda *args,**kwargs:'fixture prompt')
        model.prompt_limit, model.context_limit, model.count = 32000, 32768, 0
        model.response_grace_seconds = 60
        return model

    def test_preparation_time_reduces_server_budget_but_not_receipt_grace(self):
        model = self.model()
        observed = []
        class Client:
            def generate(self, request, *, budget_seconds):
                observed.append((budget_seconds, self.request_timeout))
                return SimpleNamespace(text='{}',input_tokens=31,output_tokens=2,
                    raw_response=dict(queue_time_seconds=.1,finish_reason='frame',budget_seconds=budget_seconds))
        model.client = Client()
        submitted = []
        with patch('model_backend.encode_prompt',return_value={'input_ids':[1]}):
            with patch('agentrewind_bench.models.time.perf_counter',side_effect=[100.,101.75]):
                reply = model.generate([{'role':'user','content':'fixture'}],1200,5.,lambda:submitted.append(True))
        self.assertEqual(observed, [(3.25,63.25)])
        self.assertEqual(submitted, [True])
        self.assertEqual(reply.stop_reason, 'frame')

    def test_expired_preparation_submits_nothing(self):
        model = self.model()
        submitted = []
        with patch('model_backend.encode_prompt',return_value={'input_ids':[1]}):
            with patch('agentrewind_bench.models.time.perf_counter',side_effect=[100.,106.]):
                with self.assertRaises(ModelDeadlineExpired):
                    model.generate([{'role':'user','content':'fixture'}],1200,5.,lambda:submitted.append(True))
        self.assertEqual(submitted, [])
        self.assertEqual(model.count, 0)

    def test_server_time_stop_retains_usage_and_never_executes_partial_output(self):
        class TimeStoppedModel:
            def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
                on_submit()
                return Reply('',31,1,.01,dict(finish_reason='time',budget_seconds=.1),'time')
        with tempfile.TemporaryDirectory() as tmp:
            result = Controller(CounterEnvironment(),TimeStoppedModel(),self.settings(),tmp,'fixture').run()
            events = [json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertEqual(result['status'], 'MODEL_TIME_LIMIT')
        self.assertEqual(result['executed_actions'], 0)
        self.assertEqual(result['cost']['model_requests'], 1)
        self.assertEqual(result['cost']['input_tokens'], 31)
        self.assertEqual(result['cost']['output_tokens'], 1)
        self.assertTrue(result['cost']['token_accounting_complete'])
        self.assertTrue(result['model_resource_error']['submitted'])
        self.assertEqual(next(e for e in events if e['event']=='model_request_completed')['text'], '')

    def test_empty_native_eos_is_charged_then_explicit_feedback(self):
        class EmptyModel:
            calls = 0
            def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
                on_submit()
                self.calls += 1
                return Reply('' if self.calls==1 else 'good',31,1,.01,dict(finish_reason='eos'),'eos')
        with tempfile.TemporaryDirectory() as tmp:
            result = Controller(CounterEnvironment(),EmptyModel(),self.settings(),tmp,'fixture').run()
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(result['protocol_rejections'][0]['raw_text'], '')
        self.assertEqual(result['cost']['model_requests'], 2)
        self.assertEqual(result['cost']['output_tokens'], 2)

    def test_pre_submission_time_limit_charges_no_phantom_request(self):
        class ExpiredModel:
            def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
                raise ModelDeadlineExpired('fixture preparation expired')
        with tempfile.TemporaryDirectory() as tmp:
            result = Controller(CounterEnvironment(),ExpiredModel(),self.settings(),tmp,'fixture').run()
        self.assertEqual(result['status'], 'MODEL_TIME_LIMIT')
        self.assertEqual(result['cost']['model_requests'], 0)
        self.assertFalse(result['model_resource_error']['submitted'])
        self.assertTrue(result['cost']['token_accounting_complete'])

    def test_transport_failure_is_not_converted_to_a_scored_outcome(self):
        class BrokenModel:
            def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
                on_submit()
                raise RuntimeError('HTTP 429 fixture; no completion receipt')
        with tempfile.TemporaryDirectory() as tmp:
            controller = Controller(CounterEnvironment(),BrokenModel(),self.settings(),tmp,'fixture')
            with self.assertRaisesRegex(RuntimeError,'HTTP 429'):
                controller.run()
            self.assertFalse(controller.cost()['token_accounting_complete'])

    def test_token_limit_uses_the_native_action_parser_instead_of_ending_the_task(self):
        class BoundedModel:
            calls=0
            def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
                on_submit()
                self.calls+=1
                text='bad' if self.calls==1 else 'good'
                return Reply(text,31,100,.01,{'finish_reason':'length'},'length')
        env=CounterEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(env,BoundedModel(),self.settings(),tmp,'fixture').run()
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertEqual(result['status'],'SUCCESS')
        self.assertEqual(result['executed_actions'],2)
        self.assertEqual(result['cost']['model_requests'],2)
        self.assertEqual(result['cost']['output_tokens'],200)
        self.assertEqual(sum(e['event']=='model_output_boundary' for e in events),2)

    def test_bounded_plain_memory_is_passed_literally_to_author_validation(self):
        responses=iter(['bad','backtrack_candidates[observed failure]',
            commit('rec_000001','Choose good instead.'),'good'])
        class BoundedModel:
            def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
                on_submit()
                text=next(responses)
                finish='length' if purpose=='recovery' else 'eos'
                return Reply(text,31,7,.01,{'finish_reason':finish},finish)
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),BoundedModel(),self.settings(),tmp,'fixture').run()
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['cost']['model_requests'],4)
        memory=next(e for e in events if e['event']=='recovery_memory_generated')
        self.assertEqual(memory['text'],'Choose good instead.')
        self.assertEqual(memory['stop_reason'],'length')


if __name__ == '__main__':
    unittest.main()
