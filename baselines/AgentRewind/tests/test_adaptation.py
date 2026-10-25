import contextlib
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get('AGENTREWIND_NATIVE_ROOT',str(ROOT.parents[1]/'STAR/code')))
sys.path.insert(0,str(ROOT))
import run_agentrewind as cli
from agentrewind_bench.config import resolve
from agentrewind_bench.controller import Controller
from agentrewind_bench.models import Reply, ResidentModel, resident_chat_messages
from agentrewind_bench.native_protocol import decode_line
from agentrewind_bench.feedback import model_feedback


def commit(uid,note):
    return 'backtrack_commit['+json.dumps(dict(record_uid=uid,memory_summary=note))+']'


class ScriptedModel:
    def __init__(self,responses):
        self.responses=iter(responses)
        self.requests=[]
        self.purposes=[]

    def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
        self.requests.append(deepcopy(messages))
        self.purposes.append(purpose)
        on_submit()
        text=next(self.responses)
        return Reply(text,31,7,.001,dict(test_fixture=True,text=text),'stop')


class CounterEnvironment:
    def __init__(self):
        self.value=0
        self.snapshots=[]
        self.actions=0
        self.windows=[]

    def reset(self):return self.observe()
    def observe(self):return dict(observation='Counter value: '+str(self.value),accepted=True,
                                  done=self.value==2,reward=1 if self.value==2 else None,state={'value':self.value})
    def set_window(self,n):self.windows.append(n)
    def initial_prompt(self):return 'Native actions: bad enters state 1; good enters state 2.'
    def checkpoint(self):
        self.snapshots.append(self.value)
        return len(self.snapshots)-1
    def restore(self,key):
        self.value=self.snapshots[key]
        return dict(value=self.value)
    def execute(self,action):
        self.actions+=1
        self.value={'bad':1,'good':2}[action]
        result=self.observe()
        result['executed_action']=action
        return result
    def accounting(self):return dict(executed_actions=self.actions)


class AdaptationTests(unittest.TestCase):
    def test_native_public_feedback_retains_task_information_without_audit_duplicates(self):
        receipt=dict(observation='Original feedback',done=False,reward=None,accepted=False,
            state=dict(numbers=['2','3'],available_actions=['look','inventory']),
            native_receipt=dict(before='audit before',after='audit after',backend_requests=123))
        for dataset in ('game24','alfworld','webshop'):
            public=model_feedback(receipt,dataset)
            self.assertEqual(public['observation'],receipt['observation'])
            self.assertIs(public['action_accepted'],False)
            self.assertNotIn('native_receipt',public)
            self.assertNotIn('audit before',json.dumps(public))
            if dataset=='game24':self.assertEqual(public['state']['numbers'],['2','3'])
            elif dataset=='alfworld':self.assertEqual(public['available_actions'],['look','inventory'])
            else:self.assertNotIn('state',public)
        with self.assertRaises(KeyError):model_feedback({'observation':'partial'},'webshop')

    def test_partial_credit_and_action_acceptance_are_not_task_success(self):
        from agentrewind_bench.native_protocol import boundary_instruction
        for reward in (0,0.5,0.75,1):
            feedback=model_feedback(dict(observation='Score: '+str(reward),done=True,
                reward=reward,accepted=True),'webshop')
            self.assertTrue(feedback['action_accepted'])
            self.assertEqual(feedback['task_success'],reward==1)
            self.assertEqual(feedback['reward'],reward)
            self.assertNotIn('accepted',feedback)
        boundary=boundary_instruction(remaining=0,recoveries=1,blocked=True,candidates=False)
        self.assertIn('below 1 is partial credit',boundary)
        self.assertIn('backtrack_candidates[reason]',boundary)

    def test_terminal_partial_credit_can_restore_and_improve_without_free_calls(self):
        class PartialEnvironment(CounterEnvironment):
            def observe(self):
                result=super().observe()
                result['done']=self.value in (1,2)
                result['reward']={0:None,1:0.75,2:1}[self.value]
                return result
        model=ScriptedModel(['bad',
            commit('rec_000001','Choose good to satisfy the unmet constraint.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(PartialEnvironment(),model,self.config(),tmp,'fixture').run()
            events=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['cost']['model_requests'],3)
        feedback='\n'.join(m['content'] for m in model.requests[1])
        self.assertIn('"task_success": false',feedback)
        self.assertIn('"action_accepted": true',feedback)
        self.assertEqual(sum(e['event']=='terminal_recovery_required' for e in events),1)
        self.assertTrue(any('"recovery_required": true' in m['content'] for m in model.requests[1]))

    def test_native_command_acceptance_alone_does_not_request_recovery(self):
        model=ScriptedModel(['bad','good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.config(),tmp,'fixture').run()
            events=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],0)
        self.assertFalse(any(e['event']=='terminal_recovery_required' for e in events))

    def test_unsuccessful_terminal_requests_recovery_but_never_exceeds_allowance(self):
        class FailedEnvironment(CounterEnvironment):
            def observe(self):
                result=super().observe()
                result.update(done=self.value!=0,reward=0.5 if self.value else None)
                return result
        model=ScriptedModel(['bad',commit('rec_000001','Choose good.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(FailedEnvironment(),model,self.config(),tmp,'fixture').run()
            events=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertFalse(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['status'],'RECOVERY_LIMIT_REACHED')
        self.assertEqual(result['cost']['model_requests'],3)
        self.assertEqual(sum(e['event']=='terminal_recovery_required' for e in events),1)

    def test_audit_receipts_stay_in_events_and_out_of_model_and_author_context(self):
        class AuditedEnvironment(CounterEnvironment):
            def observe(self):
                value=super().observe()
                value['native_receipt']={'before':'AUDIT_ONLY'*5000,'after':'AUDIT_ONLY'*5000}
                return value
        model=ScriptedModel(['bad','good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(AuditedEnvironment(),model,self.config(),tmp,'fixture').run()
            events=(Path(tmp)/'events.jsonl').read_text()
            records='\n'.join(path.read_text() for path in (Path(tmp)/'author_traces').glob('*.jsonl'))
        self.assertTrue(result['success'])
        self.assertNotIn('AUDIT_ONLY',json.dumps(model.requests))
        self.assertNotIn('AUDIT_ONLY',records)
        self.assertIn('AUDIT_ONLY',events)

    def config(self,dataset='game24',extra=()):
        flag={'game24':['--csv','fixture.csv'],'alfworld':['--manifest','fixture.json'],
              'webshop':['--website','http://127.0.0.1:3001']}[dataset]
        args=cli.parser().parse_args(['--dataset',dataset,'--model','fixture','--backend','openai',
            '--endpoint','http://127.0.0.1:8000/v1','--task-ids','0','--output-dir','fixture-output',
            '--run-namespace','fixture','--native-adapter-root',str(NATIVE),
            '--action-budget-policy','segment',*flag,*extra])
        return resolve(args)[0]

    def test_dataset_windows_and_no_model_call_cap(self):
        for dataset,window in [('game24',12),('alfworld',50),('webshop',15)]:
            settings=self.config(dataset)
            self.assertEqual((settings.initial_actions,settings.recovery_actions),(window,window))
            self.assertIsNone(settings.as_dict()['model_request_limit'])
            self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v20_author_tools_1200')

    def test_unlimited_cli_preserves_native_windows_and_deadlines(self):
        for dataset,window in [('game24',12),('alfworld',50),('webshop',15)]:
            settings=self.config(dataset,extra=['--max-recoveries','unlimited'])
            self.assertIsNone(settings.max_recoveries)
            self.assertIsNone(settings.as_dict()['max_recoveries'])
            self.assertIsNone(settings.as_dict()['model_request_limit'])
            self.assertEqual((settings.initial_actions,settings.recovery_actions),(window,window))
            self.assertEqual((settings.task_timeout,settings.request_timeout),(None,600))
        for value in ('-1','infinity','1.5'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.config(extra=['--max-recoveries',value])

    def test_unlimited_repeated_restores_refresh_windows_without_refunding_cost(self):
        responses=[]
        notes=['Inspect arithmetic operands before choosing good.',
               'Check terminal feedback against required outcome.',
               'Avoid repeating observed bad branch.',
               'Compare checkpoint context with current evidence.',
               'Verify alternative transition reaches target value.',
               'Use successful command after restoring decision.']
        for i,note in enumerate(notes):
            responses.extend(['bad','backtrack_candidates[observed failure]',
                commit('rec_000001',note)])
        responses.append('good')
        model=ScriptedModel(responses);env=CounterEnvironment()
        settings=self.config(extra=['--max-recoveries','unlimited','--initial-actions','1','--recovery-actions','1'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(env,model,settings,tmp,'unlimited').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],6)
        self.assertEqual(result['executed_actions'],7)
        self.assertEqual(env.windows,[1]*7)
        self.assertEqual(result['cost']['model_requests'],19)
        self.assertEqual(result['cost']['input_tokens'],19*31)
        self.assertEqual(result['cost']['output_tokens'],19*7)
        self.assertEqual([s['recovery_count'] for s in result['recovery_stages']],list(range(7)))
        self.assertTrue(all(s['window_actions']==1 for s in result['recovery_stages']))
        for request,purpose in zip(model.requests,model.purposes):
            runtime=[m['content'] for m in request if m['content'].startswith('[AGENTREWIND_CURRENT_ALLOWANCE]')]
            self.assertEqual(len(runtime),1)
            self.assertIn('Recoveries remaining: unlimited.',runtime[0])

    def test_unlimited_partial_terminals_continue_to_new_recovery(self):
        class PartialEnvironment(CounterEnvironment):
            def observe(self):
                result=super().observe()
                result.update(done=self.value!=0,reward={0:None,1:0.75,2:1}[self.value])
                return result
        model=ScriptedModel(['bad',commit('rec_000001','Choose good.'),
            'bad',commit('rec_000001','Verify the remaining constraint before acting.'),'good'])
        settings=self.config(extra=['--max-recoveries','unlimited'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(PartialEnvironment(),model,settings,tmp,'partial').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],2)
        self.assertEqual(result['cost']['model_requests'],5)

    def test_unlimited_has_no_model_call_cap_or_free_malformed_requests(self):
        model=ScriptedModel(['backtrack_commit[unknown]']*70+['good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,
                self.config(extra=['--max-recoveries','unlimited']),tmp,'unlimited_calls').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],71)
        self.assertEqual(len(result['protocol_rejections']),70)
        self.assertEqual(result['completed_recoveries'],0)

    def test_exact_native_action_and_recovery_command_syntax(self):
        for text in ['click[amber]','search[hair clips]','take mug 1 from table 1','1 + 2 = 3']:
            self.assertEqual(decode_line(text),('execute_action',text))
        self.assertEqual(decode_line(commit('rec_000001','Use a different action.')),
            ('backtrack_commit',dict(record_uid='rec_000001',memory_summary='Use a different action.')))
        self.assertEqual(decode_line('backtrack_candidates["observed failure"]'),('backtrack_candidates','observed failure'))
        self.assertEqual(decode_line('click[Buy Now]\nObservation: invented'),('execute_action','click[Buy Now]'))
        self.assertEqual(decode_line('Action: click[x]'),('execute_action','Action: click[x]'))
        for text in ['', 'backtrack_candidates[]','backtrack_commit rec_000001','backtrack_commit["rec_000001]']:
            with self.assertRaises(ValueError):decode_line(text)

    def test_retained_prefix_consumes_trajectory_allowance_and_costs_remain_cumulative(self):
        model=ScriptedModel(['bad']*5+['backtrack_candidates[observed failure]',
            commit('rec_000003','Inspect the retained evidence before changing strategy.')]+['bad']*4)
        env=CounterEnvironment()
        settings=self.config(extra=['--action-budget-policy','trajectory',
            '--initial-actions','5','--recovery-actions','5'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(env,model,settings,tmp,'retained').run()
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        restore=next(e for e in events if e['event']=='checkpoint_restored')
        self.assertEqual((restore['previous_trajectory_actions'],restore['trajectory_actions'],
            restore['native_actions_remaining']),(5,1,4))
        self.assertEqual(env.windows,[5,4])
        self.assertEqual(result['current_trajectory_actions'],5)
        self.assertEqual(result['executed_actions'],9)
        self.assertEqual(result['cost']['model_requests'],11)
        self.assertEqual(result['status'],'ACTION_LIMIT_REACHED')
        self.assertEqual([s['window_actions'] for s in result['recovery_stages']],[5,4])
        boundary='\n'.join(m['content'] for m in model.requests[7])
        self.assertIn('Native actions remaining: 4.',boundary)
        self.assertIn('Current retained native trajectory length: 1',boundary)

    def test_restoring_an_earlier_checkpoint_refunds_only_discarded_native_positions(self):
        model=ScriptedModel(['bad','bad','backtrack_candidates[failed branch]',
            commit('rec_000003','Inspect the surviving prefix before another choice.'),
            'bad','backtrack_candidates[observed repetition]',
            commit('rec_000001','Use an alternative transition to reach the target.'),'good'])
        settings=self.config(extra=['--max-recoveries','unlimited','--action-budget-policy','trajectory',
            '--initial-actions','2','--recovery-actions','2'])
        env=CounterEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(env,model,settings,tmp,'earlier').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],2)
        self.assertEqual(result['executed_actions'],4)
        self.assertEqual(result['current_trajectory_actions'],1)
        self.assertEqual(env.windows,[2,1,2])
        self.assertEqual(result['cost']['model_requests'],8)

    def test_trajectory_cli_defaults_and_equal_limits_are_explicit(self):
        args=cli.parser().parse_args(['--dataset','webshop','--model','fixture',
            '--backend','openai','--endpoint','http://127.0.0.1:8000/v1','--task-ids','0',
            '--output-dir','fixture-output','--run-namespace','fixture','--website','http://127.0.0.1:3001',
            '--native-adapter-root',str(NATIVE),'--max-recoveries','unlimited'])
        settings,_=resolve(args)
        self.assertEqual(settings.action_budget_policy,'trajectory')
        self.assertEqual((settings.initial_actions,settings.recovery_actions),(15,15))
        self.assertIsNone(settings.max_recoveries)
        args.recovery_actions=14
        with self.assertRaisesRegex(ValueError,'same initial and restored trajectory limit'):resolve(args)

    def test_author_replay_memory_and_nonrefunded_usage(self):
        model=ScriptedModel(['bad','backtrack_candidates[observed state 1]',
            commit('rec_000001','failed_check: bad entered state 1; corrected_strategy: choose good.'), 'good'])
        env=CounterEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(env,model,self.config(extra=['--initial-actions','1']),tmp,'fixture').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['executed_actions'],2)
        self.assertEqual(result['cost']['model_requests'],4)
        self.assertEqual(result['cost']['input_tokens'],124)
        self.assertEqual(model.purposes,['action','recovery','recovery','action'])
        notebook=[m['content'] for m in model.requests[-1] if m['role']=='system' and m['content'].startswith('Backtrack memory notebook')]
        self.assertEqual(len(notebook),1)
        self.assertIn('choose good',notebook[0])
        self.assertFalse(any(m['role']=='assistant' and m['content']=='bad' for m in model.requests[-1]))
        self.assertEqual([s['recovery_count'] for s in result['recovery_stages']],[0,1])
        self.assertEqual(result['recovery_stages'][0]['cost']['model_requests'],3)

    def test_exhausted_recovery_request_is_charged_error_not_extra_restore_or_terminal(self):
        model=ScriptedModel(['bad','backtrack_candidates[observed failure]',
            commit('rec_000001','Choose good instead of bad.'),
            'backtrack_candidates[another failure]','good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.config(extra=['--initial-actions','1']),tmp,'fixture').run()
            events=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['executed_actions'],2)
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(len(result['protocol_rejections']),1)
        self.assertEqual(sum(e['event']=='checkpoint_restored' for e in events),1)
        runtime=next(m['content'] for m in model.requests[-1] if m['content'].startswith('[AGENTREWIND_CURRENT_ALLOWANCE]'))
        self.assertIn('Recovery is unavailable',runtime)
        self.assertNotIn('To abandon a failed branch',runtime)

    def test_invalid_target_and_empty_output_are_not_repaired(self):
        model=ScriptedModel(['','bad','backtrack_candidates[failure]',
            commit('invented','Check the observed state.'),commit('rec_000001','Choose good.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.config(),tmp,'fixture').run()
        self.assertTrue(result['success'])
        self.assertEqual(len(result['protocol_rejections']),1)
        self.assertEqual(result['protocol_rejections'][0]['raw_text'],'')
        self.assertEqual(result['cost']['model_requests'],6)
        self.assertEqual(result['executed_actions'],2)

    def test_zero_recoveries_retains_native_action_budget(self):
        model=ScriptedModel(['backtrack_candidates[not allowed]','good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.config(extra=['--max-recoveries','0']),tmp,'fixture').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],0)
        self.assertEqual(result['executed_actions'],1)
        self.assertEqual(result['cost']['model_requests'],2)

    def test_more_than_64_actual_requests_have_no_call_cap(self):
        model=ScriptedModel(['backtrack_commit[unknown]']*70+['good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.config(),tmp,'fixture').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],71)
        self.assertEqual(result['executed_actions'],1)

    def test_author_memory_rejection_is_visible_and_requires_new_model_note(self):
        model=ScriptedModel(['bad','backtrack_candidates[failure]',
            commit('rec_000001','The task is not feasible.'),
            commit('rec_000001','Choose good instead.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.config(),tmp,'fixture').run()
            events=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],5)
        rejection=next(e for e in events if e['event']=='author_tool_rejected')
        self.assertEqual(rejection['arguments']['memory_summary'],'The task is not feasible.')
        notebook='\n'.join(m['content'] for m in model.requests[-1] if m['role']=='system')
        self.assertIn('Choose good instead.',notebook)
        self.assertNotIn('The task is not feasible.',notebook)

    def test_empty_memory_feedback_preserves_mistral_dialogue_and_charges_new_note(self):
        model=ScriptedModel(['bad','backtrack_candidates[failure]',commit('rec_000001',''),
            commit('rec_000001','Choose good.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.config(),tmp,'fixture').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(result['executed_actions'],2)
        for request in model.requests:
            from agentrewind_bench.native_protocol import render_native_messages
            render_native_messages(request)

    def test_unexpected_environment_failure_remains_fatal(self):
        class Broken(CounterEnvironment):
            def execute(self,action):raise RuntimeError('fixture transport failure')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'transport failure'):
                Controller(Broken(),ScriptedModel(['good']),self.config(),tmp,'fixture').run()

    def test_cli_http_native_game24_nondefault_settings_and_plain_memory(self):
        responses=['answer: 1 + 2 + 3 + 4 = 24','backtrack_candidates[wrong arithmetic]',
            commit('rec_000001','The sum failed; use multiplication instead.'),
            'answer: 1 * 2 * 3 * 4 = 24']
        received=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                data=json.dumps(dict(choices=[dict(message={'content':responses.pop(0)},finish_reason='stop')],
                                     usage={'prompt_tokens':41,'completion_tokens':9})).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);csv=root/'puzzles.csv';csv.write_text('Puzzles\n1 2 3 4\n')
                with contextlib.redirect_stdout(io.StringIO()):
                    code=cli.main(['--dataset','game24','--model','fixture-alt','--backend','openai',
                        '--endpoint',f'http://127.0.0.1:{server.server_port}/v1','--csv',str(csv),'--task-ids','0',
                        '--action-budget-policy','segment',
                        '--initial-actions','1','--recovery-actions','2','--max-output-tokens','77',
                        '--actor-output-tokens','77',
                        '--temperature','.4','--top-k','7','--output-dir',str(root/'out'),'--run-namespace','fixture',
                        '--native-adapter-root',str(NATIVE)])
                self.assertEqual(code,0)
                s=json.loads((root/'out/task_000/summary.json').read_text())
                self.assertTrue(s['success']);self.assertEqual(s['cost']['model_requests'],4)
                self.assertEqual(s['environment_accounting']['executed_actions'],2)
                self.assertTrue(all(p['model']=='fixture-alt' and p['max_tokens']==77 and p['temperature']==.4 and p['top_k']==7 for p in received))
                self.assertTrue(all('response_format' not in p for p in received))
                self.assertEqual([p.get('stop') for p in received],[['\n'],None,None,['\n']])
        finally:
            server.shutdown();server.server_close();thread.join(timeout=5)

    def test_mistral_roles_preserve_contents_without_consecutive_assistant_turns(self):
        messages=[{'role':'system','content':'base'}, {'role':'user','content':'task'},
                  {'role':'system','content':'notebook'}, {'role':'user','content':'boundary'}]
        native=resident_chat_messages(messages,'mistral')
        self.assertEqual(native,[{'role':'system','content':'base\n\nnotebook'},
                                 {'role':'user','content':'task\n\nboundary'}])

    def test_resident_native_action_and_memory_boundaries(self):
        sys.path.insert(0,str(NATIVE/'source'))
        observed=[]
        class Client:
            def generate(self,request,**kwargs):
                observed.append(request.stop_mode)
                return SimpleNamespace(text='good',input_tokens=31,output_tokens=7,
                    raw_response=dict(queue_time_seconds=.001,finish_reason='eos'))
        model=ResidentModel.__new__(ResidentModel);model.model_type='qwen2'
        model.history_format='flat'
        model.prompt_format='native-chat'
        model.candidate_history='full'
        model.interface_format='native'
        model.recovery_prompt_policy='full'
        model.text_output_boundary='eos'
        model.text_history_format='sectioned'
        model.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:'fixture')
        model.prompt_limit,model.context_limit,model.count=32000,32768,0
        model.response_grace_seconds=60;model.client=Client()
        with patch('model_backend.encode_prompt',return_value={'input_ids':[1]}):
            model.generate([{'role':'user','content':'fixture'}],100,5,lambda:None,purpose='action')
            model.generate([{'role':'system','content':'[AGENTREWIND_CURRENT_ALLOWANCE]\nOutput phase: recovery-command.'},
                {'role':'user','content':'fixture\nAct 1>'}],256,5,lambda:None,purpose='recovery')
        self.assertEqual(observed,['first_line',None])


if __name__=='__main__':unittest.main()
