"""Direct text delivery without dropping author guidance, history or accounting."""
from copy import deepcopy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from test_adaptation import AdaptationTests, CounterEnvironment, ScriptedModel
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.models import OpenAIModel, render_resident_request
from agentrewind_bench.native_protocol import (RUNTIME_PREFIX, append_actor_step,
    initial_actor_prompt, render_plain_continuation, without_runtime_allowance)


class PlainTransportTests(unittest.TestCase):
    def settings(self):
        return AdaptationTests().config(extra=['--history-format','dialogue','--recovery-format','lines',
            '--notebook-item-char-limit','4000','--rejected-history-keep','2',
            '--max-recoveries','unlimited','--max-terminal-failures','5',
            '--max-reflection-rejections','5','--prompt-format','plain'])

    def test_cli_and_metadata_preserve_selected_stopping_rules(self):
        settings=self.settings()
        self.assertEqual(settings.prompt_format,'plain')
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v22_plain_continuation_1200')
        self.assertEqual((settings.max_terminal_failures,settings.max_reflection_rejections),(5,5))
        self.assertIsNone(settings.as_dict()['model_request_limit'])
        self.assertIsNone(settings.max_recoveries)
        with self.assertRaises(ValueError):replace(settings,prompt_format='guessed')

    def test_transcript_matches_native_action_observation_format_for_all_datasets(self):
        for dataset in ('game24','alfworld','webshop'):
            first=initial_actor_prompt(dataset,'LITERAL DEMONSTRATIONS',dict(observation='TASK'))
            second=append_actor_step(dataset,first,'literal action','literal observation')
            cue={'webshop':'\n\nAction:','alfworld':'\n>','game24':'\nAct 2>'}[dataset]
            messages=[dict(role='user',content=first),dict(role='assistant',content='literal action'),
                dict(role='user',content='literal observation'+cue)]
            saved=deepcopy(messages)
            self.assertEqual(render_plain_continuation(messages,'dialogue'),second)
            self.assertEqual(messages,saved)

    def test_every_model_and_phase_bypass_chat_template_and_keep_all_content(self):
        class NoChatTokenizer:
            def apply_chat_template(self,*args,**kwargs):
                raise AssertionError('plain delivery must not invoke a chat template')
        for model_type in ('qwen2','mistral','llama'):
            for purpose in ('action','recovery'):
                phase=('recovery-command' if purpose=='recovery' else 'native-action-or-rewind-command')
                messages=[dict(role='system',content=author.instruction('lines')),
                    dict(role='system',content='Backtrack memory notebook EXACT NOTE'),
                    dict(role='system',content=RUNTIME_PREFIX+'EXACT ALLOWANCE\nOutput phase: '+phase+'.'),
                    dict(role='user',content='EXACT DEMONSTRATIONS\nTASK\n\nAction:'),
                    dict(role='assistant',content='backtrack_candidates[actual observed failure]'),
                    dict(role='user',content='EXACT CANDIDATE RECEIPT\n\nAction:')]
                original=deepcopy(messages)
                prompt,encoded,policy=render_resident_request(messages,purpose,NoChatTokenizer(),
                    model_type,'dialogue','plain')
                self.assertIsNone(encoded)
                self.assertTrue(policy.startswith('native_act_only_plain_'))
                for text in (author.instruction('lines'),'Backtrack memory notebook EXACT NOTE',
                        'EXACT DEMONSTRATIONS','TASK','backtrack_candidates[actual observed failure]',
                        'EXACT CANDIDATE RECEIPT'):
                    self.assertIn(text,prompt)
                self.assertEqual(prompt.count(RUNTIME_PREFIX),1)
                self.assertTrue(prompt.endswith('Recovery command:' if purpose=='recovery' else 'Action:'))
                self.assertEqual(messages,original)

    def test_restore_uses_exact_author_prefix_and_notebook_in_plain_text(self):
        settings=self.settings()
        note='Inspect the counter and choose good instead of the observed bad transition.'
        model=ScriptedModel(['bad','backtrack_candidates[Observed failure]',
            'backtrack_commit[rec_000001]\n'+note,'good'])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,settings,directory,'plain_restore').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            restored=next(e for e in events if e['event']=='checkpoint_restored')
            saved=json.loads(Path(restored['effective_override_input_file']).read_text())
            actual=render_plain_continuation(without_runtime_allowance(model.requests[-1]),'dialogue')
            self.assertEqual(actual,render_plain_continuation(without_runtime_allowance(saved['messages']),'dialogue'))
            self.assertIn(note,actual)
            self.assertNotIn('Observed failure',actual)
        self.assertTrue(result['success'])
        self.assertEqual((result['cost']['model_requests'],result['completed_recoveries']),(4,1))

    def test_six_candidate_requests_do_not_enable_unapproved_no_progress_stop(self):
        responses=['bad']+['backtrack_candidates[Recheck observed failure]']*6+[
            'backtrack_commit[rec_000001]\nInspect the state and choose good.','good']
        model=ScriptedModel(responses)
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,self.settings(),directory,'six_candidates').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],9)
        self.assertEqual(result['consecutive_reflection_rejections'],0)

    def test_plain_openai_uses_text_completion_payload_and_records_actual_prompt(self):
        received=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.path,body))
                data=json.dumps(dict(choices=[dict(text='good',finish_reason='stop')],
                    usage=dict(prompt_tokens=31,completion_tokens=1))).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever);worker.start()
        try:
            args=SimpleNamespace(endpoint='http://127.0.0.1:'+str(server.server_port)+'/v1',api_key_env=None)
            settings=self.settings();model=OpenAIModel(args,settings,'plain_http')
            messages=[dict(role='user',content='EXACT DEMOS\nTASK\n\nAction:')]
            submitted=[]
            reply=model.generate(messages,100,5,lambda:submitted.append(True),purpose='action')
            self.assertEqual(received[0][0],'/v1/completions')
            self.assertEqual(received[0][1]['prompt'],messages[0]['content'])
            self.assertNotIn('messages',received[0][1])
            self.assertEqual(received[0][1]['temperature'],.1)
            self.assertEqual(received[0][1]['max_tokens'],100)
            self.assertEqual(reply.raw['client_prompt'],messages[0]['content'])
            self.assertIsNone(reply.raw['client_model_messages'])
            self.assertEqual(reply.text,'good')
            self.assertEqual(submitted,[True])
        finally:
            server.shutdown();worker.join();server.server_close()

    def test_broken_history_is_rejected_without_reordering_or_completion(self):
        with self.assertRaises(ValueError):
            render_plain_continuation([dict(role='user',content='TASK\n\nAction:'),
                dict(role='user',content='SECOND TASK\n\nAction:')],'dialogue')
        with self.assertRaises(ValueError):
            render_plain_continuation([dict(role='user',content='MISSING CUE'),
                dict(role='assistant',content='raw'),dict(role='user',content='receipt\n\nAction:')],'dialogue')


if __name__=='__main__':unittest.main()
