"""Role fidelity, author rollback and charged errors in actual-turn history."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment,ScriptedModel,commit
from agentrewind_bench.controller import Controller
from agentrewind_bench.models import render_resident_request
from agentrewind_bench.native_protocol import (append_dialogue_feedback,
    decision_chat_messages,initial_actor_prompt,without_runtime_allowance,RUNTIME_PREFIX)


class DialogueTests(unittest.TestCase):
    def run_case(self,responses,*flags):
        model=ScriptedModel(responses)
        settings=fixtures.AdaptationTests().config(extra=['--history-format','dialogue',*flags])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,settings,tmp,'dialogue').run()
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        return model,result,events

    def test_exact_model_response_is_assistant_turn_and_native_feedback_is_user_turn(self):
        model,result,_=self.run_case(['bad','good'])
        dialogue=without_runtime_allowance(model.requests[1])
        self.assertEqual(dialogue[-2],dict(role='assistant',content='bad'))
        self.assertEqual(dialogue[-1],dict(role='user',content='Counter value: 1\nAct 2>'))
        self.assertEqual(result['cost']['model_requests'],2)
        self.assertEqual(result['executed_actions'],2)

    def test_invalid_submission_retains_raw_response_once_and_returns_explicit_error(self):
        raw='backtrack_commit[{"record_uid":"rec_000001","memory_summary":"UNCLOSED'
        model,result,_=self.run_case([raw,'good'])
        self.assertEqual(model.requests[1][-2],dict(role='assistant',content=raw))
        self.assertIn('Protocol error:',model.requests[1][-1]['content'])
        self.assertNotIn(raw,model.requests[1][-1]['content'])
        self.assertEqual(result['cost']['model_requests'],2)
        self.assertEqual(result['executed_actions'],1)
        self.assertEqual(result['protocol_rejections'][0]['raw_text'],raw)

    def test_author_memory_error_follows_its_actual_assistant_commit_and_is_charged(self):
        rejected=commit('rec_000001','The task is not feasible.')
        model,result,events=self.run_case(['bad','backtrack_candidates[observed failure]',
            rejected,commit('rec_000001','Use good after inspecting the failed transition.'),'good'])
        request=model.requests[3]
        self.assertEqual(request[-2],dict(role='assistant',content=rejected))
        receipt=json.loads(request[-1]['content'].rsplit('\nAct ',1)[0])
        self.assertTrue(receipt['result']['rejected'])
        self.assertIn('memory_summary rejected:',receipt['result']['error'])
        self.assertNotIn('arguments',receipt)
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(sum(e['event']=='author_tool_rejected' for e in events),1)
        self.assertEqual(result['completed_recoveries'],1)

    def test_author_restores_exact_selected_dialogue_prefix_and_drops_future_errors(self):
        model,result,_=self.run_case(['bad','backtrack_candidates[failure]',
            'backtrack_commit[INVALID_SUFFIX]',commit('rec_000001','Choose good based on the observed incorrect state.'),'good'])
        restored=without_runtime_allowance(model.requests[-1])
        self.assertNotIn('INVALID_SUFFIX',json.dumps(restored))
        notebooks=[m for m in restored if m['content'].startswith('Backtrack memory notebook')]
        self.assertEqual(len(notebooks),1)
        self.assertEqual([m for m in restored if m not in notebooks],without_runtime_allowance(model.requests[0]))
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertTrue(result['success'])

    def test_runtime_is_delivered_only_to_latest_observation_and_all_models_render_turns(self):
        model,_,_=self.run_case(['bad','good'])
        original=deepcopy(model.requests[1])
        encoded=decision_chat_messages(original,'dialogue')
        self.assertNotIn(RUNTIME_PREFIX,encoded[1]['content'])
        self.assertIn(RUNTIME_PREFIX,encoded[-1]['content'])
        for model_type in ('qwen2','mistral','llama'):
            seen=[]
            tokenizer=SimpleNamespace(apply_chat_template=lambda messages,**kwargs:
                seen.append(deepcopy(messages)) or 'rendered native chat')
            prompt,wire,policy=render_resident_request(original,'action',tokenizer,model_type,'dialogue')
            self.assertEqual(prompt,'rendered native chat')
            self.assertEqual(wire,seen[0])
            turns=[m['role'] for m in wire if m['role']!='system']
            self.assertEqual(turns,['user','assistant','user'])
            self.assertEqual(wire[-2]['content'],'bad')
        self.assertEqual(original,model.requests[1])

    def test_native_action_boundaries_are_exact_for_each_dataset(self):
        for dataset,suffix in [('webshop','\n\nAction:'),('alfworld','\n>'),('game24','\nAct 2>')]:
            messages=[dict(role='user',content=initial_actor_prompt(dataset,'DEMOS',{'observation':'TASK'}))]
            saved=deepcopy(messages)
            result=append_dialogue_feedback(messages,'literal action','literal receipt',dataset,2)
            self.assertEqual(result[-2],dict(role='assistant',content='literal action'))
            self.assertEqual(result[-1],dict(role='user',content='literal receipt'+suffix))
            self.assertEqual(messages,saved)

    def test_deterministic_terminal_listing_does_not_invent_model_response(self):
        messages=[dict(role='user',content='TASK\n\nAction:'),dict(role='assistant',content='click[Buy Now]'),
                  dict(role='user',content='Score 0.5\n\nAction:')]
        result=append_dialogue_feedback(messages,None,'AUTHOR CANDIDATES','webshop',4)
        self.assertEqual(len(result),len(messages))
        self.assertEqual(result[-2],messages[-2])
        self.assertIn('AUTHOR CANDIDATES',result[-1]['content'])
        self.assertNotIn('AUTHOR CANDIDATES',messages[-1]['content'])

    def test_broken_dialogue_order_is_rejected_without_role_repair(self):
        messages=[dict(role='system',content=RUNTIME_PREFIX+'Current allowance'),
                  dict(role='user',content='TASK\nAct 1>'),dict(role='user',content='OTHER\nAct 1>')]
        with self.assertRaisesRegex(ValueError,'alternate'):
            decision_chat_messages(messages,'dialogue')


if __name__=='__main__':unittest.main()
