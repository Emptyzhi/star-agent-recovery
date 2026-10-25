"""Compact adapter text preserves evidence, author memory and controller policy."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_adaptation import AdaptationTests,CounterEnvironment,ScriptedModel
from test_notebook_limits import LONG_NOTE
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.native_protocol import RUNTIME_PREFIX,render_plain_continuation
from agentrewind_bench.runtime_prompt import (PROTOCOL_INSTRUCTION,compact_command_instruction,
    compact_runtime_instruction)


class CompactRuntimeTests(unittest.TestCase):
    def settings(self):
        return AdaptationTests().config(extra=['--prompt-format','plain','--runtime-prompt','compact',
            '--history-format','dialogue','--recovery-format','lines','--candidate-history','once',
            '--recovery-patience','2','--max-recoveries','unlimited','--notebook-item-char-limit','900',
            '--rejected-history-keep','2','--max-terminal-failures','5','--action-budget-policy','trajectory'])

    def run_case(self,responses,settings=None):
        model=ScriptedModel(responses);env=CounterEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(env,model,settings or self.settings(),directory,'compact_runtime').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            memory=[]
            for path in (Path(directory)/'author_traces').glob('*.backtrack_memory.jsonl'):
                memory.extend(json.loads(line) for line in path.read_text().splitlines())
        return model,result,events,memory

    def test_cli_records_compact_runtime_and_original_900_character_notebook(self):
        settings=self.settings()
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v25_compact_runtime_1200')
        self.assertEqual(settings.as_dict()['notebook_limits'],dict(items=20,item_characters=900,total_characters=8000))
        self.assertEqual(settings.recovery_patience,2)
        self.assertEqual((settings.actor_output_tokens,settings.max_output_tokens),(100,1200))
        self.assertIsNone(settings.as_dict()['model_request_limit'])
        with self.assertRaises(ValueError):replace(settings,runtime_prompt='guessed')
        with self.assertRaises(ValueError):replace(settings,prompt_format='native-chat')

    def test_observation_is_not_duplicated_but_status_and_all_public_state_remain(self):
        for extra in ({},{'state':{'numbers':['2','3'],'done':False}},
                      {'available_actions':['look','take apple from table']}):
            receipt=dict(observation='LITERAL NATIVE OBSERVATION',done=False,reward=None,
                action_accepted=False,task_success=False,**extra)
            text=compact_runtime_instruction(tools=author.tools()[:2],remaining=14,recoveries=None,
                blocked=False,candidates=False,paused=False,public_receipt=receipt,output_limit=1200)
            self.assertNotIn('LITERAL NATIVE OBSERVATION',text)
            status=json.loads(text.split('Current receipt: ',1)[1])
            self.assertEqual(status,{k:v for k,v in receipt.items() if k!='observation'})
            self.assertNotIn('Serialization example',text)
            self.assertNotIn('backtrack_commit[',text)
            self.assertIn('backtrack_candidates[reason]',text)

    def test_combined_commit_format_exists_only_when_current_tool_is_available(self):
        text=compact_command_instruction(author.tools(),1200)
        self.assertIn('backtrack_commit[<record_uid>]\n<complete memory_summary>',text)
        self.assertIn('header alone on its first line',text)
        self.assertIn('author memory fields',text)
        self.assertNotIn('"properties"',text)
        self.assertNotIn('rec_000001',text)

    def test_author_guidance_and_native_transcript_are_literal_and_definition_is_single(self):
        model,result,events,_=self.run_case(['bad','good'])
        self.assertTrue(result['success'])
        for request in model.requests:
            prompt=render_plain_continuation(request,'dialogue')
            self.assertEqual(prompt.count(PROTOCOL_INSTRUCTION),1)
            self.assertIn(author.instruction('lines'),prompt)
            self.assertEqual(prompt.count('Counter value: 1'),0 if request is model.requests[0] else 1)
            self.assertEqual(prompt.count(RUNTIME_PREFIX),1)
            self.assertNotIn('Current observation:',prompt)
            self.assertNotIn('Serialization example',prompt)

    def test_900_notebook_uses_original_override_and_keeps_full_raw_memory(self):
        model,result,events,memory=self.run_case(['bad','backtrack_candidates[Observed failure]',
            'backtrack_commit[rec_000001]\n'+LONG_NOTE,'good'])
        self.assertTrue(result['success'])
        restore=next(e for e in events if e['event']=='checkpoint_restored')
        self.assertEqual(restore['original_override_input_file'],restore['effective_override_input_file'])
        self.assertEqual(restore['notebook_render_config'],dict(items=20,item_char_limit=900,notebook_char_limit=8000))
        self.assertEqual(memory[0]['memory_summary'],LONG_NOTE)
        notebook=next(m['content'] for m in model.requests[-1] if author.is_backtrack_memory_message(m))
        self.assertEqual(notebook,author.render_backtrack_memory_notebook(memory,limit=20,
            item_char_limit=900,notebook_char_limit=8000,context_policy='rollback',workspace_policy='rollback',
            target_record_uid=restore['target'],context_source_record_uid=restore['context_source']))
        self.assertNotIn('corrected_strategy',notebook)
        self.assertIn('corrected_strategy',LONG_NOTE)

    def test_actor_resumption_and_reentry_costs_match_full_prompt_variant(self):
        bad='backtrack_commit[rec_000001]\nThe task is not feasible.'
        replies=['bad','backtrack_candidates[Observed failure]',bad,bad,'bad',
            'backtrack_candidates[Observed failure]',
            'backtrack_commit[rec_000001]\nInspect the counter and choose good.','good']
        compact=self.run_case(replies)
        full=self.run_case(replies,replace(self.settings(),runtime_prompt='full'))
        for key in ('status','success','executed_actions','completed_recoveries',
                    'current_trajectory_actions','consecutive_recovery_no_progress','recovery_paused'):
            self.assertEqual(compact[1][key],full[1][key])
        self.assertEqual(compact[1]['cost']['model_requests'],8)
        self.assertEqual(compact[1]['cost']['model_requests'],full[1]['cost']['model_requests'])
        for a,b in zip(compact[0].requests,full[0].requests):
            self.assertLess(len(render_plain_continuation(a,'dialogue')),len(render_plain_continuation(b,'dialogue')))
        paused=[e for e in compact[2] if e['event']=='model_input_prepared' and e['recovery_policy_state']['paused']]
        self.assertTrue(paused)
        text=next(m['content'] for m in paused[0]['messages'] if m['content'].startswith(RUNTIME_PREFIX))
        self.assertIn('overrides earlier commit-only',text)
        self.assertIn('An environment-accepted action reopens recovery',text)


if __name__=='__main__':unittest.main()
