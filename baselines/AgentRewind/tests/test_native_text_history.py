from copy import deepcopy
from dataclasses import replace
import unittest
from agentrewind_bench import author
from agentrewind_bench.framing import COMMIT_END,sectioned_initial_prompt
from agentrewind_bench.models import render_resident_request
from agentrewind_bench.native_protocol import ACTION_BOUNDARY,append_actor_step,append_dialogue_feedback,initial_actor_prompt
import re
from agentrewind_bench.runtime_prompt import compact_runtime_instruction
from agentrewind_bench.staged_prompt import ACTION_GUIDANCE,COMPACT_MEMORY_GUIDANCE
from agentrewind_bench.native_history import COMMIT_LAYOUT,PAUSED_ACTION_CUE
from test_sectioned_interface import SectionedInterfaceTests


class NativeTextHistoryTests(unittest.TestCase):
    def test_native_transcript_matches_each_dataset_and_keeps_feedback_adjacent(self):
        for dataset in ('webshop','game24','alfworld'):
            with self.subTest(dataset=dataset):
                demo='ORIGINAL DEMONSTRATION\n'
                current=dict(observation='CURRENT TASK')
                messages=[dict(role='system',content=author.instruction('lines',commit_boundary=COMMIT_END)),
                    dict(role='user',content=sectioned_initial_prompt(dataset,demo,current))]
                messages=append_dialogue_feedback(messages,'literal command','ACTUAL OBSERVATION',dataset,2)
                allowance=compact_runtime_instruction(tools=author.tools()[:2],remaining=14,
                    recoveries=None,blocked=False,candidates=False,paused=False,
                    public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
                messages.append(dict(role='system',content=allowance));before=deepcopy(messages)
                prompt,encoded,_=render_resident_request(messages,'action',None,None,
                    'dialogue','plain','once','sectioned','staged-compact','native')
                expected=append_actor_step(dataset,initial_actor_prompt(dataset,demo,current),
                    'literal command','ACTUAL OBSERVATION')
                self.assertTrue(prompt.endswith(expected))
                self.assertLess(prompt.index(allowance),prompt.index(demo))
                self.assertIn(ACTION_GUIDANCE,prompt)
                self.assertNotIn('[RECORDED_AGENT_REPLY_BEGIN]',prompt)
                self.assertNotIn('[OBSERVED_FEEDBACK_BEGIN]',prompt)
                self.assertNotIn('[CURRENT_CONTROLLER_BEGIN]',prompt)
                self.assertNotIn(COMMIT_LAYOUT,prompt)
                self.assertIsNone(encoded);self.assertEqual(messages,before)

    def test_recovery_keeps_author_guidance_and_literal_notebook(self):
        original=author.instruction('lines',commit_boundary=COMMIT_END)
        allowance=compact_runtime_instruction(tools=author.tools()[1:],remaining=0,
            recoveries=None,blocked=True,candidates=True,paused=False,
            public_receipt=dict(done=True,reward=.5),output_limit=1200,commit_end=COMMIT_END)
        messages=[dict(role='system',content=original),dict(role='system',content='LITERAL NOTEBOOK'),
            dict(role='user',content=sectioned_initial_prompt('webshop','DEMO',dict(observation='TASK'))),
            dict(role='system',content=allowance)]
        text,_,_=render_resident_request(messages,'recovery',None,None,'dialogue','plain',
            'once','sectioned','staged-compact','native')
        self.assertIn(original,text);self.assertIn(COMPACT_MEMORY_GUIDANCE,text)
        self.assertIn('LITERAL NOTEBOOK',text);self.assertTrue(text.endswith('Recovery command:'))
        self.assertGreater(text.index(allowance),text.index('TASK'))
        self.assertTrue(text.endswith(COMMIT_LAYOUT+'\nRecovery command:'))

    def test_deterministic_feedback_on_initial_turn_is_preserved(self):
        messages=[dict(role='system',content=author.instruction('lines',commit_boundary=COMMIT_END)),
            dict(role='user',content=sectioned_initial_prompt('webshop','DEMO',dict(observation='TASK')))]
        messages=append_dialogue_feedback(messages,None,'REAL CANDIDATE RECEIPT','webshop',1)
        allowance=compact_runtime_instruction(tools=author.tools()[1:],remaining=0,
            recoveries=None,blocked=True,candidates=True,paused=False,
            public_receipt=dict(done=True,reward=0),output_limit=1200,commit_end=COMMIT_END)
        messages.append(dict(role='system',content=allowance))
        text,_,_=render_resident_request(messages,'recovery',None,None,'dialogue','plain',
            'once','sectioned','staged-compact','native')
        self.assertIn('REAL CANDIDATE RECEIPT',text)

    def test_protocol_metadata_and_invalid_combinations(self):
        base=replace(SectionedInterfaceTests().settings(),backend='resident',
            recovery_prompt_policy='staged-compact',text_output_boundary='command')
        native=replace(base,text_history_format='native')
        self.assertEqual(native.as_dict()['adaptation_protocol'],'native_act_only_v44_game24_initial_separator_1200')
        for changes in (dict(text_output_boundary='eos'),dict(interface_format='json-tools'),dict(text_history_format='guess')):
            with self.assertRaises(ValueError):replace(native,**changes)

    def test_paused_actor_current_commands_follow_historical_recovery_errors(self):
        messages=[dict(role='system',content=author.instruction('lines',commit_boundary=COMMIT_END)),
            dict(role='user',content=sectioned_initial_prompt('webshop','DEMO',dict(observation='TASK')))]
        for _ in range(3):
            messages=append_dialogue_feedback(messages,'backtrack_commit[rec_000001]\nunfinished note',
                'Protocol error: missing delimiter','webshop',1)
        allowance=compact_runtime_instruction(tools=author.tools()[:1],remaining=12,
            recoveries=None,blocked=False,candidates=False,paused=True,
            public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
        messages.append(dict(role='system',content=allowance))
        text,_,_=render_resident_request(messages,'action',None,None,'dialogue','plain',
            'once','sectioned','staged-compact','native')
        self.assertTrue(text.endswith(PAUSED_ACTION_CUE+'\n\nAction:'))
        self.assertGreater(text.index(PAUSED_ACTION_CUE),text.rindex('Protocol error: missing delimiter'))
        self.assertEqual(text.count(allowance),1)
        self.assertEqual(text.count('unfinished note'),3)
        self.assertNotIn(COMPACT_MEMORY_GUIDANCE,text)
        self.assertNotIn(COMMIT_LAYOUT,text)

    def test_commit_layout_preserves_all_fields_without_task_specific_examples(self):
        from agentrewind_bench.native_protocol import MEMORY_FIELDS
        fields=re.findall(r'^([a-z0-9_]+): <value>$',COMMIT_LAYOUT,re.MULTILINE)
        self.assertEqual(set(fields),set(MEMORY_FIELDS))
        self.assertEqual(len(fields),15)
        self.assertTrue(COMMIT_LAYOUT.endswith(COMMIT_END))
        self.assertNotIn('rec_000001',COMMIT_LAYOUT)
        self.assertNotIn('search[',COMMIT_LAYOUT)
        self.assertNotIn('click[',COMMIT_LAYOUT)

    def test_candidate_only_decision_has_no_commit_layout(self):
        allowance=compact_runtime_instruction(tools=author.tools()[1:2],remaining=0,
            recoveries=None,blocked=True,candidates=False,paused=False,
            public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
        messages=[dict(role='system',content=author.instruction('lines',commit_boundary=COMMIT_END)),
            dict(role='user',content=sectioned_initial_prompt('webshop','DEMO',dict(observation='TASK'))),
            dict(role='system',content=allowance)]
        before=deepcopy(messages)
        text,_,_=render_resident_request(messages,'recovery',None,None,'dialogue','plain',
            'once','sectioned','staged-compact','native')
        self.assertNotIn(COMMIT_LAYOUT,text)
        self.assertEqual(messages,before)

    def test_game24_initial_and_paused_actor_include_native_command_separator(self):
        for paused in (False,True):
            with self.subTest(paused=paused):
                messages=[dict(role='system',content=author.instruction('lines',commit_boundary=COMMIT_END)),
                    dict(role='system',content='LITERAL NOTEBOOK'),
                    dict(role='user',content=sectioned_initial_prompt('game24','ORIGINAL EXAMPLES\n',
                        dict(observation='Input: 1 1 4 6')))]
                allowance=compact_runtime_instruction(tools=author.tools()[:1] if paused else author.tools()[:2],
                    remaining=12,recoveries=None,blocked=False,candidates=False,paused=paused,
                    public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
                messages.append(dict(role='system',content=allowance));before=deepcopy(messages)
                text,encoded,_=render_resident_request(messages,'action',None,None,'dialogue','plain',
                    'once','sectioned','staged-compact','native')
                self.assertTrue(text.endswith('\nAct 1> '))
                self.assertFalse(text.endswith('\nAct 1>  '))
                self.assertIn('ORIGINAL EXAMPLES\n',text)
                self.assertIn('LITERAL NOTEBOOK',text)
                self.assertEqual(messages,before)
                self.assertIsNone(encoded)

    def test_game24_recovery_keeps_its_existing_command_boundary(self):
        allowance=compact_runtime_instruction(tools=author.tools()[1:],remaining=0,
            recoveries=None,blocked=True,candidates=True,paused=False,
            public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
        messages=[dict(role='system',content=author.instruction('lines',commit_boundary=COMMIT_END)),
            dict(role='user',content=sectioned_initial_prompt('game24','ORIGINAL EXAMPLES\n',
                dict(observation='Input: 1 1 4 6'))),dict(role='system',content=allowance)]
        before=deepcopy(messages)
        text,_,_=render_resident_request(messages,'recovery',None,None,'dialogue','plain',
            'once','sectioned','staged-compact','native')
        self.assertTrue(text.endswith(COMMIT_LAYOUT+'\nRecovery command:'))
        self.assertEqual(messages,before)

    def test_game24_separator_protocol_is_shared_across_models(self):
        base=replace(SectionedInterfaceTests().settings(),dataset='game24',backend='resident',
            recovery_prompt_policy='staged-compact',text_output_boundary='command',text_history_format='native')
        for model in ('qwen','mistral','llama'):
            self.assertEqual(replace(base,model=model).as_dict()['adaptation_protocol'],
                'native_act_only_v44_game24_initial_separator_1200')

    def test_game24_continuation_and_answer_cues_retain_their_original_frame(self):
        for next_action in (2,4,12):
            with self.subTest(next_action=next_action):
                messages=[dict(role='system',content=author.instruction('lines',commit_boundary=COMMIT_END)),
                    dict(role='user',content=sectioned_initial_prompt('game24','ORIGINAL EXAMPLES\n',
                        dict(observation='Input: 6 6 6 6')))]
                messages=append_dialogue_feedback(messages,'6 + 6 = 12\n\n','numbers left: 6 6 12',
                    'game24',next_action)
                allowance=compact_runtime_instruction(tools=author.tools()[:2],remaining=10,
                    recoveries=None,blocked=False,candidates=False,paused=False,
                    public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
                messages.append(dict(role='system',content=allowance));before=deepcopy(messages)
                text,_,_=render_resident_request(messages,'action',None,None,'dialogue','plain',
                    'once','sectioned','staged-compact','native')
                self.assertTrue(text.endswith(f'\nAct {next_action}>'))
                self.assertFalse(text.endswith(' '))
                self.assertIn('6 + 6 = 12\n\n',text)
                self.assertEqual(messages,before)


if __name__=='__main__':unittest.main()
