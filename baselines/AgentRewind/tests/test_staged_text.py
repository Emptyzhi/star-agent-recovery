from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_adaptation import CounterEnvironment, ScriptedModel
from test_sectioned_interface import SectionedInterfaceTests
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.framing import COMMIT_END, decode_sectioned_reply, sectioned_initial_prompt
from agentrewind_bench.models import render_resident_request
from agentrewind_bench.native_protocol import RUNTIME_PREFIX
from agentrewind_bench.runtime_prompt import compact_runtime_instruction
from agentrewind_bench.staged_prompt import ACTION_GUIDANCE, COMPACT_MEMORY_GUIDANCE


def render(messages,purpose):
    return render_resident_request(messages,purpose,None,None,'dialogue','plain',
        'once','sectioned','staged-compact')[0]


class StagedTextTests(unittest.TestCase):
    def test_all_datasets_use_native_action_cues_and_text_recovery(self):
        for dataset,cue in (('game24','Act 1>'),('alfworld','>'),('webshop','Action:')):
            with self.subTest(dataset=dataset):
                original=author.instruction('lines',commit_boundary=COMMIT_END)
                messages=[dict(role='system',content=original),dict(role='user',content=
                    sectioned_initial_prompt(dataset,'ORIGINAL DEMOS',dict(observation='CURRENT TASK')))]
                action=compact_runtime_instruction(tools=author.tools()[:2],remaining=5,
                    recoveries=None,blocked=False,candidates=False,paused=False,
                    public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
                payload=[*messages,dict(role='system',content=action)];before=deepcopy(payload)
                text=render(payload,'action')
                self.assertIn(ACTION_GUIDANCE,text)
                self.assertNotIn(original,text)
                self.assertTrue(text.endswith(cue))
                self.assertNotIn('Tool call JSON:',text)
                self.assertNotIn('Output one JSON',text)
                self.assertEqual(payload,before)
                recovery=compact_runtime_instruction(tools=author.tools()[1:],remaining=0,
                    recoveries=None,blocked=True,candidates=True,paused=False,
                    public_receipt=dict(done=False,reward=None),output_limit=1200,commit_end=COMMIT_END)
                text=render([*messages,dict(role='system',content=recovery)],'recovery')
                self.assertIn(original,text)
                self.assertIn(COMPACT_MEMORY_GUIDANCE,text)
                self.assertIn('backtrack_commit[<record_uid>]',text)
                self.assertTrue(text.endswith('Recovery command:'))

    def test_literal_combined_commit_restore_and_native_action(self):
        settings=replace(SectionedInterfaceTests().settings(),recovery_prompt_policy='staged-compact',
            initial_actions=1,recovery_actions=1)
        note='corrected_strategy: execute good; do_not_repeat: execute bad; failed_check: Counter value: 1'
        model=ScriptedModel(['bad','backtrack_candidates[Observed failed attempt]',
            'backtrack_commit[rec_000001]\n'+note+'\n'+COMMIT_END,'good'])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,settings,directory,'staged_text').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            restore=next(e for e in events if e['event']=='checkpoint_restored')
            memory=author.load_backtrack_memory(Path(directory)/'author_traces',restore['trace'])
            self.assertEqual(memory[-1]['memory_summary'],note)
            self.assertEqual(restore['original_override_input_file'],restore['effective_override_input_file'])
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],4)
        self.assertEqual(result['completed_recoveries'],1)
        prepared=[e for e in events if e['event']=='model_input_prepared']
        self.assertEqual([e['output_limit'] for e in prepared],[100,100,1200,100])
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v35_staged_text_tools_1200')
        self.assertIn('corrected_strategy: execute good',render(model.requests[-1],'action'))

    def test_whole_recovery_response_is_checked_without_repair(self):
        complete='backtrack_commit[rec_000001]\ncorrected_strategy: inspect\n'+COMMIT_END
        self.assertEqual(decode_sectioned_reply(complete,whole_response=True)[1]['memory_summary'],
            'corrected_strategy: inspect')
        for invalid in (complete+'\nObservation: invented',complete.removesuffix(COMMIT_END),
                        'backtrack_candidates[reason]\nbacktrack_commit[rec_000001]'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                decode_sectioned_reply(invalid,whole_response=True)


if __name__=='__main__':unittest.main()
