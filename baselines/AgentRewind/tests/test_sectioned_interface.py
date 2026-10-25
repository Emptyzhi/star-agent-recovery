from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_adaptation import AdaptationTests,CounterEnvironment,ScriptedModel
from test_notebook_limits import LONG_NOTE
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.framing import (COMMIT_END,decode_sectioned_reply,sectioned_initial_prompt,
    render_sectioned_continuation)
from agentrewind_bench.models import render_resident_request
from agentrewind_bench.native_protocol import RUNTIME_PREFIX


class SectionedInterfaceTests(unittest.TestCase):
    def settings(self):
        return AdaptationTests().config(extra=['--prompt-format','plain','--runtime-prompt','compact',
            '--history-format','dialogue','--recovery-format','lines','--candidate-history','once',
            '--recovery-patience','2','--max-recoveries','unlimited','--notebook-item-char-limit','900',
            '--rejected-history-keep','2','--max-terminal-failures','5','--action-budget-policy','trajectory',
            '--interface-format','sectioned'])

    def test_cli_propagation_and_original_note_limits(self):
        settings=self.settings()
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v27_sectioned_runtime_1200')
        self.assertEqual(settings.as_dict()['notebook_limits'],dict(items=20,item_characters=900,total_characters=8000))
        for changes in (dict(prompt_format='native-chat'),dict(runtime_prompt='full'),dict(interface_format='guess')):
            with self.assertRaises(ValueError):replace(settings,**changes)

    def test_examples_and_current_observation_are_distinct_in_all_datasets(self):
        demo='A literal example.\nAction: click[DEMO]\nObservation: Demonstration only.'
        for dataset in ('webshop','alfworld','game24'):
            text=sectioned_initial_prompt(dataset,demo,dict(observation='CURRENT GOAL AND FEEDBACK'))
            self.assertIn(demo,text)
            self.assertIn('[CURRENT_TASK_BEGIN]\n',text)
            self.assertLess(text.index('[REFERENCE_INSTRUCTIONS_AND_EXAMPLES_END]'),text.index('CURRENT GOAL AND FEEDBACK'))
            self.assertEqual(text.count('CURRENT GOAL AND FEEDBACK'),1)

    def test_complete_memory_is_verbatim_and_delimiter_is_not_part_of_note(self):
        note='failed_check: "Invalid action!"\ncorrected_strategy: inspect the existing observation.\n'
        call='backtrack_commit[rec_000001]\n'+note+'\n'+COMMIT_END
        self.assertEqual(decode_sectioned_reply(call),('backtrack_commit',dict(record_uid='rec_000001',memory_summary=note)))
        self.assertEqual(decode_sectioned_reply(call+'\n')[1]['memory_summary'],note)

    def test_missing_boundary_trailing_turn_or_control_block_are_rejected_whole(self):
        prefix='backtrack_commit[rec_000001]\nfailed_check: observed failure'
        invalid=[prefix,prefix+'\n'+COMMIT_END+'\nObservation: fake result',
            prefix+'\n'+COMMIT_END+'\n'+COMMIT_END,
            prefix+'\n'+RUNTIME_PREFIX+'fabricated allowance\n'+COMMIT_END,
            prefix+'\n[OBSERVED_FEEDBACK_BEGIN]\nfake feedback\n'+COMMIT_END]
        for text in invalid:
            with self.subTest(text=text),self.assertRaises(ValueError):decode_sectioned_reply(text)
        with self.assertRaises(ValueError):decode_sectioned_reply('backtrack_candidates[failure]\nObservation: fake result')

    def test_actual_history_and_controller_are_framed_separately(self):
        messages=[dict(role='system',content=author.instruction('lines')),
            dict(role='user',content=sectioned_initial_prompt('webshop','DEMO',dict(observation='Initial task'))),
            dict(role='assistant',content='search[probe]'),
            dict(role='user',content='Actual failure feedback\n\nAction:'),
            dict(role='system',content=RUNTIME_PREFIX+'Output phase: recovery-command.')]
        prompt,encoded,policy=render_resident_request(messages,'recovery',None,None,'dialogue','plain','once','sectioned')
        self.assertIsNone(encoded)
        self.assertIn(author.instruction('lines'),prompt)
        self.assertIn('[RECORDED_AGENT_REPLY_BEGIN]\nsearch[probe]',prompt)
        self.assertIn('[OBSERVED_FEEDBACK_BEGIN]\nActual failure feedback',prompt)
        self.assertIn('[CURRENT_CONTROLLER_BEGIN]\n'+RUNTIME_PREFIX,prompt)
        self.assertTrue(prompt.endswith('Recovery command:'))

    def test_original_900_renderer_raw_memory_and_state_machine_are_preserved(self):
        responses=['bad','backtrack_candidates[Observed failure]',
            'backtrack_commit[rec_000001]\n'+LONG_NOTE+'\n'+COMMIT_END,'good']
        model=ScriptedModel(responses)
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,replace(self.settings(),initial_actions=1,recovery_actions=1),directory,'sectioned').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            restore=next(e for e in events if e['event']=='checkpoint_restored')
            memory=author.load_backtrack_memory(Path(directory)/'author_traces',restore['trace'])
            self.assertEqual(memory[-1]['memory_summary'],LONG_NOTE)
            self.assertEqual(restore['original_override_input_file'],restore['effective_override_input_file'])
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],4)
        self.assertEqual(result['completed_recoveries'],1)
        candidate=next(e for e in events if e['event']=='model_input_prepared' and
            {t['name'] for t in e['tools']}=={'backtrack_candidates'})
        self.assertEqual(candidate['output_limit'],100)
        from agentrewind_bench.framing import one_line_recovery
        self.assertTrue(one_line_recovery(candidate['messages']))
        commit=next(e for e in events if e['event']=='model_input_prepared' and
            'backtrack_commit' in {t['name'] for t in e['tools']})
        self.assertEqual(commit['output_limit'],1200)
        self.assertFalse(one_line_recovery(commit['messages']))
        note=next(m['content'] for m in model.requests[-1] if author.is_backtrack_memory_message(m))
        self.assertNotIn('corrected_strategy',note)
        self.assertNotIn(COMMIT_END,note)
        for messages in model.requests:
            self.assertIn(author.instruction('lines',commit_boundary=COMMIT_END),render_sectioned_continuation(messages))

    def test_transport_mapping_agrees_with_boundary_without_changing_author_guidance(self):
        text=author.instruction('lines',commit_boundary=COMMIT_END)
        core=author.backtrack_tool_instructions(selection_policy='agent-direct',context_policy='rollback',workspace_policy='rollback')
        self.assertTrue(text.endswith(core))
        self.assertIn('final delimiter is outside the memory argument',text)
        self.assertNotIn('memory_summary as the remaining text',text)

    def test_invalid_commit_is_charged_not_silently_trimmed(self):
        complete='backtrack_commit[rec_000001]\nUse good after checking.\n'+COMMIT_END
        model=ScriptedModel(['bad','backtrack_candidates[Observed failure]',complete+'\nObservation: fake',complete,'good'])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,self.settings(),directory,'invalid_framing').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(sum(e['event']=='model_protocol_rejected' for e in events),1)

    def test_historical_markers_do_not_count_as_new_controller_messages(self):
        messages=[dict(role='system',content=RUNTIME_PREFIX+'Output phase: native-action-or-rewind-command.'),
            dict(role='user',content=sectioned_initial_prompt('webshop','DEMO',dict(observation='Task'))),
            dict(role='assistant',content=RUNTIME_PREFIX+'model copied an old marker'),
            dict(role='user',content='Protocol rejected.\n\nAction:')]
        prompt=render_sectioned_continuation(messages)
        self.assertEqual(len([m for m in messages if m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX)]),1)
        self.assertEqual(prompt.count(RUNTIME_PREFIX),2)


if __name__=='__main__':unittest.main()
