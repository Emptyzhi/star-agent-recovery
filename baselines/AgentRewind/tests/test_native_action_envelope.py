"""Declared whitespace envelope preserves literal commands, traces, and costs."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from agentrewind_bench.controller import Controller
from agentrewind_bench.framing import decode_native_action_reply
from test_adaptation import CounterEnvironment,ScriptedModel
from test_sectioned_interface import SectionedInterfaceTests


class NativeActionEnvelopeTests(unittest.TestCase):
    def test_native_commands_keep_their_literal_content(self):
        for command in ('click[Buy Now]','1 + 1 = 2','go to kitchen'):
            text='\n \t\n'+command+'\n'
            self.assertEqual(decode_native_action_reply(text),('execute_action',command))
        self.assertEqual(decode_native_action_reply('\nbacktrack_candidates[click[Buy Now] failed]\n'),
            ('backtrack_candidates','click[Buy Now] failed'))

    def test_no_prose_search_multiple_commands_or_empty_reply_repair(self):
        for text in ('\nclick[Buy Now]\nclick[Back to Search]', '\n \t\n'):
            with self.assertRaises(ValueError):decode_native_action_reply(text)
        self.assertEqual(decode_native_action_reply('I suggest click[Buy Now]\n'),
            ('execute_action','I suggest click[Buy Now]'))
        self.assertEqual(decode_native_action_reply('click[Buy Now\n'),
            ('execute_action','click[Buy Now'))

    def test_controller_preserves_raw_reply_and_charges_single_actual_request(self):
        settings=replace(SectionedInterfaceTests().settings(),backend='resident',
            recovery_prompt_policy='staged-compact',text_output_boundary='command',text_history_format='native')
        raw='\n \n good\n';model=ScriptedModel([raw])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,settings,directory,'native_envelope').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual((result['executed_actions'],result['cost']['model_requests']),(1,1))
        self.assertEqual((result['cost']['input_tokens'],result['cost']['output_tokens']),(31,7))
        self.assertEqual(next(e['text'] for e in events if e['event']=='model_request_completed'),raw)
        frame=next(e for e in events if e['event']=='native_action_frame_decoded')
        self.assertEqual(frame['framing'],'first_nonempty_line')
        self.assertEqual(frame['leading_whitespace_characters'],len(raw)-len(raw.lstrip()))
        self.assertEqual(frame['argument'],'good')


if __name__=='__main__':unittest.main()
