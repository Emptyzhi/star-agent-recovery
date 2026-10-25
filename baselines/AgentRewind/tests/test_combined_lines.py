"""Combined unescaped argument transport preserves the author operation."""
import json
from pathlib import Path
import tempfile
import unittest

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment,ScriptedModel
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.native_protocol import decode_line,available_command_instruction,boundary_instruction


class CombinedLinesTests(unittest.TestCase):
    def test_quoted_multiline_memory_is_one_literal_argument_without_escaping(self):
        note='failed_check: "Invalid action!"\nroot_cause: uncertain\\path\ncorrected_strategy: click[Back to Search].\n'
        tool,arguments=decode_line('backtrack_commit[rec_000001]\n'+note,'lines')
        self.assertEqual(tool,'backtrack_commit')
        self.assertEqual(arguments,dict(record_uid='rec_000001',memory_summary=note))
        self.assertEqual(decode_line('backtrack_candidates[Invalid action!]','lines'),
            ('backtrack_candidates','Invalid action!'))

    def test_missing_arguments_or_wrong_envelopes_are_not_repaired(self):
        invalid=['backtrack_commit[rec_000001]','backtrack_commit[rec_000001]\n  ',
            'backtrack_commit[]\nNote','backtrack_commit["rec_000001"]\nNote',
            'backtrack_commit(record_uid="rec_000001", memory_summary="Note")',
            'backtrack_commit[{"record_uid":"rec_000001","memory_summary":"Note"}]']
        for text in invalid:
            with self.subTest(text=text),self.assertRaises(ValueError):decode_line(text,'lines')

    def test_controller_submits_target_and_note_together_and_charges_rejected_notes(self):
        settings=fixtures.AdaptationTests().config(extra=['--history-format','dialogue','--recovery-format','lines'])
        model=ScriptedModel(['bad','backtrack_candidates[Observed failure]',
            'backtrack_commit[rec_000001]\nThe task is not feasible.',
            'backtrack_commit[rec_000001]\nCheck the prior incorrect transition, then choose good.','good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,settings,tmp,'combined_lines').run()
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['executed_actions'],2)
        self.assertEqual(sum(e['event']=='author_tool_rejected' for e in events),1)
        self.assertEqual(sum(e['event']=='recovery_memory_generated' for e in events),2)
        self.assertTrue(all(e['output_limit']==1200 for e in events
            if e['event']=='model_input_prepared' and e['purpose']=='recovery'))

    def test_author_guidance_and_runtime_forms_match_combined_transport(self):
        original=author.backtrack_tool_instructions(selection_policy='agent-direct',context_policy='rollback',workspace_policy='rollback')
        self.assertTrue(author.instruction('lines').endswith(original))
        instruction=available_command_instruction([t for t in author.tools() if t['name']=='backtrack_commit'],1200,'lines')
        self.assertIn('backtrack_commit[<record_uid>]\n<complete recovery note>',instruction)
        self.assertIn('passed verbatim',instruction)
        self.assertNotIn('JSON argument object inside brackets',author.instruction('lines'))
        self.assertIn('backtrack_candidates[reason]',boundary_instruction(remaining=0,recoveries=None,
            blocked=True,candidates=False,recovery_format='lines'))


if __name__=='__main__':unittest.main()
