"""Literal function transport: no execution, output repair or guessed fields."""
import json
from pathlib import Path
import tempfile
import unittest

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment,ScriptedModel
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.native_protocol import decode_line,available_command_instruction,boundary_instruction


def commit(uid,memory):
    return 'backtrack_commit(record_uid='+repr(uid)+', memory_summary=r"""'+memory+'""")'


class LiteralCallsTests(unittest.TestCase):
    def test_unescaped_structured_memory_is_transmitted_exactly_in_one_call(self):
        note='failed_check: Invalid action!\nroot_cause: uncertain; evidence: 22" item.\ncorrected_strategy: click[Back to Search].'
        name,arguments=decode_line(commit('rec_000001',note),'python')
        self.assertEqual(name,'backtrack_commit')
        self.assertEqual(arguments,dict(record_uid='rec_000001',memory_summary=note))
        inner=json.dumps(dict(failed_check='quoted "receipt"',root_cause='uncertain'))
        self.assertEqual(decode_line(commit('rec_000001',inner),'python')[1]['memory_summary'],inner)

    def test_named_reason_and_optional_commit_reason_are_actual_arguments(self):
        self.assertEqual(decode_line('backtrack_candidates(reason="Observed failure")','python'),
            ('backtrack_candidates','Observed failure'))
        self.assertEqual(decode_line('backtrack_commit(record_uid="r", memory_summary="m", reason="why")','python')[1],
            dict(record_uid='r',memory_summary='m',reason='why'))

    def test_calls_cannot_execute_expressions_imports_unpacking_or_missing_fields(self):
        invalid=['backtrack_commit(record_uid="r")',
            'backtrack_commit(record_uid="r", memory_summary=str(1))',
            'backtrack_commit(record_uid="r", memory_summary=f"value {1}")',
            'backtrack_commit(**{"record_uid":"r","memory_summary":"m"})',
            'backtrack_commit(record_uid="r", record_uid="s", memory_summary="m")',
            'backtrack_commit("r","m")','backtrack_commit(record_uid="r", memory_summary="m", extra="x")',
            '__import__("os").system("echo forbidden")','backtrack_candidates(reason="")',
            'backtrack_commit(record_uid="r", memory_summary="unclosed)',
            'backtrack_commit[{"record_uid":"r","memory_summary":"m"}]',
            '```python\nbacktrack_commit(record_uid="r", memory_summary="m")\n```']
        for text in invalid:
            from agentrewind_bench.native_protocol import decode_python_recovery
            with self.subTest(text=text),self.assertRaises(ValueError):decode_python_recovery(text)

    def test_controller_charges_one_combined_commit_and_preserves_author_memory_rules(self):
        settings=fixtures.AdaptationTests().config(extra=['--history-format','dialogue','--recovery-format','python'])
        model=ScriptedModel(['bad','backtrack_candidates(reason="Observed failure")',
            commit('rec_000001','The task is not feasible.'),
            commit('rec_000001','Check the previous failure, then use good.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,settings,tmp,'literal').run()
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(sum(e['event']=='author_tool_rejected' for e in events),1)
        for event in events:
            if event['event']=='model_input_prepared' and event['purpose']=='recovery':
                self.assertEqual(event['output_limit'],1200)
                self.assertIn('triple-quoted',json.dumps(event['messages']))

    def test_author_reflection_instructions_are_identical_for_both_transports(self):
        original=author.backtrack_tool_instructions(selection_policy='agent-direct',context_policy='rollback',workspace_policy='rollback')
        self.assertTrue(author.instruction('python').endswith(original))
        self.assertTrue(author.instruction('bracket-json').endswith(original))
        self.assertNotIn('JSON argument object inside brackets',author.instruction('python'))

    def test_all_advertised_recovery_syntax_matches_selected_transport(self):
        listing=available_command_instruction([t for t in author.tools() if t['name']!='backtrack_commit'],1200,'python')
        self.assertIn('backtrack_candidates(reason=',listing)
        self.assertNotIn('backtrack_candidates[',listing)
        boundary=boundary_instruction(remaining=0,recoveries=None,blocked=True,candidates=False,recovery_format='python')
        self.assertNotIn('backtrack_candidates[',boundary)
        current=available_command_instruction([t for t in author.tools() if t['name']=='backtrack_commit'],1200,'python')
        self.assertIn('backtrack_commit(record_uid=',current)
        self.assertIn('memory_summary=r"""',current)
        self.assertNotIn('backtrack_commit[',current)


if __name__=='__main__':unittest.main()
