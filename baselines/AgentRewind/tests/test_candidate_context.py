"""Lossless receipt presentation, strict command transport and stopping parity."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_adaptation import AdaptationTests, CounterEnvironment, ScriptedModel
from agentrewind_bench import author
from agentrewind_bench.candidate_context import (candidate_receipts, project_candidate_history,
    expand_candidate_history, CATALOG_PREFIX, FEEDBACK_MARKER)
from agentrewind_bench.controller import Controller
from agentrewind_bench.models import render_resident_request
from agentrewind_bench.native_protocol import (available_command_instruction, decode_line,
    boundary_instruction, initial_actor_prompt)


def receipt(entries, terminal=False):
    value = dict(tool='backtrack_candidates', result=entries)
    if terminal:
        value.update(recovery_required=True, reason='Actual partial purchase: 0.75')
    return json.dumps(value, ensure_ascii=False)


class CandidateContextTests(unittest.TestCase):
    def settings(self):
        return AdaptationTests().config(extra=['--history-format','dialogue', '--prompt-format','plain',
            '--recovery-format','lines', '--candidate-history','once', '--notebook-item-char-limit','4000',
            '--rejected-history-keep','2', '--max-recoveries','unlimited',
            '--max-terminal-failures','5', '--max-reflection-rejections','5'])

    def test_exact_entry_versions_order_membership_and_raw_dialogue_survive(self):
        shared = dict(record_uid='rec_000001', tool_calls_preview=['x'*4000], created_at='literal')
        old = dict(record_uid='rec_000003', decision_hint='old evidence')
        changed = dict(record_uid='rec_000003', decision_hint='new evidence')
        messages=[dict(role='system', content=author.instruction('lines')),
            dict(role='user',content='TASK\n\nAction:'),
            dict(role='assistant',content='backtrack_candidates[actual failure]'),
            dict(role='user',content=receipt([shared,old,shared])+'\n\nAction:'),
            dict(role='assistant',content='RAW MODEL OUTPUT MUST STAY'),
            dict(role='user',content=receipt([old,shared])+'\n\nAction:'),
            dict(role='assistant',content='backtrack_candidates[actual second failure]'),
            dict(role='user',content=receipt([changed,shared])+'\n\nAction:')]
        original=deepcopy(messages)
        projected,report=project_candidate_history(messages)
        self.assertEqual(messages,original)
        self.assertEqual(expand_candidate_history(projected),original)
        self.assertEqual(report['historical_unique_entries'],1)
        self.assertEqual(report['referenced_receipts'],2)
        self.assertEqual(candidate_receipts(projected)[-1]['value'],candidate_receipts(original)[-1]['value'])
        prompt,_,_=render_resident_request(messages,'recovery',None,'qwen2','dialogue','plain','once')
        self.assertEqual(prompt.count('x'*4000),1)
        self.assertIn('old evidence',prompt)
        self.assertIn('new evidence',prompt)
        self.assertIn('RAW MODEL OUTPUT MUST STAY',prompt)

    def test_automatic_terminal_receipt_and_all_native_cues(self):
        entry=dict(record_uid='rec_000001',decision_hint='literal old decision')
        for dataset,cue in [('webshop','\n\nAction:'),('alfworld','\n>'),('game24','\nAct 2>')]:
            messages=[dict(role='user',content=initial_actor_prompt(dataset,'DEMONSTRATIONS',
                dict(observation='TASK'))),dict(role='assistant',content='actual environment action'),
                dict(role='user',content='NATIVE FEEDBACK\n\n'+FEEDBACK_MARKER+receipt([entry],True)+cue),
                dict(role='assistant',content='backtrack_candidates[actual failure]'),
                dict(role='user',content=receipt([entry])+cue)]
            projected,report=project_candidate_history(messages)
            self.assertEqual(expand_candidate_history(projected),messages)
            prompt,_,_=render_resident_request(messages,'recovery',None,'mistral','dialogue','plain','once')
            self.assertIn('NATIVE FEEDBACK',prompt)
            self.assertIn('Actual partial purchase: 0.75',prompt)
            self.assertEqual(report['referenced_receipts'],1)

    def test_one_receipt_and_assistant_fabrications_are_not_rewritten(self):
        entry=dict(record_uid='rec_000001',decision_hint='ACTUAL EVIDENCE')
        messages=[dict(role='user',content='TASK\n\nAction:'),
            dict(role='assistant',content=FEEDBACK_MARKER+receipt([entry])),
            dict(role='user',content=receipt([entry])+'\n\nAction:')]
        projected,report=project_candidate_history(messages)
        self.assertEqual(projected,messages)
        self.assertEqual(report['receipts'],1)
        self.assertFalse(any(m['content'].startswith(CATALOG_PREFIX) for m in projected))

    def test_long_identical_list_is_one_range_and_membership_is_reversible(self):
        entries=[dict(record_uid='rec_%06d'%i,decision_hint='literal %s'%i) for i in range(1,81)]
        messages=[dict(role='user',content='TASK\n\nAction:')]
        for _ in range(5):
            messages += [dict(role='assistant',content='backtrack_candidates[observed failure]'),
                dict(role='user',content=receipt(entries)+'\n\nAction:')]
        projected,_=project_candidate_history(messages)
        for old in candidate_receipts(projected)[:-1]:
            self.assertEqual(old['value']['result'],dict(candidate_history_refs=[['latest',0,80]]))
        self.assertEqual(expand_candidate_history(projected),messages)

    def test_controller_restore_cost_and_stop_rules_are_identical(self):
        settings=self.settings()
        responses=['bad']+['backtrack_candidates[Observed failure]']*6+[
            'backtrack_commit[rec_000001]\nInspect the current state and choose good.','good']
        outputs=[]
        for policy in ('full','once'):
            model=ScriptedModel(responses)
            with tempfile.TemporaryDirectory() as directory:
                result=Controller(CounterEnvironment(),model,replace(settings,candidate_history=policy),
                    directory,'candidate_parity').run()
                # Render every actual decision and verify reversible source receipts.
                for messages in model.requests:
                    projected,_=project_candidate_history(messages)
                    self.assertEqual(expand_candidate_history(projected),messages)
                outputs.append(result)
        keys=('success','executed_actions','completed_recoveries','consecutive_reflection_rejections','status')
        self.assertEqual({k:outputs[0][k] for k in keys},{k:outputs[1][k] for k in keys})
        for output in outputs:
            self.assertEqual(output['cost']['model_requests'],9)
            self.assertTrue(output['success'])
            self.assertEqual(output['consecutive_reflection_rejections'],0)

    def test_command_guidance_uses_one_layout_and_no_json_schema_for_lines(self):
        text=available_command_instruction(author.tools(),1200,'lines')
        self.assertNotIn('"properties"',text)
        self.assertNotIn('backtrack_commit[rec_000001]',text)
        self.assertIn('Wait for the real candidate result',text)
        self.assertIn('A real newline',text)
        self.assertIn('backtrack_commit[<record_uid>]\n<complete recovery note>',text)
        self.assertIn('a later response',boundary_instruction(remaining=0,recoveries=None,
            blocked=True,candidates=False,recovery_format='lines'))
        for invalid in ('backtrack_commit[1]\nNote', 'backtrack_commit[rec_000001] Note',
                        'backtrack_candidates[reason] backtrack_commit[1] failed_check: text'):
            with self.assertRaises(ValueError):decode_line(invalid,'lines')
        original=author.backtrack_tool_instructions(selection_policy='agent-direct',
            context_policy='rollback',workspace_policy='rollback')
        self.assertTrue(author.instruction('lines').endswith(original))

    def test_cli_records_projection_and_preserves_existing_budgets(self):
        settings=self.settings()
        self.assertEqual(settings.candidate_history,'once')
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v23_candidate_references_1200')
        self.assertIsNone(settings.max_recoveries)
        self.assertIsNone(settings.as_dict()['model_request_limit'])
        self.assertEqual((settings.max_terminal_failures,settings.max_reflection_rejections),(5,5))
        with self.assertRaises(ValueError):replace(settings,prompt_format='native-chat')


if __name__=='__main__':unittest.main()
