"""Role boundaries and receipt-based recovery evidence in the Game24 adapter."""
from dataclasses import replace
import unittest

from test_game24.prompts import DEMONSTRATIONS
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller


class RoleEvidenceTests(unittest.TestCase):
    def test_red_first_reflects_and_second_only_selects_without_extra_calls(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        c, requests = controller(BAD + GOOD, settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        red = [q for q in requests if q['role'].startswith('red')]
        self.assertEqual([q['role'] for q in red], ['red_reflection', 'red_checkpoint'])
        self.assertEqual([q['max_new_tokens'] for q in red], [256, 100])
        self.assertFalse(red[0]['commands'])
        self.assertEqual(red[1]['commands'], ('ACTION 0', 'ACTION 1', 'ACTION 2', 'ACTION 3'))
        self.assertNotIn('Public candidates and their saved operand provenance', red[0]['prompt'])
        self.assertIn('Output exactly one listed ACTION n', red[1]['prompt'])
        self.assertIn('Observed failure:', red[0]['prompt'])
        self.assertNotIn('Input: 2 12 13 13', red[0]['prompt'])
        self.assertNotIn('Input: 1 2 4 8', red[0]['prompt'])
        self.assertEqual(result.model_requests, 10)

    def test_actor_examples_are_unchanged_and_green_is_not_constrained(self):
        c, requests = controller(GOOD)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        for q in requests:
            self.assertEqual(q['role'], 'green')
            self.assertIn(DEMONSTRATIONS, q['prompt'])
            self.assertEqual(q['stop'], ('\n',))
            self.assertEqual(q['max_new_tokens'], 100)
            self.assertFalse(q['commands'])
            self.assertNotIn('parameter_language', q)

    def test_recovery_note_is_bound_to_actual_active_continuation_only(self):
        rejected = 'answer: 4 * 6 = 24'
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        c, requests = controller(BAD + [rejected, GOOD[-1]], settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        greens = [q['prompt'] for q in requests if q['role'] == 'green']
        self.assertNotIn('[EXECUTED_CONTINUATION_AFTER_RESTORE]', greens[4])
        tail = greens[5].split('[EXECUTED_CONTINUATION_AFTER_RESTORE]', 1)[1]
        self.assertIn(rejected, tail)
        self.assertIn('final expression must use every original number exactly once', tail)
        self.assertNotIn('action=' + BAD[-1], tail)
        self.assertIn('not proof that its suggested plan was followed', tail)
        self.assertEqual((result.forward_actions, result.recovery_actions), (4, 2))
        self.assertEqual(result.model_requests, 8)

    def test_public_continuation_is_not_a_subtree_ablation_bypass(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0, ablation='without_event_subtree')
        c, requests = controller(BAD + ['answer: 4 * 6 = 24', GOOD[-1]], settings=cfg)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        greens = [q['prompt'] for q in requests if q['role'] == 'green']
        self.assertTrue(all('[EVENT_SUBTREE]' not in prompt for prompt in greens))
        self.assertFalse(c.event_memory)
        self.assertIn('[EXECUTED_CONTINUATION_AFTER_RESTORE]', greens[-1])
        self.assertNotIn('action=' + BAD[-1], greens[-1].split('[EXECUTED_CONTINUATION_AFTER_RESTORE]', 1)[1])

    def test_recurrent_failed_yellow_retrieved_once_with_actual_outcome(self):
        note = 'FALLIBLE_LOCAL_CORRECTION'
        cfg = replace(SHORT_SETTINGS, initial_actions=4, yellow_entries=1, ablation='without_red')
        c, requests = controller(['1 + 1 = 99'] * 4, settings=cfg, diagnosis=note)
        self.assertFalse(c.run(0, '1 1 4 6').success)
        last = requests[-1]['prompt']
        self.assertEqual(last.count('Yellow model advice (unverified): ' + note), 1)
        self.assertIn('Public local repair outcome: UNRESOLVED', last)
        self.assertIn('feedback=declared result does not match the operation', last)
        self.assertEqual(c.yellow_entries, 1)

    def test_yellow_sees_feedback_and_current_provenance_then_costs_one_action(self):
        cfg = replace(SHORT_SETTINGS, initial_actions=2, ablation='without_red')
        c, requests = controller(['1 + 1 = 99', GOOD[0]], settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertEqual([q['role'] for q in requests], ['green', 'yellow_diagnosis', 'yellow_action'])
        yellow = requests[-1]
        self.assertIn('declared result does not match the operation', yellow['prompt'])
        self.assertIn('Current operand occurrences:', yellow['prompt'])
        self.assertIn('[EXECUTED_OPERAND_LINEAGE]', yellow['prompt'])
        self.assertIn(DEMONSTRATIONS, yellow['prompt'])
        self.assertIn('parameter_language', yellow)
        self.assertEqual((result.forward_actions, result.yellow_entries, result.red_attempts), (2, 1, 0))


if __name__ == '__main__':
    unittest.main()
