"""Recovery roles must receive the task goal even without Actor demonstrations."""
from dataclasses import replace
import unittest

from test_game24.prompts import TASK_GOAL
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller


class RecoveryGoalTests(unittest.TestCase):
    def test_green_gets_detailed_lineage_when_the_native_state_requires_a_final_submission(self):
        c, requests = controller(GOOD)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        greens = [row for row in requests if row['role']=='green']
        self.assertTrue(all('[EXECUTED_OPERAND_LINEAGE]' not in row['prompt'] for row in greens[:3]))
        self.assertIn('[EXECUTED_OPERAND_LINEAGE]', greens[3]['prompt'])
        self.assertIn('executed expression ((4 * 6) + (1 - 1))', greens[3]['prompt'])
        self.assertTrue(all('Required ORIGINAL occurrences in a final expression:' in row['prompt'] for row in greens))
        self.assertTrue(all(not row['commands'] and 'parameter_language' not in row for row in greens))
        self.assertEqual((result.forward_actions,result.recovery_actions,result.model_requests),(4,0,4))

    def test_both_red_requests_include_explicit_target_without_new_call_or_review(self):
        c, requests = controller(BAD + GOOD, settings=replace(SHORT_SETTINGS, yellow_entries=0))
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        red = [row for row in requests if row['role'].startswith('red')]
        self.assertEqual([row['role'] for row in red], ['red_reflection', 'red_checkpoint'])
        self.assertTrue(all(row['prompt'].startswith(TASK_GOAL) for row in red))
        self.assertEqual(red[1]['commands'], ('ACTION 0', 'ACTION 1', 'ACTION 2', 'ACTION 3'))
        self.assertIn('Output exactly one listed ACTION n', red[1]['prompt'])
        self.assertNotIn('Public candidates and their saved operand provenance', red[0]['prompt'])
        self.assertEqual([row['max_new_tokens'] for row in red], [256, 100])
        self.assertEqual(result.model_requests, 10)

    def test_yellow_receives_the_same_goal_and_uses_shared_action_slots(self):
        c, requests = controller(['1 + 1 = 99', GOOD[0]],
            settings=replace(SHORT_SETTINGS, initial_actions=2, ablation='without_red'))
        result = c.run(0, '1 1 4 6')
        yellow = [row for row in requests if row['role'].startswith('yellow')]
        self.assertEqual([row['role'] for row in yellow], ['yellow_diagnosis', 'yellow_action'])
        self.assertTrue(all(TASK_GOAL in row['prompt'] for row in yellow))
        self.assertEqual((result.forward_actions, result.yellow_entries, result.red_attempts), (2, 1, 0))

    def test_empty_initial_output_still_preserves_native_root_boundary(self):
        c, requests = controller(['', *GOOD])
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions, 0)
        self.assertFalse(any(row['role']=='red_checkpoint' for row in requests))
        reflection = next(row for row in requests if row['role']=='red_reflection')
        self.assertTrue(reflection['prompt'].startswith(TASK_GOAL))
        self.assertFalse(reflection['commands'])


if __name__ == '__main__':
    unittest.main()
