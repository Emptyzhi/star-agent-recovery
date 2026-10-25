"""Reflection receives the real task and unchanged public final-action evidence."""
from dataclasses import replace
import unittest

from test_game24.prompts import DEMONSTRATIONS, RED_DEMONSTRATIONS, red_demonstrations
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller


class TaskBoundRedTests(unittest.TestCase):
    def test_projection_keeps_public_final_actions_without_simulated_observations(self):
        self.assertEqual(RED_DEMONSTRATIONS.count('Input: '), 5)
        self.assertEqual(RED_DEMONSTRATIONS.count('Plan: answer: '), 5)
        self.assertIn('Input: 4 4 6 8\nPlan: answer: (6 - 4) * (4 + 8) = 24', RED_DEMONSTRATIONS)
        self.assertIn('Input: 5 5 5 9\nPlan: answer: ((5 + 5) + 5) + 9 = 24', RED_DEMONSTRATIONS)
        self.assertNotIn('Act ', RED_DEMONSTRATIONS)
        self.assertNotIn('Obs ', RED_DEMONSTRATIONS)

    def test_malformed_public_assets_are_rejected_without_filling_or_dropping(self):
        for broken in (DEMONSTRATIONS.replace('Input: 4 4 6 8', 'Task: 4 4 6 8', 1),
                       DEMONSTRATIONS.replace('Act 4> answer: (6 - 4) * (4 + 8) = 24',
                           'Act 4> answer: (6 - 4) * (4 + 8) = 24\nAct 5> answer: 99 = 24', 1)):
            with self.subTest(asset=broken[:60]), self.assertRaises(ValueError):
                red_demonstrations(broken)

    def test_actual_current_task_is_supplied_after_the_failed_evidence_without_extra_calls(self):
        runner, requests = controller(BAD + GOOD, settings=replace(SHORT_SETTINGS, yellow_entries=0))
        result = runner.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        red = [request for request in requests if request['role'].startswith('red')]
        tail = red[0]['prompt'].split('[CURRENT_GAME24_TASK]', 1)[1]
        self.assertTrue(tail.startswith('\nInput: 1 1 4 6\n'))
        self.assertNotIn('Input: 5 5 5 9', tail)
        self.assertIn('Plan: answer:', tail)
        self.assertIn('fallible repair advice', tail)
        self.assertEqual([request['role'] for request in red], ['red_reflection', 'red_checkpoint'])
        self.assertEqual([request['max_new_tokens'] for request in red], [256, 100])
        self.assertFalse(red[0]['commands'])
        self.assertEqual(red[1]['commands'], ('ACTION 0', 'ACTION 1', 'ACTION 2', 'ACTION 3'))
        self.assertEqual(result.model_requests, 10)


if __name__ == '__main__':
    unittest.main()
