"""Native examples ground Red without adding calls or constraining Green."""
from dataclasses import replace
import unittest

from test_game24.prompts import RED_DEMONSTRATIONS, TASK_GOAL
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller


class NativeRedDemonstrationTests(unittest.TestCase):
    def test_actual_red_request_receives_actor_examples_and_operand_consumption(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        runner, requests = controller(BAD + GOOD, settings=cfg)
        result = runner.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        red = [request for request in requests if request['role'].startswith('red')]
        self.assertEqual([request['role'] for request in red], ['red_reflection', 'red_checkpoint'])
        self.assertEqual([request['max_new_tokens'] for request in red], [256, 100])
        self.assertTrue(red[0]['prompt'].startswith(TASK_GOAL))
        self.assertLess(red[0]['prompt'].index(RED_DEMONSTRATIONS), red[0]['prompt'].index('You are Red'))
        self.assertEqual(red[0]['prompt'].count(RED_DEMONSTRATIONS), 1)
        self.assertIn('exactly three accepted binary operations', red[0]['prompt'])
        self.assertIn('Equal original values are distinct occurrences', red[0]['prompt'])
        self.assertNotIn('Public candidates and their saved operand provenance', red[0]['prompt'])
        self.assertFalse(red[0]['commands'])
        self.assertEqual(red[1]['commands'], ('ACTION 0', 'ACTION 1', 'ACTION 2', 'ACTION 3'))
        self.assertEqual(result.model_requests, 10)
        for request in requests:
            if request['role'] == 'green':
                self.assertFalse(request['commands'])
                self.assertEqual(request['stop'], ('\n',))
                self.assertEqual(request['max_new_tokens'], 100)


if __name__ == '__main__':
    unittest.main()

