"""Finalization advice must use receipts without executing or repairing outputs."""
from dataclasses import replace
import unittest

from test_game24.game24_env import Game24Environment
from test_game24.public_evidence import executed_operand_lineage
from test_game24_migration import GOOD, SHORT_SETTINGS, controller


WRONG = 'answer: 1 + 1 + 4 + 6 = 24'


class FinalizationEvidenceTests(unittest.TestCase):
    def test_receipt_lineage_is_factual_without_becoming_a_proposed_action(self):
        env = Game24Environment('1 1 4 6')
        for action in GOOD[:3]:
            self.assertTrue(env.execute(action).accepted)
        before = env.checkpoint()
        view = executed_operand_lineage(env.puzzle, env.history, env.public_state())
        self.assertIn('recorded environment facts, not proposed repairs', view)
        self.assertIn('executed expression ((4 * 6) + (1 - 1))', view)
        self.assertNotIn('answer:', view)
        self.assertEqual(env.checkpoint(), before)
        self.assertFalse(env.public_state()['done'])

    def test_green_finalization_is_unconstrained_and_accepts_an_equivalent_expression(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0, ablation='without_red')
        runner, requests = controller(GOOD, settings=cfg)
        result = runner.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions, 4)
        self.assertEqual(result.model_requests, 4)
        self.assertIn('preserving its operands, operators and parentheses', requests[-1]['prompt'])
        self.assertTrue(all(not request['commands'] for request in requests))
        self.assertTrue(all('parameter_language' not in request for request in requests))
        self.assertEqual(result.ett['nodes'][-1]['receipt']['action'], GOOD[-1])

    def test_wrong_model_final_action_is_submitted_without_rewriting(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0, ablation='without_red')
        runner, requests = controller(GOOD[:3] + [WRONG], settings=cfg)
        result = runner.run(0, '1 1 4 6')
        self.assertFalse(result.success)
        self.assertEqual(result.forward_actions, 4)
        self.assertEqual(result.model_requests, 4)
        self.assertEqual(result.ett['nodes'][-1]['receipt']['action'], WRONG)
        self.assertFalse(result.ett['nodes'][-1]['receipt']['accepted'])
        self.assertEqual(len(requests), 4)

    def test_yellow_uses_current_receipts_and_still_consumes_the_shared_budget(self):
        cfg = replace(SHORT_SETTINGS, initial_actions=6, yellow_entries=1, ablation='without_red')
        runner, requests = controller(GOOD[:3] + [WRONG] * 3, settings=cfg)
        result = runner.run(0, '1 1 4 6')
        self.assertFalse(result.success)
        self.assertEqual(result.forward_actions, 6)
        self.assertEqual(result.model_requests, 7)
        yellow = [request for request in requests if request['role'].startswith('yellow')]
        self.assertEqual([request['role'] for request in yellow], ['yellow_diagnosis', 'yellow_action'])
        self.assertIn('CURRENT executed pool is 24', yellow[0]['prompt'])
        self.assertIn('recorded executed expression', yellow[1]['prompt'])
        self.assertEqual(result.ett['nodes'][4]['receipt']['action'], WRONG)
        self.assertFalse(result.ett['nodes'][4]['receipt']['accepted'])

    def test_restored_root_does_not_inherit_a_historical_finalization_hint(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        runner, requests = controller(GOOD[:3] + [WRONG] + GOOD,
                                      checkpoint='ACTION 0', settings=cfg)
        result = runner.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        greens = [request for request in requests if request['role'] == 'green']
        self.assertIn('OPERATION STATE:', greens[4]['prompt'])
        self.assertNotIn('FINAL-ACTION STATE: the current pool is 24.', greens[4]['prompt'])
        red = [request for request in requests if request['role'].startswith('red')]
        self.assertEqual([request['role'] for request in red], ['red_reflection', 'red_checkpoint'])
        self.assertFalse(red[0]['commands'])
        self.assertEqual(red[1]['commands'], ('ACTION 0', 'ACTION 1', 'ACTION 2', 'ACTION 3'))


if __name__ == '__main__':
    unittest.main()
