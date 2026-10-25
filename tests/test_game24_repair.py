"""Failure-driven contracts without a model, arithmetic search, or extra actions."""
from dataclasses import replace
import unittest

from test_game24_migration import controller, GOOD, BAD, SHORT_SETTINGS
from test_game24.model_api import CallableModelClient, Generation
from test_game24.star_controller import Game24STARController
from source.experiment_settings import Experiment


class RepairV22Tests(unittest.TestCase):
    def test_twelve_plus_twelve_is_dataset_default(self):
        cfg = Experiment.for_dataset('game24')
        self.assertEqual((cfg.initial_actions, cfg.recovery_actions), (12, 12))
        self.assertEqual((cfg.red_reflection_tokens, cfg.red_selection_tokens), (256, 100))
        self.assertIsNone(cfg.model_request_limit)

    def test_yellow_repairs_format_without_controller_rewriting(self):
        c, requests = controller(['(1 - 1) + 4 * 6 = 24', GOOD[-1]])
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions, 1)
        self.assertEqual(result.yellow_entries, 1)
        self.assertEqual(result.history[0]['action'], GOOD[-1])
        self.assertEqual(result.ett['nodes'][0]['scope'], 'YELLOW')
        self.assertIn('INVALID_ACTION_FORMAT', requests[1]['prompt'])
        self.assertFalse(any(n['receipt']['action'].startswith('(1 - 1)') for n in result.ett['nodes']))

    def test_rejected_answer_receipt_reaches_yellow(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_red')
        c, requests = controller(['answer: 1 + 1 + 4 + 6 = 24', GOOD[-1]], settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual((result.forward_actions, result.yellow_entries), (2, 1))
        diagnosis = next(r['prompt'] for r in requests if r['role']=='yellow_diagnosis')
        self.assertIn('declared result does not match the expression', diagnosis)

    def test_current_checkpoint_is_eligible_after_exhausting_invalid_format(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        c, requests = controller(GOOD[:3]+['```', '```', GOOD[-1]],
                                 checkpoint='ACTION 3', settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.recovery_actions, 2)
        prompt = next(r['prompt'] for r in requests if r['role']=='red_checkpoint')
        self.assertIn('ACTION 3: after 3 recorded actions; numbers left: 24', prompt)
        reflection = next(r['prompt'] for r in requests if r['role']=='red_reflection')
        self.assertIn('invalid arithmetic action', reflection)

    def test_no_yellow_ablation_uses_native_feedback_and_action_bound(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_yellow')
        c, requests = controller(['```']*8, settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertEqual(result.status, 'RECOVERY_WINDOW_EXHAUSTED')
        self.assertEqual(result.forward_actions+result.recovery_actions, 8)
        self.assertEqual(result.yellow_entries, 0)
        self.assertEqual(result.model_requests, 10)
        greens = [r['prompt'] for r in requests if r['role']=='green']
        self.assertIn('invalid arithmetic action', greens[1])

    def test_different_protocol_errors_do_not_use_up_request_budget(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_yellow')
        c, _ = controller(['```', 'Failure', 'think: no', 'I cannot solve']*2, settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertEqual(r.status, 'RECOVERY_WINDOW_EXHAUSTED')
        self.assertEqual(r.model_requests, 10)
        self.assertEqual(r.red_attempts, 1)

    def test_last_slot_is_disclosed_without_forcing_answer(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_red')
        c, requests = controller(['1 + 1 = 99', '1 - 1 = 0', '4 * 6 = 24', '24 + 0 = 24'], settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertFalse(r.success)
        self.assertEqual(r.forward_actions, 4)
        self.assertIn('Actions remaining in this window: 1.', requests[-1]['prompt'])
        self.assertEqual(r.history[-1]['action'], '24 + 0 = 24')

    def test_yellow_malformed_output_is_publicly_reported_and_preserved_for_red(self):
        c, requests = controller(['1 + 1 = 99', '```', '1 - 1 = 0', '4 * 6 = 24',
                                  '24 + 0 = 24', *GOOD])
        r = c.run(0, '1 1 4 6')
        self.assertTrue(r.success)
        self.assertTrue(any('Not executed (yellow_action)' in q['prompt'] for q in requests if q['role']=='green'))
        reflection = next(q['prompt'] for q in requests if q['role']=='red_reflection')
        self.assertIn('Unexecuted yellow_action proposal:', reflection)

    def test_rollback_request_does_not_create_additional_trial(self):
        c, _ = controller(['1 + 1 = 99', 'BACKTRACK', '1 + 1 = 99',
                           'BACKTRACK', *GOOD])
        result = c.run(0, '1 1 4 6')
        self.assertEqual(result.red_attempts, 1)
        self.assertEqual(result.forward_actions, 4)
        self.assertEqual(result.recovery_actions, 1)
        self.assertEqual(result.yellow_entries, 2)

    def test_fraction_correction_uses_model_text_and_equal_occurrences_are_allowed(self):
        c, requests = controller(['8 / 3 = 2.6667','8 / 3 = 8/3',
                                 '3 * 3 = 9','9 * 8/3 = 24',
                                 'answer: (3 * 3) * (8 / 3) = 24'])
        result = c.run(0, '3 3 3 8')
        self.assertTrue(result.success)
        self.assertEqual(result.ett['nodes'][1]['receipt']['action'], '8 / 3 = 8/3')
        self.assertTrue(all('two distinct current values' not in r['prompt'] for r in requests))
        self.assertIn('Exact fractions', next(r['prompt'] for r in requests if r['role']=='yellow_diagnosis'))


if __name__ == '__main__':
    unittest.main()
