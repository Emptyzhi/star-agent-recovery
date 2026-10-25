import unittest

from test_web_v16 import make_episode, make_controller
from experiment_settings import Experiment
from icr_core import Ledger, GenerationRequest, Generation, BudgetExhausted
from web_v16_run import run_controller


class NativeOptionsUnlimitedTests(unittest.TestCase):
    def test_native_accepted_option_outside_public_catalog_is_recorded(self):
        episode = make_episode()
        episode.reset()
        episode.step('search[cup]')
        episode.step('click[A]')
        env = episode.environment
        env._session['option_types']['old option'] = 'color'
        self.assertNotIn('old option', env.snapshot()['option_types'])
        receipt = episode.step('click[old option]')
        self.assertTrue(receipt['native_accepted'])
        self.assertEqual(receipt['observation'], 'You have clicked old option.')
        self.assertEqual(episode.snapshot()['selected_options']['color'], 'old option')
        self.assertNotIn('old option', episode.snapshot()['option_types'])

    def test_native_rejection_stays_an_actual_invalid_action(self):
        episode = make_episode()
        episode.reset()
        receipt = episode.step('click[Back to Search]')
        self.assertFalse(receipt['native_accepted'])
        self.assertTrue(receipt['invalid_action'])

    def test_unlimited_ledger_records_more_than_160_model_requests(self):
        ledger = Ledger(Experiment(max_recoveries=4).ledger_limits(32000))
        for index in range(200):
            request = GenerationRequest(str(index), 'Action:', 1, 'normal', (), 100)
            ledger.record_generation(Generation(request, 'search[cup]', 32000, 100, {}), physical=True)
        self.assertEqual(ledger.model_requests, 200)
        self.assertIsNone(ledger.limits.model_requests)
        self.assertIsNone(ledger.limits.input_tokens)
        self.assertIsNone(ledger.limits.output_tokens)

    def test_four_recovery_resource_interruption_retains_real_stages(self):
        controller, requests, _ = make_controller(['search[cup]'], Experiment(
            max_recoveries=4, initial_actions=1, recovery_actions=1,
            model_request_limit=1))
        state = run_controller(controller)
        self.assertEqual(state['status'], 'RESOURCE_LIMIT')
        self.assertEqual(len(controller.recovery_stages), 1)
        self.assertEqual(controller.recovery_stages[0]['recovery_count'], 0)

    def test_explicit_finite_limit_still_works(self):
        ledger = Ledger(Experiment(model_request_limit=1).ledger_limits(32000))
        ledger.model_requests = 1
        with self.assertRaises(BudgetExhausted):
            ledger.reserve_model(1)


if __name__ == '__main__':
    unittest.main()
