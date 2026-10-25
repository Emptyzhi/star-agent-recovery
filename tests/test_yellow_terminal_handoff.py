"""Terminal Yellow receipts must merge without querying a nonexistent action interface."""
import unittest
from unittest.mock import patch
from test_web_v16 import make_controller
from experiment_settings import Experiment
from scoped_recovery import ScopeKind


class YellowTerminalHandoff(unittest.TestCase):
    def test_initial_yellow_purchase_returns_actual_terminal_failure(self):
        c, requests, events = make_controller([
            'search[cup]', 'click[A]', 'search[another cup]',
            'Searching is unavailable on the current item page.', 'click[Buy Now]',
        ], Experiment(ablation='without_red'))
        support = c.episode.support

        def live_support():
            self.assertFalse(c.episode.snapshot()['done'])
            return support()

        with patch.object(c.episode, 'support', side_effect=live_support):
            state = c.run()
        self.assertEqual(state['status'], 'TERMINAL_FAILURE')
        self.assertEqual(c.episode.snapshot()['public_terminal_score'], .5)
        self.assertEqual(c.active_scope.kind, ScopeKind.GREEN)
        self.assertEqual(c.local_outcomes[-1]['outcome'], 'ENVIRONMENT_TERMINAL')
        self.assertEqual(requests[-1].commands.count('click[Buy Now]'), 1)
        self.assertTrue(any(e['event'] == 'scope_returned' and e['reason'] == 'ENVIRONMENT_TERMINAL' for e in events))

    def test_yellow_purchase_after_red_keeps_terminal_receipt_and_costs(self):
        c, requests, events = make_controller([
            'search[cup]', 'click[A]', 'click[Buy Now]',
            'Select the requested red color before buying.', 'ACTION 2',
            'search[another cup]', 'Searching is unavailable on the current item page.',
            'click[Buy Now]',
        ])
        state = c.run()
        self.assertEqual(state['status'], 'TERMINAL_FAILURE')
        self.assertEqual(c.episode.snapshot()['public_terminal_score'], .5)
        self.assertEqual(c.checkpoint_restore_count, 1)
        self.assertEqual(c.episode.backend_requests, 5)
        self.assertEqual(len(requests), 8)
        self.assertEqual(c.local_outcomes[-1]['outcome'], 'ENVIRONMENT_TERMINAL')
        self.assertFalse(any(e['event'] == 'fatal' for e in events))


if __name__ == '__main__':
    unittest.main()
