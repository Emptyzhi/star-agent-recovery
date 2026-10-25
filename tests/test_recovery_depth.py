import unittest

from test_web_v16 import make_controller
from experiment_settings import Experiment


class RecoveryDepthTests(unittest.TestCase):
    def test_four_real_restores_keep_distinct_sessions_and_prefix_scores(self):
        outputs = ['search[cup]']
        for _ in range(4):
            outputs += ['Search was not sufficient.', 'ACTION 0', 'search[cup]']
        controller, requests, events = make_controller(outputs, Experiment(
            max_recoveries=4, initial_actions=1, recovery_actions=1, model_request_limit=160))
        state = controller.run()
        self.assertEqual(controller.checkpoint_restore_count, 4)
        self.assertEqual(len(controller.review_sessions), 4)
        self.assertEqual(len(controller.forward_trials), 4)
        self.assertEqual([r['recovery_count'] for r in state['recovery_stages']], list(range(5)))
        self.assertEqual([r['best_purchase_reward'] for r in state['recovery_stages']], [0]*5)
        self.assertEqual(sum(e['event'] == 'red_checkpoint_restored' for e in events), 4)

    def test_initial_success_does_not_reflect_or_restore(self):
        controller, requests, events = make_controller(
            ['search[cup]', 'click[A]', 'click[red]', 'click[Buy Now]'],
            Experiment(max_recoveries=4, model_request_limit=160))
        state = controller.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(controller.checkpoint_restore_count, 0)
        self.assertEqual(len(state['recovery_stages']), 1)
        self.assertEqual(state['recovery_stages'][0]['best_purchase_reward'], 1)


if __name__ == '__main__':
    unittest.main()
