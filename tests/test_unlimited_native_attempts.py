"""Native feedback and action limits replace free post-Yellow format retries."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from test_web_v16 import make_controller
from test_game24_migration import controller, GOOD, SHORT_SETTINGS
from test_alfworld_ablations import execute as execute_alf
from test_unified_cli import arguments
from alf_configured import ConfiguredAlfController
from experiment_settings import Experiment
from icr_core import ProtocolError
import run_star


class UnlimitedNativeAttemptTests(unittest.TestCase):
    def test_all_dataset_cli_defaults_have_no_call_or_derived_token_cap(self):
        for dataset in ('webshop', 'game24', 'alfworld'):
            with self.subTest(dataset=dataset):
                args = run_star.build_parser().parse_args(arguments(dataset))
                cfg, _, _ = run_star.resolve(args)
                self.assertIsNone(cfg.model_request_limit)
                limits = cfg.ledger_limits(32000)
                self.assertIsNone(limits.model_requests)
                self.assertIsNone(limits.input_tokens)
                self.assertIsNone(limits.output_tokens)

    def test_webshop_bad_spacing_receipt_reaches_next_actor(self):
        cfg = Experiment(yellow_entries=0)
        c, requests, _ = make_controller(
            ['Click [Buy Now]', 'search[cup]', 'click[A]', 'click[red]', 'click[Buy Now]'], cfg)
        self.assertEqual(c.run()['status'], 'SUCCESS')
        self.assertEqual(c.occurrences[0].action, 'click [Buy Now]')
        self.assertFalse(c.occurrences[0].receipt['native_accepted'])
        self.assertIn('Action: click [Buy Now]\nObservation: Invalid action!', requests[1].prompt)
        self.assertEqual(len(c.occurrences), 5)
        self.assertEqual(c.checkpoint_restore_count, 0)

    def test_webshop_persistent_bad_or_empty_actions_exhaust_window(self):
        for text in ('Click [Buy Now]', ''):
            with self.subTest(text=text):
                cfg = Experiment(ablation='without_red', yellow_entries=0, initial_actions=70)
                c, requests, _ = make_controller([text]*70, cfg)
                self.assertEqual(c.run()['status'], 'INITIAL_WINDOW_EXHAUSTED')
                self.assertEqual(len(c.occurrences), 70)
                self.assertEqual(len(requests), 70)
                self.assertFalse(c.empty_actor_events)

    def test_game24_bad_and_empty_actions_exhaust_both_windows(self):
        for text in ('```', ''):
            with self.subTest(text=text):
                cfg = replace(SHORT_SETTINGS, yellow_entries=0)
                c, requests = controller([text]*8, settings=cfg)
                result = c.run(0, '1 1 4 6')
                self.assertEqual(result.status, 'RECOVERY_WINDOW_EXHAUSTED')
                self.assertEqual((result.forward_actions, result.recovery_actions), (4, 4))
                self.assertEqual(result.model_requests, 10)
                self.assertEqual(len(result.ett['nodes']), 8)
                greens = [r['prompt'] for r in requests if r['role']=='green']
                self.assertIn('invalid arithmetic action', greens[1])
                self.assertFalse(any(e['event']=='game24_protocol_stall' for e in result.events))

    def test_game24_yellow_can_still_correct_before_submission(self):
        c, _ = controller(['```', GOOD[-1]])
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.yellow_entries, 1)
        self.assertEqual(result.forward_actions, 1)

    def test_game24_exhausted_yellow_submits_later_bad_and_empty_actions(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_red', yellow_entries=1)
        c, requests = controller(['```', GOOD[0], '```', '', GOOD[-1]], settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual((result.forward_actions, result.yellow_entries), (4, 1))
        self.assertEqual([n['receipt']['action'] for n in result.ett['nodes']],
                         [GOOD[0], '```', '', GOOD[-1]])
        self.assertIn('invalid arithmetic action', requests[-1]['prompt'])

    def test_alfworld_empty_or_malformed_action_reaches_native_backend(self):
        for text, action in (('', ''), ('look\rbroken', 'look\rbroken')):
            with self.subTest(text=text):
                result, requests, _, episode = execute_alf(
                    [text, 'take apple 1 from table 1', 'put apple 1 in/on table 2'],
                    ablation='without_red', yellow_entries=0)
                self.assertEqual(result['status'], 'SUCCESS')
                self.assertEqual(episode.history[0]['action'], action)
                self.assertEqual(episode.history[0]['observation'], 'Nothing happens.')
                self.assertIn('Nothing happens.', requests[1].prompt)
                self.assertEqual(len(episode.history), 3)

    def test_alfworld_exhausted_nonzero_yellow_allowance_cannot_veto_empty(self):
        original = ConfiguredAlfController.start
        def start_exhausted(c):
            original(c)
            c.attempt_scope_entries['YELLOW'] = c.YELLOW_ENTRY_LIMIT
        with patch.object(ConfiguredAlfController, 'start', start_exhausted):
            result, requests, events, episode = execute_alf(
                ['', 'take apple 1 from table 1', 'put apple 1 in/on table 2'],
                ablation='without_red', yellow_entries=1)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(episode.history[0]['action'], '')
        self.assertTrue(any(e['event']=='actor_protocol_failure_submitted' for e in events))

    def test_alfworld_persistent_empty_actions_exhaust_native_window(self):
        result, requests, _, episode = execute_alf(['']*50,
            ablation='without_red', yellow_entries=0)
        self.assertEqual(result['status'], 'INITIAL_WINDOW_EXHAUSTED')
        self.assertEqual(len(episode.history), 50)
        self.assertEqual(len(requests), 50)

    def test_alfworld_empty_attempts_exhaust_both_50_action_windows(self):
        result, requests, _, _ = execute_alf(
            ['']*50 + ['Try a supported native command.', 'ACTION 0'] + ['']*50,
            yellow_entries=0)
        self.assertEqual(result['status'], 'RECOVERY_WINDOW_EXHAUSTED')
        self.assertEqual(len(result['state']['occurrences']), 100)
        self.assertEqual(len(requests), 102)

    def test_empty_tree_edge_requires_the_matching_native_receipt(self):
        c, _, _ = make_controller([''], Experiment(
            ablation='without_red', yellow_entries=0, initial_actions=1))
        c.run()
        receipt = dict(c.occurrences[0].receipt, action='look')
        with self.assertRaisesRegex(ProtocolError, 'empty tree action requires'):
            c.scopes[0].tree.add_edge(0, '', receipt, visit_kind='SCOPED_EXECUTION', attempt=1)


if __name__ == '__main__':
    unittest.main()
