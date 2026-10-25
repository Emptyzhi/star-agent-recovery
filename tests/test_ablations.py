"""Component ablations preserve independent execution and accounting contracts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'source')]
from experiment_settings import Experiment, settings_from_args
from dispatch_webshop import build_parser, resolve_run_settings, worker_command, delegate_package_dispatch
from scoped_recovery_batch import build_parser as worker_parser
from test_web_v16 import make_controller


class AblationTests(unittest.TestCase):
    def test_old_resident_dispatches_selected_package_in_fresh_process(self):
        import dispatch_webshop
        with patch.object(dispatch_webshop, '__file__', str(PACKAGE.parent/'old/dispatch_webshop.py')), \
                patch.object(sys, 'argv', ['dispatcher', '--request', 'queued.json', '--model-path', '/models/qwen']), \
                patch.object(dispatch_webshop.os, 'execv') as execute:
            delegate_package_dispatch(PACKAGE)
            execute.assert_called_once_with(sys.executable, [sys.executable,
                str(PACKAGE/'dispatch_webshop.py'), '--request', 'queued.json', '--model-path', '/models/qwen'])
        with patch.object(dispatch_webshop.os, 'execv') as execute:
            delegate_package_dispatch(PACKAGE)
            execute.assert_not_called()

    def test_ablation_survives_queue_dispatch_cli_and_controller(self):
        args = build_parser().parse_args(['--model-path', '/models/qwen', '--request', 'r.json',
            '--results-root', 'out', '--parent-job-id', 'test-job', '--node', 'test-node'])
        for variant in ('without_yellow', 'without_red', 'without_event_subtree'):
            cfg, sampling, capacity = resolve_run_settings(args, {'experiment': {'ablation': variant}})
            command = worker_command(PACKAGE, args, 'run', [0, 2], 0, Path('out'),
                experiment=cfg, sampling=sampling, **capacity)
            parsed = worker_parser().parse_args(command[2:])
            actual = settings_from_args(parsed, Experiment)
            c, _, _ = make_controller([], actual)
            self.assertEqual(c.settings.ablation, variant)
            self.assertEqual(actual.ledger_limits(32000), Experiment().ledger_limits(32000))
            c.close()

    def test_without_yellow_skips_local_calls_but_keeps_red(self):
        c, requests, events = make_controller(['not an action', 'search[cup]', 'click[A]',
            'click[Buy Now]', 'Select the requested red color before buying.', 'ACTION 3',
            'click[red]', 'click[Buy Now]'], Experiment(ablation='without_yellow'))
        self.assertEqual(c.run()['status'], 'SUCCESS')
        self.assertEqual(c.attempt_scope_entries['YELLOW'], 0)
        self.assertEqual(c.checkpoint_restore_count, 1)
        self.assertFalse(any(e['event'] == 'yellow_local_diagnosis' for e in events))
        self.assertTrue(any('RED_HANDOFF' in r.prompt for r in requests))

    def test_without_red_keeps_partial_first_terminal_and_yellow(self):
        c, requests, _ = make_controller(['search[cup]', 'click[Next >]',
            'Inspect a visible product.', 'click[A]', 'click[Buy Now]'],
            Experiment(ablation='without_red'))
        state = c.run()
        self.assertEqual(state['status'], 'TERMINAL_FAILURE')
        self.assertEqual(c.episode.snapshot()['public_terminal_score'], .5)
        self.assertEqual(c.attempt_scope_entries['YELLOW'], 1)
        self.assertEqual(c.checkpoint_restore_count, 0)
        self.assertEqual(state['max_red_model_calls'], 0)
        self.assertEqual(state['max_recovery_actions'], 0)
        self.assertEqual(len(requests), 5)

    def test_without_red_does_not_reassign_recovery_budget(self):
        c, requests, _ = make_controller(['search[cup]'],
            Experiment(ablation='without_red', initial_actions=1))
        self.assertEqual(c.run()['status'], 'INITIAL_WINDOW_EXHAUSTED')
        self.assertEqual(len(requests), 1)
        self.assertEqual(c.checkpoint_restore_count, 0)

    def test_without_red_empty_actor_does_not_restore(self):
        c, requests, _ = make_controller(['\n'], Experiment(ablation='without_red'))
        self.assertEqual(c.run()['status'], 'INITIAL_EMPTY_ACTOR_OUTPUT')
        self.assertEqual(len(requests), 1)
        self.assertEqual(c.checkpoint_restore_count, 0)

    def test_without_subtree_preserves_red_reflection_once_and_full_ett(self):
        lesson = 'Select the requested red color before buying.'
        c, requests, events = make_controller(['search[cup]', 'click[A]', 'click[Buy Now]',
            lesson, 'ACTION 2', 'click[red]', 'click[Buy Now]'],
            Experiment(ablation='without_event_subtree'))
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(c.checkpoint_restore_count, 1)
        self.assertEqual(len(c.occurrences), 5)
        self.assertEqual(len(c.archived_lineages), 1)
        self.assertTrue(c.checkpoint_cache)
        recovery = [r for r in requests if 'Actions in this recovery attempt:' in r.prompt]
        self.assertIn(lesson, recovery[0].prompt)
        self.assertNotIn(lesson, recovery[1].prompt)
        self.assertTrue(all('EVENT_SUBTREE' not in r.prompt for r in requests))
        self.assertTrue(all('PRODUCT_REQUIREMENT_REVIEW' not in r.prompt for r in requests))
        self.assertTrue(all('HISTORICAL_BRANCH' not in r.prompt for r in recovery))
        self.assertFalse(any(e['event'] == 'web_event_subtree_delivered' for e in events))

    def test_without_subtree_keeps_yellow_trigger_menu_feedback_and_advice(self):
        diagnosis = 'Inspect a visible product.'
        c, requests, _ = make_controller(['search[cup]', 'click[Next >]', diagnosis,
            'click[A]', 'click[red]', 'click[Buy Now]'],
            Experiment(ablation='without_event_subtree'))
        self.assertEqual(c.run()['status'], 'SUCCESS')
        self.assertEqual(c.attempt_scope_entries['YELLOW'], 1)
        self.assertTrue(all('EVENT_SUBTREE' not in r.prompt for r in requests))
        choice = next(r for r in requests if r.route == 'command_trie')
        self.assertIn('click[A]', choice.commands)
        self.assertIn('current_public_page', choice.prompt)
        self.assertNotIn('occurrence_id', choice.prompt)
        self.assertIn(diagnosis, requests[-2].prompt)
        self.assertNotIn(diagnosis, requests[-1].prompt)

    def test_ablation_is_task_local_and_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            Experiment(ablation='guess')
        disabled, _, _ = make_controller([], Experiment(ablation='without_yellow'))
        enabled, _, _ = make_controller([])
        self.assertEqual(disabled.YELLOW_ENTRY_LIMIT, 0)
        self.assertEqual(enabled.YELLOW_ENTRY_LIMIT, 2)
        disabled.close()
        enabled.close()


if __name__ == '__main__':
    unittest.main()
