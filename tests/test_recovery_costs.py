"""Check exact stage accounting and real-controller boundary coverage offline."""
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'source')]
from recovery_costs import COST_KEYS, RecoveryCostRecorder
from test_web_v16 import make_controller
from test_alfworld_ablations import execute as execute_alf
from test_game24_migration import controller, SHORT_SETTINGS
from dataclasses import replace


class RecoveryCostTests(unittest.TestCase):
    def test_segments_partition_calls_tokens_and_time(self):
        tick = [0]
        costs = RecoveryCostRecorder(0, clock=lambda: tick[0])
        def request(input_tokens, output_tokens):
            costs.record_generation(SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                raw_response={'queue_time_seconds': .25}))
        request(10, 1)
        tick[0] = 3
        costs.observe('red_preparation_started')
        request(20, 2)
        request(30, 3)
        tick[0] = 7
        costs.observe('red_checkpoint_restored')
        request(40, 4)
        tick[0] = 12
        costs.observe('recovery_stage_finished')
        result = costs.report(12)
        self.assertEqual([s['phase'] for s in result['segments']],
                         ['execution', 'recovery_preparation', 'execution'])
        self.assertEqual([s['cost']['model_requests'] for s in result['segments']], [1, 2, 1])
        self.assertEqual([s['cost']['elapsed_seconds'] for s in result['segments']], [3, 4, 5])
        for key in (*COST_KEYS, 'elapsed_seconds', 'queue_excluded_elapsed_seconds'):
            self.assertAlmostEqual(sum(s['cost'][key] for s in result['segments']), result['totals'][key])

    def test_no_restore_does_not_invent_a_recovered_execution(self):
        costs = RecoveryCostRecorder(0, clock=lambda: 1)
        costs.observe('red_preparation_started')
        result = costs.report(2)
        self.assertEqual(len(result['segments']), 2)
        self.assertEqual(result['segments'][-1]['end_reason'], 'task_ended_before_restore')

    def test_all_three_actual_controllers_emit_preparation_before_restore(self):
        web, _, web_events = make_controller(['search[cup]', 'click[A]', 'click[Buy Now]',
            'Select red.', 'ACTION 2', 'click[red]', 'click[Buy Now]'])
        self.assertEqual(web.run()['status'], 'SUCCESS')
        alf, _, alf_events, _ = execute_alf(['look', 'Take the apple.', 'ACTION 0',
            'take apple 1 from table 1', 'put apple 1 in/on table 2'], initial_actions=1)
        self.assertTrue(alf['success'])
        game, _ = controller(['bad']*8, settings=replace(SHORT_SETTINGS, yellow_entries=0))
        game_result = game.run(0, '1 1 4 6')
        for events in (web_events, alf_events, game_result.events):
            kinds = [e['event'] for e in events]
            self.assertEqual(kinds.count('red_preparation_started'), 1)
            self.assertEqual(kinds.count('red_checkpoint_restored'), 1)
            self.assertLess(kinds.index('red_preparation_started'), kinds.index('red_checkpoint_restored'))


if __name__ == '__main__':
    unittest.main()
