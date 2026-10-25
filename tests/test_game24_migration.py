"""Regression cases for the audited Game24 migration failures."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_game24.game24_env import Game24Environment
from test_game24.model_api import CallableModelClient, Generation, ResidentModelClient
from test_game24.star_controller import Game24STARController
from source.experiment_settings import Experiment

GOOD = ['1 - 1 = 0', '4 * 6 = 24', '24 + 0 = 24', 'answer: (1 - 1) + (4 * 6) = 24']
BAD = ['1 + 1 = 2', '4 * 6 = 24', '24 / 2 = 12', 'answer: (4 * 6) / (1 + 1) = 12']
SHORT_SETTINGS = replace(Experiment.for_dataset('game24'), initial_actions=4, recovery_actions=4)


def controller(actions, *, checkpoint='ACTION 0', settings=None, diagnosis='Use the available numbers.'):
    requests = []
    pending = iter(actions)
    def generate(prompt, **kw):
        requests.append(dict(prompt=prompt, **kw))
        role = kw['role']
        if role == 'red_reflection': text = 'The earlier sum was valid but led to an unsuitable value. Reconsider it.'
        elif role == 'red_checkpoint': text = checkpoint
        elif role == 'yellow_diagnosis': text = diagnosis
        else: text = next(pending)
        return Generation(text, input_tokens=10, output_tokens=2, queue_seconds=0)
    return Game24STARController(CallableModelClient(generate),
                                settings=SHORT_SETTINGS if settings is None else settings), requests


class GameMigrationTests(unittest.TestCase):
    def test_rejected_attempt_does_not_consume_an_accepted_arithmetic_stage(self):
        env = Game24Environment('1 1 4 6')
        self.assertFalse(env.execute('1 + 1 = 99').accepted)
        for action in GOOD:
            self.assertTrue(env.execute(action).accepted)
        self.assertEqual(env.action_count, 5)
        self.assertEqual(env.public_state()['accepted_operations'], 3)

    def test_invalid_fraction_is_an_explicit_rejection(self):
        env = Game24Environment('1 1 4 6')
        self.assertFalse(env.execute('1 / 1 = 1/0').accepted)
        self.assertEqual(env.action_count, 1)

    def test_recovery_gets_full_window_and_ett_retains_both_branches(self):
        c, requests = controller(BAD+GOOD)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual((result.forward_actions, result.recovery_actions), (4, 4))
        self.assertEqual(len(result.ett['nodes']), 8)
        self.assertEqual(result.ett['archived_paths'], [[0, 1, 2, 3]])
        self.assertEqual(result.ett['active_path'], [4, 5, 6, 7])
        self.assertIsNone(result.ett['nodes'][4]['parent_id'])
        self.assertEqual(len(c.node_checkpoints), 8)
        self.assertEqual([r['max_new_tokens'] for r in requests if r['role'].startswith('red')], [256, 100])
        self.assertEqual(result.cost['model_requests'], 10)
        self.assertEqual(result.cost['input_tokens'], 100)

    def test_nonroot_selection_restores_before_action_and_green_chooses(self):
        actions = ['1 - 1 = 0', '4 + 6 = 10', '10 + 0 = 10', 'answer: (1 - 1) + (4 + 6) = 10']
        c, _ = controller(actions+GOOD[1:], checkpoint='ACTION 1')
        r = c.run(0, '1 1 4 6')
        self.assertTrue(r.success)
        self.assertEqual(r.ett['active_path'], [0, 4, 5, 6])
        self.assertEqual(r.ett['nodes'][4]['parent_id'], 0)
        self.assertEqual(r.recovery_actions, 3)
        self.assertEqual([e['actions'] for e in r.events if e['event']=='window_started'], [4, 4])

    def test_two_failed_windows_do_not_start_a_third_attempt(self):
        c, _ = controller(BAD+BAD)
        r = c.run(0, '1 1 4 6')
        self.assertEqual(r.status, 'RECOVERY_WINDOW_EXHAUSTED')
        self.assertEqual(r.red_attempts, 1)
        self.assertEqual(r.cost['model_requests'], 10)

    def test_yellow_action_uses_shared_last_slot_and_is_counted(self):
        cfg = replace(SHORT_SETTINGS, initial_actions=2, ablation='without_red')
        c, _ = controller(['1 + 1 = 99', GOOD[0]], settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertEqual(r.forward_actions, 2)
        self.assertEqual(r.yellow_entries, 1)
        self.assertEqual(r.status, 'INITIAL_WINDOW_EXHAUSTED')
        self.assertEqual(len(r.ett['nodes']), 2)

    def test_no_yellow_after_last_green_slot(self):
        cfg = replace(SHORT_SETTINGS, initial_actions=1, ablation='without_red')
        c, _ = controller(['1 + 1 = 99'], settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertEqual(r.yellow_entries, 0)
        self.assertEqual(r.forward_actions, 1)

    def test_repeated_invalid_action_does_not_enter_red_before_window_end(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        c, requests = controller(['1 + 1 = 99']*4+GOOD, settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertTrue(r.success)
        self.assertEqual([q['role'] for q in requests[:5]], ['green']*4+['red_reflection'])

    def test_yellow_quota_resets_per_recovery_window(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=1)
        c, _ = controller(['1 + 1 = 99', '1 + 1 = 2', '4 * 6 = 24',
                           '24 * 2 = 48', '1 + 1 = 99', 'BACKTRACK',
                           '4 * 6 = 24', '24 * 1 = 24', '24 + 1 = 25',
                           'answer: (1 + 1) * 4 * 6 = 48'], settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertEqual(r.yellow_entries, 2)
        self.assertEqual((r.forward_actions, r.recovery_actions), (4, 4))

    def test_rejected_yellow_action_returns_control_to_green(self):
        cfg = replace(SHORT_SETTINGS, initial_actions=2, ablation='without_red')
        c, requests = controller(['1 + 1 = 99', 'invalid', '1 - 1 = 0'], settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertEqual([q['role'] for q in requests], ['green','yellow_diagnosis','yellow_action','green'])
        self.assertEqual(r.yellow_entries,1)
        self.assertEqual(r.forward_actions,2)

    def test_empty_root_can_be_selected_by_red_second_call(self):
        c, req = controller(['', *GOOD])
        r = c.run(0, '1 1 4 6')
        self.assertTrue(r.success)
        self.assertFalse(any(q['role']=='red_checkpoint' for q in req))
        self.assertEqual(r.forward_actions, 0)

    def test_global_request_limit_does_not_reset_at_red(self):
        cfg = replace(SHORT_SETTINGS, model_request_limit=4)
        c, _ = controller(BAD+GOOD, settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertEqual(r.status, 'RESOURCE_LIMIT')
        self.assertEqual(r.model_requests, 4)
        self.assertEqual(r.cost['checkpoint_restores'], 0)

    def test_without_subtree_keeps_plain_reflection_once(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_event_subtree')
        c, requests = controller(BAD+GOOD, settings=cfg)
        r = c.run(0, '1 1 4 6')
        self.assertTrue(r.success)
        green = [q['prompt'] for q in requests if q['role']=='green']
        self.assertTrue(all('[EVENT_SUBTREE]' not in q for q in green))
        for prompt in green[4:]:
            self.assertEqual(prompt.count('Red recovery reflection'), 1)
        self.assertEqual(len(r.ett['nodes']), 8)

    def test_first_line_applies_to_yellow_action_and_unknown_usage_is_not_zero(self):
        seen = []
        def generate(prompt, **kw):
            seen.append(kw)
            return Generation('1 + 1 = 99' if kw['role']=='green' else '1 - 1 = 0')
        cfg = replace(SHORT_SETTINGS, initial_actions=2, ablation='without_red')
        r = Game24STARController(CallableModelClient(generate), settings=cfg).run(0, '1 1 4 6')
        self.assertEqual(next(q for q in seen if q['role']=='yellow_action')['stop'], ('\n',))
        self.assertIsNone(r.cost['input_tokens'])
        self.assertIsNone(r.cost['queue_excluded_elapsed_seconds'])


if __name__ == '__main__': unittest.main()
