"""No-model regression tests for every ablated input route and checkpoint use."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from test_alfworld_ablations import execute
from test_web_v16 import make_controller
from test_game24_migration import controller, GOOD, SHORT_SETTINGS
from experiment_settings import Experiment

FORBIDDEN = ('EVENT_SUBTREE', 'LOCAL_REPAIR_HANDOFF', 'HISTORICAL_REJECTION',
             'REPEATED_REJECTION', 'Receipt nodes', 'dependency nodes',
             'accepted_state_change_nodes', 'previous_ineffective_actions_at_current_state',
             'parent_occurrence_id', 'occurrence_id', 'RED_PHASE1_EVIDENCE',
             'PRODUCT_REQUIREMENT_REVIEW', 'commands_added', 'commands_removed')


class PlainHistoryTests(unittest.TestCase):
    def assert_plain(self, prompts):
        for prompt in prompts:
            for forbidden in FORBIDDEN:
                self.assertNotIn(forbidden, prompt)

    def test_alf_all_roles_keep_raw_diagnosis_actions_and_restore(self):
        outputs = ['bad command', 'bad command', 'LOCAL_TEST_NOTE',
                   'take apple 1 from table 1', 'REFLECTION_TEST_NOTE', 'ACTION 0',
                   'take apple 1 from table 1', 'put apple 1 in/on table 2']
        with patch('alf_recovery_projection.recovery_subtree', side_effect=AssertionError('retrieval')):
            result, requests, events, episode = execute(
                outputs, 'without_event_subtree', initial_actions=2)
        self.assertTrue(result['success'])
        self.assert_plain([r.prompt for r in requests])
        self.assertIn('LOCAL_TEST_NOTE', requests[-1].prompt)
        self.assertIn('REFLECTION_TEST_NOTE', requests[-1].prompt)
        self.assertIn('> bad command\nNothing happens.', requests[-1].prompt)
        self.assertIn('ACTION 0: bad command', next(r.prompt for r in requests if r.commands and 'ACTION 0' in r.commands))
        self.assertEqual(result['replay_resets'], 1)
        self.assertEqual(episode.recovery_event_subtree, {'events': [], 'nodes': []})

    def test_web_all_roles_keep_chronology_without_retrieved_transition_pack(self):
        outputs = ['search[cup]', 'click[Next >]', 'LOCAL_TEST_NOTE', 'click[A]',
                   'click[Buy Now]', 'REFLECTION_TEST_NOTE', 'ACTION 3',
                   'click[red]', 'click[Buy Now]']
        c, requests, _ = make_controller(outputs, Experiment(ablation='without_event_subtree'))
        with patch('web_v16_controller.green_events', side_effect=AssertionError('retrieval')):
            result = c.run()
        self.assertEqual(result['status'], 'SUCCESS')
        self.assert_plain([r.prompt for r in requests])
        self.assertIsNone(c.red_phase1_evidence)
        prompt = requests[-1].prompt
        ordered = ['Action: search[cup]', 'Action: click[Next >]', 'LOCAL_TEST_NOTE',
                   'Action: click[A]', 'Failure reflection (model advice): REFLECTION_TEST_NOTE',
                   'Restored the environment to before ACTION', 'Action: click[red]']
        offsets = [prompt.index(item) for item in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(c.checkpoint_restore_count, 1)

    def test_game24_tcr_no_local_projection_even_after_repeated_failure(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_event_subtree', yellow_entries=1)
        c, requests = controller(['1 + 1 = 99'] * 4 + [GOOD[-1]],
                                diagnosis='LOCAL_TEST_NOTE', settings=cfg)
        with patch.object(c, '_local_event_projection', side_effect=AssertionError('local projection')):
            result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assert_plain([r['prompt'] for r in requests])
        reflection = next(r['prompt'] for r in requests if r['role'] == 'red_reflection')
        self.assertIn('LOCAL_TEST_NOTE', reflection)
        self.assertEqual(reflection.count('> 1 + 1 = 99'), 4)
        selection = next(r['prompt'] for r in requests if r['role'] == 'red_checkpoint')
        self.assertIn('ACTION 0: 1 + 1 = 99', selection)
        self.assertNotIn('parents=', selection)
        self.assertNotIn('consumed=', selection)
        self.assertEqual((result.forward_actions, result.recovery_actions), (4, 1))
        self.assertTrue(result.ett['nodes'])


if __name__ == '__main__':
    unittest.main()
