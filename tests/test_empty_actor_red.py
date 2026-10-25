"""Empty Actor failures preserve real WebShop state and use at most one Red."""
import json
import unittest

from test_web_v16 import make_controller
from icr_core import ProtocolError


class EmptyActorRedTests(unittest.TestCase):
    def test_initial_empty_recovers_without_fake_action_or_http_request(self):
        lesson = 'Search for the requested cup and select red before buying.'
        c, requests, events = make_controller([
            '', lesson, 'search[cup]', 'click[A]', 'click[red]', 'click[Buy Now]'])
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(c.selected_fault_index, 0)
        self.assertEqual(len(c.occurrences), 4)
        self.assertEqual(c.episode.backend_requests, 5)
        self.assertEqual(c.checkpoint_restore_count, 1)
        self.assertEqual(c.forward_trials[0].forward_actions_granted, 15)
        self.assertIsNone(c.forward_trials[0].source_checkpoint_occurrence_id)
        self.assertEqual([r.route for r in requests],
                         ['normal', 'reflection', 'normal', 'normal', 'normal', 'normal'])
        self.assertEqual(requests[1].output_limit, 256)
        self.assertIn('No action was submitted', requests[1].prompt)
        scenario = requests[1].prompt.rsplit('Instruction:', 1)[1]
        self.assertIn('Buy a red cup below 20 dollars.', scenario)
        self.assertIn('EMPTY_ACTOR_OUTPUT', scenario)
        self.assertIn('EMPTY_ACTOR_OUTPUT', requests[2].prompt)
        self.assertIn('RED_HANDOFF', requests[2].prompt)
        self.assertIn(lesson, requests[2].prompt)
        self.assertTrue(all(lesson not in r.prompt for r in requests[3:]))
        self.assertEqual(len([e for e in events if e['event'] == 'red_memory_delivered']), 0)
        failure = c.empty_actor_events[0]
        self.assertFalse(failure['action_executed'])
        self.assertEqual(failure['executed_prefix_length'], 0)
        self.assertFalse(any(o.action == '' for o in c.occurrences.values()))

    def test_empty_on_item_preserves_product_options_and_prefix(self):
        c, requests, events = make_controller([
            'search[cup]', 'click[A]', 'click[red]', '',
            'The requested red option is selected; purchase this item.', 'click[Buy Now]'])
        state = c.run()
        self.assertEqual(state['status'], 'SUCCESS')
        self.assertEqual(c.selected_fault_index, 3)
        self.assertEqual(c.episode.action_limit, 18)
        self.assertEqual(c.forward_trials[0].forward_actions_granted, 15)
        self.assertEqual(c.forward_trials[0].forward_actions_used, 1)
        self.assertEqual(len(c.lineage), 4)
        self.assertEqual(c.episode.backend_requests, 5)
        restore = next(e for e in events if e['event'] == 'red_checkpoint_restored')
        self.assertEqual(restore['actual_state']['asin'], 'A')
        self.assertEqual(restore['actual_state']['selected_options'], {'color': 'red'})
        self.assertEqual(restore['actual_state']['page_type'], 'item')
        self.assertIsNone(restore['selection_request_id'])
        self.assertIn('EMPTY_ACTOR_OUTPUT', requests[-1].prompt)
        self.assertIn('No action was submitted', requests[4].prompt)
        self.assertIn('Action: click[red]', requests[4].prompt.rsplit('Instruction:', 1)[1])
        self.assertNotIn('HIDDEN', requests[-1].prompt)
        self.assertFalse(any(r.route == 'command_trie' for r in requests))

    def test_second_empty_is_bounded_failure_without_new_red(self):
        c, requests, events = make_controller(['', 'Begin with a search for the requested cup.', ''])
        state = c.run()
        self.assertEqual(state['status'], 'RECOVERY_EMPTY_ACTOR_OUTPUT')
        self.assertEqual(len(c.occurrences), 0)
        self.assertEqual(len(requests), 3)
        self.assertEqual(c.episode.backend_requests, 1)
        self.assertEqual(c.checkpoint_restore_count, 1)
        self.assertEqual(len(c.empty_actor_events), 2)
        self.assertIsNone(c.episode.snapshot()['public_terminal_score'])
        self.assertEqual(c.forward_trials[0].forward_actions_used, 0)
        self.assertEqual(c.forward_trials[0].reason, 'EMPTY_ACTOR_OUTPUT_AFTER_RED')
        json.dumps(state, allow_nan=False)

    def test_empty_after_normal_red_does_not_allow_second_recovery(self):
        c, requests, events = make_controller([
            'search[cup]', 'click[A]', 'click[Buy Now]',
            'Select red before buying.', 'ACTION 2', ''])
        self.assertEqual(c.run()['status'], 'RECOVERY_EMPTY_ACTOR_OUTPUT')
        self.assertEqual(c.selected_fault_index, 2)
        self.assertEqual(c.checkpoint_restore_count, 1)
        self.assertEqual(sum(r.route == 'reflection' for r in requests), 1)
        self.assertEqual(len(c.occurrences), 3)
        self.assertEqual(c.episode.backend_requests, 4)

    def test_whitespace_only_failure_keeps_literal_model_evidence(self):
        c, requests, events = make_controller([' \n', 'Search for the requested cup.', ''])
        self.assertEqual(c.run()['status'], 'RECOVERY_EMPTY_ACTOR_OUTPUT')
        self.assertEqual(c.empty_actor_events[0]['raw_text'], ' \n')
        self.assertEqual(c.episode.backend_requests, 1)

    def test_empty_red_reflection_remains_explicit_protocol_failure(self):
        c, requests, events = make_controller(['', ''])
        with self.assertRaisesRegex(ProtocolError, 'Red reflection is empty'):
            c.run()
        self.assertEqual(c.checkpoint_restore_count, 0)
        self.assertEqual(c.episode.backend_requests, 1)


if __name__ == '__main__':
    unittest.main()
