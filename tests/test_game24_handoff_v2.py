"""Evidence delivery tests use recorded receipts and scripted model replies only."""
from dataclasses import replace
import unittest

from test_game24_migration import controller, GOOD, SHORT_SETTINGS
from test_game24_receipt_projection import initialized, deliver


class Game24HandoffV2Tests(unittest.TestCase):
    def test_yellow_advice_is_delivered_once_and_stays_unverified(self):
        note = 'WRONG_NOTE_USE_INTERMEDIATE_AS_ORIGINAL'
        c, requests = controller(['1 + 1 = 99', *GOOD], diagnosis=note,
            settings=replace(SHORT_SETTINGS, initial_actions=5, ablation='without_red'))
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        greens = [q['prompt'] for q in requests if q['role'] == 'green']
        self.assertIn('Yellow model advice (unverified): '+note, greens[1])
        self.assertIn('ACCEPTED_PUBLIC_STATE_CHANGE', greens[1])
        self.assertIn('LOCAL_ACCEPTANCE_ONLY_NOT_VERIFIED_SOLUTION_PROGRESS', greens[1])
        self.assertNotIn(note, greens[2])
        self.assertEqual(result.forward_actions, 5)
        self.assertEqual(result.recovery_actions, 0)
        self.assertEqual(result.ett['adaptation_version'], 'game24_receipt_finalization_v12')

    def test_unexecuted_yellow_proposal_is_not_an_environment_fact(self):
        c, requests = controller(['1 + 1 = 99', 'malformed', *GOOD],
            settings=replace(SHORT_SETTINGS, initial_actions=5, ablation='without_red'))
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        prompt = [q['prompt'] for q in requests if q['role'] == 'green'][1]
        self.assertIn('No environment action was executed by this Yellow incident.', prompt)
        self.assertIn('Public local repair outcome: UNRESOLVED', prompt)
        self.assertNotIn('malformed', [n['receipt']['action'] for n in result.ett['nodes']])

    def test_same_state_rejections_recalled_after_restore_not_new_branch(self):
        c = initialized('1 1 4 6')
        root = c.env.checkpoint()
        action = '1 + 1 = 99'
        c._execute(action, 'GREEN')
        c.archived_paths.append(c.active_path[:])
        c.active_path = []
        c.env.restore(root)
        preview = c._event_view()
        self.assertIn('HISTORICAL_REJECTION', preview)
        self.assertIn('declared result does not match the operation', preview)
        self.assertEqual(c.event_memory, [])
        self.assertEqual(deliver(c), c._green_prompt())
        c._execute(GOOD[0], 'GREEN')
        self.assertNotIn('HISTORICAL_REJECTION', c._event_view())
        self.assertEqual(c.nodes[0]['receipt']['action'], action)

    def test_repeat_summary_uses_observed_counts_without_banning_action(self):
        c = initialized('1 1 4 6')
        c._execute('1 + 1 = 99', 'GREEN')
        self.assertNotIn('REPEATED_REJECTION', c._event_view())
        c._execute('1 + 1 = 99', 'GREEN')
        view = c._event_view()
        self.assertIn('REPEATED_REJECTION', view)
        self.assertIn('recorded occurrences=2', view)
        self.assertIn('not a ban or an evaluated alternative action', view)
        self.assertNotIn(GOOD[-1], view)

    def test_same_values_with_different_executed_provenance_do_not_match(self):
        c = initialized('2 2 3 4')
        root = c.env.checkpoint()
        c._execute('2 + 2 = 4', 'GREEN')
        c._execute('4 + 3 = 99', 'GREEN')
        c.archived_paths.append(c.active_path[:])
        c.active_path = []
        c.env.restore(root)
        c._execute('2 * 2 = 4', 'GREEN')
        self.assertNotIn('HISTORICAL_REJECTION', c._event_view())

    def test_checkpoint_records_only_operations_in_its_own_prefix(self):
        c, requests = controller(GOOD[:3]+['answer: 4 * 6 = 24', GOOD[-1]],
            checkpoint='ACTION 3', settings=replace(SHORT_SETTINGS, yellow_entries=0))
        self.assertTrue(c.run(0, '1 1 4 6').success)
        prompt = next(q['prompt'] for q in requests if q['role'] == 'red_checkpoint')
        candidate0 = prompt.split('ACTION 0:', 1)[1].split('ACTION 1:', 1)[0]
        candidate1 = prompt.split('ACTION 1:', 1)[1].split('ACTION 2:', 1)[0]
        self.assertNotIn('Recorded operation', candidate0)
        self.assertIn('1 [constant.1], 1 [constant.2]', candidate0)
        self.assertIn('Recorded operation 1: 1 - 1 = 0', candidate1)
        self.assertNotIn('Recorded operation 2:', candidate1)
        self.assertNotIn(GOOD[-1], prompt)

    def test_yellow_advice_waits_in_ett_until_the_failed_window_boundary(self):
        note = 'UNVERIFIED_EARLIER_CHOICE_NEEDS_REVISION'
        c, requests = controller(['1 + 1 = 99', 'BACKTRACK']+['1 + 1 = 99']*3+[GOOD[-1]],
            diagnosis=note, settings=replace(SHORT_SETTINGS, yellow_entries=1))
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        prompt = next(q['prompt'] for q in requests if q['role'] == 'red_reflection')
        self.assertIn('Public local repair outcome: PREFIX_REVISION_ADVICE_RECORDED', prompt)
        self.assertIn('Local repair evidence status: NONE', prompt)
        self.assertIn('Yellow model advice (unverified): '+note, prompt)
        self.assertEqual(result.forward_actions, 4)
        self.assertEqual(result.recovery_actions, 1)
        self.assertEqual(result.red_attempts, 1)

    def test_without_event_subtree_suppresses_new_handoffs_too(self):
        note = 'UNVERIFIED_LOCAL_NOTE'
        c, requests = controller(['1 + 1 = 99', *GOOD], diagnosis=note,
            settings=replace(SHORT_SETTINGS, initial_actions=5, ablation='without_event_subtree'))
        self.assertTrue(c.run(0, '1 1 4 6').success)
        self.assertTrue(c.yellow_incidents)
        self.assertFalse(c.event_memory)
        greens = [q['prompt'] for q in requests if q['role'] == 'green']
        self.assertTrue(all('[EVENT_SUBTREE]' not in q for q in greens))
        self.assertTrue(all(note in q for q in greens[1:]))


if __name__ == '__main__':
    unittest.main()
