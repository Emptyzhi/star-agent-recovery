"""Minimal public-event projection; fixtures never call a model or solver."""
from dataclasses import replace
import unittest

from test_game24_migration import controller, GOOD, BAD, SHORT_SETTINGS


def initialized(puzzle):
    c, _ = controller(['']+['unused']*20,
                      settings=replace(SHORT_SETTINGS, ablation='without_red'))
    c.run(0, puzzle)
    c.window_budget = 12  # Offline receipt fixtures, not experiment settings.
    return c


def deliver(c):
    prompt = c._green_prompt()
    c._request(prompt, 'green', 100)
    return prompt


class MinimalReceiptProjectionTests(unittest.TestCase):
    def test_current_rejection_stays_in_plain_history_without_reminder(self):
        c = initialized('3 8 13 13')
        for action in ('3 + 8 = 11', '13 + 11 = 24', '24 + 9 = 33'):
            c._execute(action, 'GREEN')
        prompt = deliver(c)
        self.assertNotIn('[EVENT_SUBTREE]', prompt)
        self.assertEqual(prompt.count('24 + 9 = 33'), 1)
        self.assertIn('operand occurrence unavailable', prompt)
        self.assertEqual(c.event_memory, [])

    def test_task178_obstacle_change_is_delivered_once_not_on_preview(self):
        c = initialized('1 5 7 11')
        for action in ('11 - 7 = 4', '4 * 6 = 24', '5 + 1 = 6'):
            c._execute(action, 'GREEN')
        self.assertIn('OPERAND_OBSTACLE_CHANGED', c._event_view())
        self.assertIn('OPERAND_OBSTACLE_CHANGED', c._event_view())
        prompt = deliver(c)
        self.assertIn('does not certify the old operation', prompt)
        self.assertIn('see Act/Obs 2', prompt)
        self.assertIn('see Act/Obs 3', prompt)
        self.assertNotIn('[EVENT_SUBTREE]', deliver(c))
        self.assertEqual(len(c.event_memory), 1)
        c._execute('4 * 6 = 24', 'GREEN')
        self.assertEqual(c._event_view(), '')
        self.assertIn('FINAL-ACTION STATE: the current pool is 24', c._green_prompt())

    def test_task008_restore_occurrence_counts_without_certifying_arithmetic(self):
        c, requests = controller(['2 * 10 = 20', '10 + 10 = 999',
            '20 + 2 = 22', '10 + 10 = 999']+['']*4,
            settings=replace(SHORT_SETTINGS, yellow_entries=0))
        c.run(8, '2 2 10 10')
        green = [q['prompt'] for q in requests if q['role']=='green'][4]
        self.assertIn('OPERAND_OBSTACLE_CHANGED', green)
        self.assertIn('restored CHECKPOINT 0', green)
        self.assertIn('10 + 10 = 999', green)
        self.assertIn('does not certify', green)
        self.assertEqual(len(c.nodes), 8)

    def test_new_arithmetic_rejection_supersedes_old_missing_operand(self):
        c = initialized('1 5 7 11')
        for action in ('11 - 7 = 4', '4 * 6 = 999', '5 + 1 = 6'):
            c._execute(action, 'GREEN')
        self.assertIn('OPERAND_OBSTACLE_CHANGED', c._event_view())
        receipt, _ = c._execute('4 * 6 = 999', 'YELLOW')
        self.assertFalse(receipt.accepted)
        self.assertEqual(c._event_view(), '')
        self.assertIn(receipt.detail, c._green_prompt())

    def test_restore_without_actual_obstacle_change_does_not_notify(self):
        c = initialized('1 5 7 11')
        for action in ('11 - 7 = 4', '4 * 6 = 24', '5 + 1 = 6'):
            c._execute(action, 'GREEN')
        c.archived_paths = [c.active_path[:]]
        c.pending_events = [dict(kind='RED_HANDOFF', event_key='test:restore',
            checkpoint_index=3, failed_public_state=c.env.public_state())]
        self.assertEqual(c._event_view(), '')

    def test_equal_numeric_values_need_two_occurrences(self):
        c = initialized('2 2 10 10')
        c._execute('2 * 10 = 20', 'GREEN')
        receipt, _ = c._execute('10 + 10 = 20', 'GREEN')
        self.assertFalse(c._operands_present(receipt.to_dict(), c.env.public_state()))
        c._execute('20 / 2 = 10', 'YELLOW')
        self.assertIn('OPERAND_OBSTACLE_CHANGED', c._event_view())

    def test_numeric_fraction_equivalence_checks_only_presence(self):
        c = initialized('1 2 3 4')
        receipt, _ = c._execute('2/2 + 1/2 = 77', 'GREEN')
        c._execute('1 / 2 = 1/2', 'YELLOW')
        self.assertEqual(c._event_view(), '')
        c._execute('4 - 3 = 1', 'GREEN')
        self.assertTrue(c._operands_present(receipt.to_dict(), c.env.public_state()))
        self.assertIn('OPERAND_OBSTACLE_CHANGED', c._event_view())

    def test_only_missing_historical_suffix_persists_along_actual_chain(self):
        chain = ['4 + 4 = 8', '8 - 4 = 4', '6 * 4 = 24']
        bad = 'answer: (4 + 4) * (6 - 4) = 24'
        good = 'answer: ((4 + 4) - 4) * 6 = 24'
        c, requests = controller(chain+[bad]+chain[1:]+[good], checkpoint='ACTION 1')
        self.assertTrue(c.run(38, '4 4 4 6').success)
        green = [q['prompt'] for q in requests if q['role']=='green']
        recovery = green[4]
        self.assertIn('HISTORICAL_POOL_24', recovery)
        self.assertIn('Historical receipt node 1: 8 - 4 = 4', recovery)
        self.assertIn('consumed=operation.1,constant.3', recovery)
        self.assertNotIn('Receipt node 0:', recovery)
        self.assertIn(bad, recovery)
        self.assertIn('independently of the intermediate pool', recovery)
        self.assertNotIn(good, recovery)
        self.assertIn('Historical receipt node 2: 6 * 4 = 24', green[5])
        self.assertNotIn('Historical receipt node 1:', green[5])
        self.assertNotIn('HISTORICAL_POOL_24', green[6])
        self.assertIn('HISTORICAL_REJECTION', green[6])
        self.assertIn(bad, green[6])
        self.assertEqual(len(c.nodes), 7)
        self.assertEqual(len(c.event_memory), 1)

    def test_task158_retains_executed_chain_without_constructing_final(self):
        c, requests = controller(['13 - 5 = 8', '8 * 3 = 24', '12 / 4 = 3',
            '8 * 3 = 24', 'answer: (13 - 5) * (12 / 4) = 24'], checkpoint='ACTION 3')
        self.assertTrue(c.run(158, '4 5 12 13').success)
        greens = [q['prompt'] for q in requests if q['role']=='green']
        self.assertIn('OPERAND_OBSTACLE_CHANGED', greens[-2])
        self.assertIn('HISTORICAL_POOL_24', greens[-1])
        self.assertIn('Historical receipt node 3: 8 * 3 = 24', greens[-1])
        self.assertNotIn('answer: (13 - 5) * (12 / 4) = 24', greens[-1])
        self.assertNotIn('OPERAND_OBSTACLE_CHANGED', greens[-1])

    def test_cropped_suffix_bridges_shifted_operation_ids_after_reexecution(self):
        c, requests = controller(['1 + 1 = 99']+GOOD[:3]+GOOD,
            settings=replace(SHORT_SETTINGS, yellow_entries=0))
        self.assertTrue(c.run(0, '1 1 4 6').success)
        greens = [q['prompt'] for q in requests if q['role']=='green']
        prompt = greens[6]
        self.assertIn('Historical receipt node 3: 24 + 0 = 24', prompt)
        self.assertIn('historical operation.2 -> current operation.1 (value 0)', prompt)
        self.assertIn('historical operation.3 -> current operation.2 (value 24)', prompt)
        self.assertNotIn('Historical receipt node 1:', prompt)
        self.assertNotIn('Historical receipt node 2:', prompt)
        self.assertIn('consumed=operation.3,operation.2; produced=operation.4', prompt)
        self.assertEqual(c.nodes[1]['receipt']['produced'], 'operation.2')
        self.assertEqual(c.nodes[3]['receipt']['consumed'], ['operation.3', 'operation.2'])
        self.assertEqual(c.nodes[4]['receipt']['produced'], 'operation.1')

    def test_historical_chain_retires_on_new_arithmetic_branch(self):
        c, requests = controller(GOOD[:3]+['answer: 1 + 1 + 4 + 6 = 24',
            '1 + 1 = 2', '4 + 6 = 10', '10 + 2 = 12', GOOD[-1]])
        c.run(0, '1 1 4 6')
        green = [q['prompt'] for q in requests if q['role']=='green']
        self.assertIn('HISTORICAL_POOL_24', green[4])
        self.assertNotIn('HISTORICAL_POOL_24', green[5])

    def test_same_numbers_with_different_provenance_do_not_keep_chain(self):
        chain = ['2 + 2 = 4', '4 + 4 = 8', '3 * 8 = 24']
        c, requests = controller(chain+['answer: 2 + 2 + 3 + 4 = 24',
            '2 * 2 = 4']+['']*3, settings=replace(SHORT_SETTINGS, yellow_entries=0))
        c.run(0, '2 2 3 4')
        greens = [q['prompt'] for q in requests if q['role']=='green']
        self.assertIn('HISTORICAL_POOL_24', greens[4])
        self.assertNotIn('HISTORICAL_POOL_24', greens[5])

    def test_current_final_rejection_remains_in_history_without_extra_event(self):
        c = initialized('1 1 4 6')
        for action in GOOD[:3]:
            c._execute(action, 'GREEN')
        c._execute('answer: 4 * 6 = 24', 'GREEN')
        prompt = deliver(c)
        self.assertEqual(c.event_memory, [])
        self.assertNotIn('[EVENT_SUBTREE]', prompt)
        self.assertIn('final expression must use every original number exactly once', prompt)
        self.assertNotIn(GOOD[-1], prompt)
        self.assertEqual(prompt.count('Recorded operation 1:'), 1)

    def test_no_diagnosis_promoted_to_fact(self):
        c, requests = controller(['1 + 1 = 99', '1 - 1 = 99']+GOOD[:2]+GOOD,
            diagnosis='UNVERIFIED_DIAGNOSIS_OMIT_ORIGINAL_INPUTS')
        c.run(0, '1 1 4 6')
        for q in requests:
            if q['role']=='green' and 'UNVERIFIED_DIAGNOSIS_OMIT_ORIGINAL_INPUTS' in q['prompt']:
                self.assertIn('Yellow model advice (unverified): UNVERIFIED_DIAGNOSIS_OMIT_ORIGINAL_INPUTS', q['prompt'])
                self.assertIn('Public local repair outcome: UNRESOLVED', q['prompt'])

    def test_bad_final_recalled_only_at_matching_state_and_raw_ett_stays_complete(self):
        old = 'answer: 1 + 1 + 4 + 6 = 24'
        new = 'answer: 4 * 6 = 24'
        c, requests = controller(GOOD[:3]+[old, new, GOOD[-1]],
            checkpoint='ACTION 3', settings=replace(SHORT_SETTINGS, yellow_entries=0))
        self.assertTrue(c.run(0, '1 1 4 6').success)
        greens = [q['prompt'] for q in requests if q['role']=='green']
        self.assertIn('HISTORICAL_REJECTION', greens[4])
        self.assertIn(old, greens[4])
        self.assertIn(old, greens[5])
        self.assertIn(new, greens[5])
        self.assertEqual(c.nodes[3]['receipt']['action'], old)
        self.assertEqual(c.archived_paths, [[0, 1, 2, 3]])

    def test_no_subtree_has_no_projection_or_delivery(self):
        c, requests = controller(GOOD[:3]+['answer: 4 * 6 = 24']+GOOD,
            settings=replace(SHORT_SETTINGS, ablation='without_event_subtree'))
        self.assertTrue(c.run(0, '1 1 4 6').success)
        self.assertEqual(c.event_facts, {})
        self.assertEqual(c.event_memory, [])
        self.assertTrue(all('EVENT_SUBTREE]' not in q['prompt'] for q in requests))


if __name__ == '__main__':
    unittest.main()
