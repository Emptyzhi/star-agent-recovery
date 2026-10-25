"""Receipt-only rendering must not alter actions, budgets, or arithmetic decisions."""
from copy import deepcopy
from dataclasses import replace
import unittest

from test_game24.game24_env import Game24Environment
from test_game24.public_evidence import executed_operand_lineage
from test_game24.star_controller import Game24STARController
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller
from test_game24_receipt_projection import initialized


class ExecutedEvidenceTests(unittest.TestCase):
    def test_equal_inputs_keep_distinct_occurrences_and_exact_sources(self):
        env = Game24Environment('1 1 4 6')
        env.execute(GOOD[0])
        before = env.checkpoint()
        view = executed_operand_lineage(env.puzzle, env.history, env.public_state())
        self.assertIn('operation.1: recorded value 0; executed expression (1 - 1); '
                      'original occurrences constant.1, constant.2', view)
        self.assertEqual(env.checkpoint(), before)
        self.assertNotIn('answer:', view)

    def test_nested_lineage_follows_actual_operand_order(self):
        env = Game24Environment('1 1 4 6')
        for action in GOOD[:3]:
            env.execute(action)
        view = executed_operand_lineage(env.puzzle, env.history, env.public_state())
        self.assertIn('executed expression ((4 * 6) + (1 - 1))', view)
        self.assertIn('constant.3, constant.4, constant.1, constant.2', view)
        self.assertNotIn(GOOD[-1], view)

    def test_rejected_candidate_creates_no_lineage(self):
        env = Game24Environment('1 1 4 6')
        before = executed_operand_lineage(env.puzzle, env.history, env.public_state())
        env.execute('1 + 1 = 99')
        self.assertEqual(executed_operand_lineage(env.puzzle, env.history, env.public_state()), before)
        self.assertNotIn('99', before)

    def test_fraction_is_only_a_recorded_value_not_rounded_or_solved(self):
        env = Game24Environment('1 3 4 6')
        self.assertTrue(env.execute('1 / 3 = 1/3').accepted)
        view = executed_operand_lineage(env.puzzle, env.history, env.public_state())
        self.assertIn('recorded value 1/3; executed expression (1 / 3)', view)
        self.assertNotIn('0.333', view)

    def test_restore_with_reused_operation_id_has_only_current_branch_lineage(self):
        env = Game24Environment('2 2 3 4')
        root = env.checkpoint()
        env.execute('2 + 2 = 4')
        abandoned = executed_operand_lineage(env.puzzle, env.history, env.public_state())
        env.restore(root)
        env.execute('2 * 2 = 4')
        current = executed_operand_lineage(env.puzzle, env.history, env.public_state())
        self.assertIn('(2 + 2)', abandoned)
        self.assertIn('(2 * 2)', current)
        self.assertNotIn('(2 + 2)', current)

    def test_saved_checkpoint_has_no_future_lineage(self):
        c = Game24STARController(None)
        c.env = Game24Environment('1 1 4 6')
        c.env.execute(GOOD[0])
        saved = c.env.checkpoint()
        c.env.execute(GOOD[1])
        view = c._operand_record(checkpoint=saved)
        self.assertIn('executed expression (1 - 1)', view)
        self.assertNotIn('(4 * 6)', view)
        self.assertNotIn('operation.2:', view)

    def test_corrupted_executed_provenance_is_rejected_without_guessing(self):
        env = Game24Environment('1 1 4 6')
        receipt = env.execute(GOOD[0])
        corrupted = replace(receipt, consumed=('constant.1', 'constant.1'))
        with self.assertRaisesRegex(ValueError, 'reuses an input occurrence'):
            executed_operand_lineage(env.puzzle, [corrupted], env.public_state())

    def test_accepted_receipt_with_no_binary_operation_is_not_repaired(self):
        env = Game24Environment('1 1 4 6')
        receipt = env.execute(GOOD[0])
        corrupted = replace(receipt, action='explanation without arithmetic')
        with self.assertRaisesRegex(ValueError, 'lacks its binary provenance'):
            executed_operand_lineage(env.puzzle, [corrupted], env.public_state())

    def test_yellow_feedback_precedes_fallible_advice_and_records_later_repetition(self):
        note = 'UNVERIFIED_NOTE'
        cfg = replace(SHORT_SETTINGS, initial_actions=4, ablation='without_red', yellow_entries=1)
        c, requests = controller(['1 + 1 = 99'] * 4, settings=cfg, diagnosis=note)
        result = c.run(0, '1 1 4 6')
        event = c.yellow_incidents[0]
        before = deepcopy(result.ett)
        view = c._local_event_projection(event)
        self.assertLess(view.index('feedback=declared result'), view.index(note))
        self.assertIn('Later executions of the same repair in the same provenance state: 2,3.', view)
        self.assertEqual(result.ett, before)
        self.assertEqual(result.forward_actions, 4)
        self.assertEqual([q['role'] for q in requests],
                         ['green', 'yellow_diagnosis', 'yellow_action', 'green', 'green'])
        self.assertIn('Linked local repair yellow:1: UNRESOLVED', requests[-1]['prompt'])

    def test_rejection_projection_retains_all_groups_in_latest_observed_order(self):
        c = initialized('1 1 4 6')
        c._execute('1 + 1 = 99', 'GREEN')
        c._execute('1 + 1 = 99', 'GREEN')
        c._execute('4 + 6 = 99', 'GREEN')
        c._execute('4 + 6 = 99', 'GREEN')
        before = deepcopy(c.nodes)
        view = c._event_view()
        self.assertLess(view.index('action="4 + 6 = 99"'), view.index('action="1 + 1 = 99"'))
        self.assertEqual(view.count('recorded occurrences=2'), 2)
        self.assertEqual(c.nodes, before)

    def test_scripted_recovery_keeps_protocol_budget_and_model_call_count(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        c, requests = controller(BAD + GOOD, settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual((result.forward_actions, result.recovery_actions, result.red_attempts), (4, 4, 1))
        self.assertEqual(result.model_requests, 10)
        greens = [q for q in requests if q['role'] == 'green']
        self.assertTrue(all(not q['commands'] and 'parameter_language' not in q for q in greens))
        self.assertEqual(next(q['commands'] for q in requests if q['role'] == 'red_checkpoint'),
                         ('ACTION 0', 'ACTION 1', 'ACTION 2', 'ACTION 3'))

    def test_subtree_ablation_still_suppresses_subtree_but_keeps_public_state(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0, ablation='without_event_subtree')
        c, requests = controller(BAD + GOOD, settings=cfg)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        greens = [q['prompt'] for q in requests if q['role'] == 'green']
        self.assertTrue(all('[EVENT_SUBTREE]' not in q for q in greens))
        self.assertNotIn('[EXECUTED_OPERAND_LINEAGE]', greens[0])
        self.assertNotIn('[EXECUTED_OPERAND_LINEAGE]', greens[3])
        self.assertNotIn('[EXECUTED_OPERAND_LINEAGE]', greens[-1])
        self.assertIn('Act 3> 24 / 2 = 12\nObs 3> numbers left: 12', greens[3])
        self.assertIn('Current numbers: 24', greens[-1])


if __name__ == '__main__':
    unittest.main()
