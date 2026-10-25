"""Actual execution evidence must not become inferred arithmetic advice."""
from dataclasses import replace
import unittest
from test_game24_migration import controller, GOOD, BAD, SHORT_SETTINGS
from test_game24.game24_env import Game24Environment


class EvidenceV24Tests(unittest.TestCase):
    def test_occurrence_error_does_not_ban_equal_values(self):
        env = Game24Environment('6 6 6 6')
        self.assertTrue(env.execute('6 * 6 = 36').accepted)
        self.assertTrue(env.execute('6 * 6 = 36').accepted)
        receipt = env.execute('36 / 6 = 6')
        self.assertFalse(receipt.accepted)
        self.assertIn('operand occurrence unavailable', receipt.detail)
        self.assertIn('equal values are legal', receipt.detail)
        self.assertEqual(receipt.after['numbers'], ['36', '36'])

    def test_valid_prefix_final_error_can_restore_current_checkpoint(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        wrong = 'answer: 1 + 1 + 4 + 6 = 24'
        c, requests = controller(GOOD[:3]+[wrong, GOOD[-1]], checkpoint='ACTION 3', settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual((result.forward_actions, result.recovery_actions), (4, 1))
        reflection = next(q['prompt'] for q in requests if q['role']=='red_reflection')
        self.assertIn('incorrect final expression or a missing final submission', reflection)
        self.assertNotIn('by writing three proposed atomic calculations', reflection)
        self.assertIn('Current operand occurrences: 24 [operation.3]', reflection)
        self.assertIn('constant.1, constant.2 -> operation.1', reflection)
        self.assertIn('ACTION 3: after 3 recorded actions; numbers left: 24',
                      next(q['prompt'] for q in requests if q['role']=='red_checkpoint'))

    def test_subtree_renders_receipt_edges_not_a_computed_answer(self):
        c, requests = controller(GOOD[:3]+['answer: 1 + 1 + 4 + 6 = 24']+GOOD)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        first_recovery = [q['prompt'] for q in requests if q['role']=='green'][4]
        self.assertIn('Recorded pool before: 1 1 4 6; after: 4 6 0', first_recovery)
        self.assertIn('consumed=constant.1,constant.2; produced=operation.1', first_recovery)
        self.assertIn('These receipts do not prescribe or forbid an action', first_recovery)
        self.assertNotIn(GOOD[-1], first_recovery)

    def test_yellow_receives_current_occurrences_without_fabricated_answer(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_red')
        c, requests = controller(['1 + 1 = 99', *GOOD[:3]], settings=cfg)
        c.run(0, '1 1 4 6')
        prompt = next(q['prompt'] for q in requests if q['role']=='yellow_diagnosis')
        self.assertIn('Current operand occurrences: 1 [constant.1], 1 [constant.2]', prompt)
        self.assertNotIn('Recorded operation 1:', prompt)


if __name__ == '__main__':
    unittest.main()
