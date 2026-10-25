"""Public occurrence counts and saved evidence must retain their actual scope."""
from dataclasses import replace
import unittest

from test_game24.game24_env import Game24Environment
from test_game24.star_controller import Game24STARController
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller


class OccurrenceScopeTests(unittest.TestCase):
    def test_original_multiplicities_are_public_input_facts(self):
        c = Game24STARController(None)
        c.env = Game24Environment('1 1 4 6')
        before = c.env.checkpoint()
        text = c._operand_record()
        self.assertIn('Required ORIGINAL occurrences in a final expression: 1 twice, 4 once, 6 once.', text)
        self.assertEqual(c.env.checkpoint(), before)
        self.assertNotIn('answer:', text)

    def test_equivalent_native_numbers_are_counted_as_occurrences(self):
        c = Game24STARController(None)
        c.env = Game24Environment('1 1.0 4 6')
        self.assertIn('1 twice, 4 once, 6 once.', c._operand_record())

    def test_four_equal_inputs_are_not_treated_as_one_occurrence(self):
        c = Game24STARController(None)
        c.env = Game24Environment('6 6 6 6')
        self.assertIn('6 four times.', c._operand_record())

    def test_checkpoint_record_does_not_claim_to_be_the_live_pool(self):
        c = Game24STARController(None)
        c.env = Game24Environment('1 1 4 6')
        saved = c.env.checkpoint()
        c.env.execute(GOOD[0])
        text = c._operand_record(checkpoint=saved)
        self.assertIn('Saved checkpoint operand occurrences: 1 [constant.1], 1 [constant.2]', text)
        self.assertNotIn('Current operand occurrences:', text)
        self.assertIn('Current operand occurrences: 4 [constant.3], 6 [constant.4], 0 [operation.1]',
                      c._operand_record())

    def test_persistent_recovery_memory_labels_its_old_snapshot(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        c, requests = controller(BAD + GOOD, settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        greens = [row for row in requests if row['role'] == 'green']
        after_first_recovery_action = greens[5]['prompt']
        self.assertIn('Saved-state snapshot at recovery entry (historical after further actions)',
                      after_first_recovery_action)
        self.assertIn('Saved checkpoint operand occurrences: 1 [constant.1], 1 [constant.2]',
                      after_first_recovery_action)
        self.assertEqual(after_first_recovery_action.count('Current operand occurrences:'), 1)
        self.assertIn('Current operand occurrences: 4 [constant.3], 6 [constant.4], 0 [operation.1]',
                      after_first_recovery_action)
        self.assertEqual((result.forward_actions, result.recovery_actions), (4, 4))
        self.assertEqual(result.model_requests, 10)
        self.assertTrue(all('parameter_language' not in row and not row['commands'] for row in greens))

    def test_yellow_receives_counts_but_does_not_restore_or_get_free_actions(self):
        cfg = replace(SHORT_SETTINGS, initial_actions=2, ablation='without_red')
        c, requests = controller(['1 + 1 = 99', GOOD[0]], settings=cfg)
        result = c.run(0, '1 1 4 6')
        prompt = next(row['prompt'] for row in requests if row['role'] == 'yellow_diagnosis')
        self.assertIn('1 twice, 4 once, 6 once.', prompt)
        self.assertIn('Your judgment is stored in ETT', prompt)
        self.assertEqual(result.forward_actions, 2)
        self.assertEqual(result.red_attempts, 0)

    def test_executed_provenance_never_constructs_a_final_answer(self):
        c = Game24STARController(None)
        c.env = Game24Environment('1 1 4 6')
        for action in GOOD[:3]:
            self.assertTrue(c.env.execute(action).accepted)
        text = c._operand_record()
        self.assertIn('Current operand occurrences: 24 [operation.3]', text)
        self.assertIn('constant.1, constant.2 -> operation.1', text)
        self.assertNotIn('answer:', text)
        self.assertNotIn(GOOD[-1], text)


if __name__ == '__main__':
    unittest.main()
