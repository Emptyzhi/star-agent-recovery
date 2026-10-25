"""Regression scenarios from the frozen v24 trajectory audit; no solver."""
from dataclasses import replace
import unittest

from test_game24_migration import controller, GOOD, BAD, SHORT_SETTINGS
from test_game24.game24_env import Game24Environment


class ModuleV25Tests(unittest.TestCase):
    def test_unresolved_suffix_not_dumped_after_reexecuting_prefix(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        c, requests = controller(BAD + BAD[:2] + [GOOD[-1]], settings=cfg)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        greens = [q['prompt'] for q in requests if q['role'] == 'green']
        self.assertIn('HISTORICAL_REJECTION', greens[4])
        self.assertIn('HISTORICAL_REJECTION', greens[5])
        self.assertNotIn('Recorded continuation nodes', ''.join(greens))
        self.assertEqual(c.nodes[0]['before_state_key'], c.nodes[4]['before_state_key'])
        self.assertEqual(c.nodes[1]['before_state_key'], c.nodes[5]['before_state_key'])
        self.assertNotEqual(c.nodes[1]['parent_id'], c.nodes[5]['parent_id'])

    def test_same_pool_with_different_executed_provenance_does_not_match(self):
        c, _ = controller([])
        c.env = Game24Environment('2 2 4 6')
        c.env.execute('2 + 2 = 4')
        first = c._branch_state_key()
        first_public = c.env.public_state()
        c.env.reset()
        c.env.execute('2 * 2 = 4')
        self.assertEqual(first_public['numbers'], c.env.public_state()['numbers'])
        self.assertEqual(first_public['sources'], c.env.public_state()['sources'])
        self.assertNotEqual(first, c._branch_state_key())

    def test_rejected_step_does_not_change_operand_provenance(self):
        c, _ = controller([])
        c.env = Game24Environment('1 1 4 6')
        c.env.execute('1 - 1 = 0')
        first = c._branch_state_key()
        c.env.reset()
        c.env.execute('1 - 1 = 99')
        c.env.execute('1 - 1 = 0')
        self.assertEqual(first, c._branch_state_key())

    def test_accepted_spacing_variants_match_without_rewriting_original_receipt(self):
        c, _ = controller([])
        c.env = Game24Environment('1 1 4 6')
        c.env.execute('1 - 1 = 0')
        first = c._branch_state_key()
        c.env.reset()
        c.env.execute('1-1=0')
        self.assertEqual(first, c._branch_state_key())
        self.assertEqual(c.env.history[-1].action, '1-1=0')

    def test_unsolved_final_handoff_has_no_extra_event_projection(self):
        c, requests = controller(BAD + GOOD)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        greens = [q['prompt'] for q in requests if q['role'] == 'green']
        self.assertIn('HISTORICAL_REJECTION', greens[4])
        for prompt in greens[5:]:
            self.assertNotIn('[PAST_EVENT_SUBTREE]', prompt)
            self.assertIn('Historical receipt node 3:', prompt)
        self.assertIn('HISTORICAL_REJECTION', greens[-1])
        self.assertIn('FINAL-ACTION STATE: the current pool is 24', greens[-1])
        self.assertEqual(sum(e['event']=='event_subtree_recorded' for e in result.events), 1)

    def test_no_subtree_ablation_suppresses_history_and_retrieval(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_event_subtree')
        c, requests = controller(BAD + GOOD, settings=cfg)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        self.assertEqual(c.event_memory, [])
        self.assertTrue(all('EVENT_SUBTREE]' not in q['prompt'] for q in requests))

    def test_yellow_diagnosis_and_action_receive_remaining_budget_and_final_correction(self):
        diagnosis = 'The answer omits 1. Submit answer: ((8 / 2) * 6) * 1 = 24.'
        cfg = replace(SHORT_SETTINGS, ablation='without_red')
        actions = ['8 / 2 = 4', '4 * 6 = 24', 'answer: (8 / 2) * 6 = 24',
                   'answer: ((8 / 2) * 6) * 1 = 24']
        c, requests = controller(actions, settings=cfg, diagnosis=diagnosis)
        result = c.run(192, '1 2 6 8')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions, 4)
        for q in requests:
            if q['role'].startswith('yellow'):
                self.assertIn('Actions remaining in this window: 1.', q['prompt'])
                self.assertIn('final expression must use every original number exactly once', q['prompt'])
        action = next(q['prompt'] for q in requests if q['role']=='yellow_action')
        self.assertIn(diagnosis, action)
        self.assertIn('rather\nthan replacing it with an intermediate operation', action)

    def test_model_retains_action_choice_even_when_final_submission_would_be_better(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_red')
        actions = ['8 / 2 = 4', '4 * 6 = 24', 'answer: (8 / 2) * 6 = 24', '24 * 1 = 24']
        c, _ = controller(actions, settings=cfg)
        result = c.run(192, '1 2 6 8')
        self.assertFalse(result.success)
        self.assertEqual(result.history[-1]['action'], '24 * 1 = 24')
        event = next(e for e in result.events if e['event']=='yellow_returned_green')
        self.assertEqual(event['progress_evidence'], 'CURRENT_POOL_24_WITHOUT_FINAL_SUBMISSION')
        self.assertEqual(result.forward_actions, 4)

    def test_recovery_yellow_keeps_red_notes_and_exact_grammar_feedback(self):
        c, requests = controller(BAD + ['(1 - 1) + 4 * 6 = 24', GOOD[-1]])
        self.assertTrue(c.run(0, '1 1 4 6').success)
        yellow = [q['prompt'] for q in requests if q['role'].startswith('yellow')]
        self.assertEqual(len(yellow), 2)
        for prompt in yellow:
            self.assertIn('Red recovery reflection', prompt)
            self.assertIn('a fraction literal is written a/b without parentheses', prompt)
            self.assertIn('Not executed (green)', prompt)

    def test_multi_action_yellow_refreshes_budget_and_latest_receipt(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_red', initial_actions=3, yellow_actions=2)
        c, requests = controller(['1 + 1 = 99', '1 + 1 = 98', GOOD[-1]], settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        prompts = [q['prompt'] for q in requests if q['role']=='yellow_action']
        self.assertEqual(len(prompts), 2)
        self.assertIn('Actions remaining in this window: 2.', prompts[0])
        self.assertIn('Actions remaining in this window: 1.', prompts[1])
        self.assertIn('The last proposal was 1 + 1 = 98.', prompts[1])


if __name__ == '__main__':
    unittest.main()
