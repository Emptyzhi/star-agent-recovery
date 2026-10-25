"""Fixed-history prompt parity and all-role information-boundary tests."""
from dataclasses import replace
import re
import unittest
from unittest.mock import patch

from test_alfworld_ablations import execute
from test_web_v16 import make_controller
from test_game24_migration import controller, GOOD, SHORT_SETTINGS
from test_game24_migration import BAD
from experiment_settings import Experiment
from test_game24.prompts import DEMONSTRATIONS


FORBIDDEN = (
    'EVENT_SUBTREE', 'LOCAL_REPAIR_HANDOFF', 'HISTORICAL_REJECTION',
    'REPEATED_REJECTION', 'EXECUTED_OPERAND_LINEAGE', 'dependency_node_ids',
    'parent_occurrence_id', 'accepted_state_change_nodes',
    'previous_ineffective_actions_at_current_state', 'commands_added',
    'commands_removed', 'CURRENT_TRANSFER_EVIDENCE', 'RED_PHASE1_EVIDENCE',
    'Focus from the recorded issue:',
    'REPEATED_STATE_ACTION', 'REPEATED_DECISION_WITHOUT_NEW_PUBLIC_EVIDENCE',
    'webshop_semantics',
)


class NativeHistoryContractTests(unittest.TestCase):
    def assert_no_extracted_records(self, prompts):
        for prompt in prompts:
            for marker in FORBIDDEN:
                self.assertNotIn(marker, prompt)
            self.assertNotIn('Continue the original task using the history and current observation.', prompt)

    def test_alf_fixed_initial_and_forward_history_match_native_actor(self):
        actions = ['take apple 1 from table 1', 'put apple 1 in/on table 2']
        full, full_requests, _, _ = execute(actions, 'full')
        plain, plain_requests, _, _ = execute(actions, 'without_event_subtree')
        for original, ablated in zip(full_requests, plain_requests):
            native = re.sub(r'\n+\[EVENT_SUBTREE\].*?Return one next action using the current task and actual receipts\.\n',
                            '\n', original.prompt, flags=re.S)
            self.assertEqual(native, ablated.prompt)
        self.assertEqual(full['success'], plain['success'])
        self.assert_no_extracted_records([r.prompt for r in plain_requests])

    def test_web_fixed_initial_and_forward_history_match_native_actor(self):
        actions = ['search[cup]', 'click[A]', 'click[red]', 'click[Buy Now]']
        full, full_requests, _ = make_controller(actions, Experiment())
        plain, plain_requests, _ = make_controller(actions, Experiment(ablation='without_event_subtree'))
        self.assertEqual(full.run()['status'], plain.run()['status'])
        for original, ablated in zip(full_requests, plain_requests):
            native = re.sub(r'\n+\[EVENT_SUBTREE\].*?(?=\nAction:$)', '\n', original.prompt, flags=re.S)
            self.assertEqual(native, ablated.prompt)
        self.assert_no_extracted_records([r.prompt for r in plain_requests])

    def test_game_fixed_actor_shell_differs_only_in_removed_operand_records(self):
        full, full_requests = controller(GOOD)
        plain, plain_requests = controller(GOOD, settings=replace(SHORT_SETTINGS, ablation='without_event_subtree'))
        full.run(0, '1 1 4 6')
        plain.run(0, '1 1 4 6')
        # Before any final-action instruction, identical executed operations
        # must produce the identical native shell after removing ETT provenance.
        for original, ablated in zip(full_requests[:3], plain_requests[:3]):
            a = original['prompt']
            b = ablated['prompt']
            a = re.sub(r'Original input occurrences:.*?(?=OPERATION STATE:)', '', a, flags=re.S)
            b = re.sub(r'Original numbers:.*?(?=OPERATION STATE:)', '', b, flags=re.S)
            self.assertEqual(a, b)
            self.assertIn(DEMONSTRATIONS, b)
        self.assertIn('Act 1> 1 - 1 = 0\nObs 1> numbers left:', plain_requests[1]['prompt'])
        self.assert_no_extracted_records([r['prompt'] for r in plain_requests])

    def test_alf_all_roles_keep_chronology_once_without_retrieval(self):
        actions = ['bad command', 'bad command', 'LOCAL_NOTE',
                   'take apple 1 from table 1', 'REFLECTION_NOTE', 'ACTION 0',
                   'take apple 1 from table 1', 'put apple 1 in/on table 2']
        with patch('alf_recovery_projection.recovery_subtree', side_effect=AssertionError('retrieval')):
            result, requests, _, _ = execute(actions, 'without_event_subtree', initial_actions=2)
        self.assertTrue(result['success'])
        self.assert_no_extracted_records([r.prompt for r in requests])
        selection = next(r.prompt for r in requests if r.commands and 'ACTION 0' in r.commands)
        self.assertEqual(selection.count('REFLECTION_NOTE'), 1)
        resumed = requests[-1].prompt
        self.assertEqual(resumed.count('LOCAL_NOTE'), 1)
        self.assertIn('historical records, not the current environment state', resumed)
        self.assertIn('Current available commands:', resumed)

    def test_web_all_roles_and_checkpoint_have_no_automatic_fault_mapping(self):
        actions = ['search[cup]', 'click[Next >]', 'LOCAL_NOTE', 'click[A]',
                   'click[Buy Now]', 'REFLECTION_NOTE', 'ACTION 3',
                   'click[red]', 'click[Buy Now]']
        c, requests, _ = make_controller(actions, Experiment(ablation='without_event_subtree'))
        with patch('web_v16_controller.green_events', side_effect=AssertionError('retrieval')), \
             patch('web_v16_controller.public_fault_action_index', side_effect=AssertionError('mapping')):
            self.assertEqual(c.run()['status'], 'SUCCESS')
        self.assert_no_extracted_records([r.prompt for r in requests])
        selection = next(r.prompt for r in requests if r.commands and 'ACTION 0' in r.commands)
        self.assertEqual(selection.count('REFLECTION_NOTE'), 1)
        self.assertEqual(requests[-1].prompt.count('LOCAL_NOTE'), 1)
        self.assertIsNone(c.red_phase1_evidence)

    def test_game_all_roles_keep_repeated_receipts_and_single_raw_advice(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_event_subtree', yellow_entries=1)
        c, requests = controller(['1 + 1 = 99'] * 4 + [GOOD[-1]],
                                 diagnosis='LOCAL_NOTE', settings=cfg)
        with patch.object(c, '_local_event_projection', side_effect=AssertionError('retrieval')):
            self.assertTrue(c.run(0, '1 1 4 6').success)
        self.assert_no_extracted_records([r['prompt'] for r in requests])
        yellow = next(r['prompt'] for r in requests if r['role'] == 'yellow_action')
        self.assertEqual(yellow.count('LOCAL_NOTE'), 1)
        reflection = next(r['prompt'] for r in requests if r['role'] == 'red_reflection')
        self.assertEqual(reflection.count('> 1 + 1 = 99'), 4)
        selection = next(r['prompt'] for r in requests if r['role'] == 'red_checkpoint')
        advice = 'The earlier sum was valid but led to an unsuitable value. Reconsider it.'
        self.assertEqual(selection.count(advice), 1)
        self.assertIn('ACTION 0: 1 + 1 = 99', selection)
        self.assertNotIn('parents=', selection)

    def test_scripted_game_control_budgets_and_candidate_ids_are_unchanged(self):
        records = []
        for mode in ('full', 'without_event_subtree'):
            c, requests = controller(BAD + GOOD, settings=replace(SHORT_SETTINGS, ablation=mode))
            result = c.run(0, '1 1 4 6')
            records.append((result.forward_actions, result.recovery_actions,
                            result.red_attempts, c.restore_count,
                            [(r['role'], r['max_new_tokens'], r.get('commands')) for r in requests]))
        self.assertEqual(records[0], records[1])

    def test_scripted_alf_control_budgets_and_candidate_ids_are_unchanged(self):
        actions = ['bad command', 'bad command', 'LOCAL_NOTE',
                   'take apple 1 from table 1', 'REFLECTION_NOTE', 'ACTION 0',
                   'take apple 1 from table 1', 'put apple 1 in/on table 2']
        records = []
        for mode in ('full', 'without_event_subtree'):
            result, requests, _, _ = execute(actions, mode, initial_actions=2)
            records.append((result['success'], result['replay_resets'],
                            [(r.output_limit, r.commands) for r in requests]))
        self.assertEqual(records[0], records[1])

    def test_context_limit_rejects_without_shortening_chronology(self):
        from icr_core import BudgetExhausted
        c, _, _ = make_controller([], Experiment(ablation='without_event_subtree'))
        record = 'Action: search[cup]\nObservation: ' + 'literal page receipt ' * 1000
        c.plain_history.append(record)
        c.episode.prompt_context_limit = 200
        with self.assertRaises(BudgetExhausted):
            c._bind_scope(c._green_scope(), '')
        self.assertEqual(c.plain_history, [record])

    def test_web_return_to_search_does_not_cut_reflection_history(self):
        import plain_history
        c, requests, _ = make_controller(
            ['search[cup]', 'click[Back to Search]', 'REFLECTION_NOTE', 'ACTION 0',
             'search[cup]', 'click[A]'],
            Experiment(ablation='without_event_subtree', initial_actions=2,
                       recovery_actions=2, yellow_entries=0))
        c.run()
        reflection = next(r.prompt for r in requests if 'New plan:' in r.prompt)
        self.assertIn('Action: search[cup]', reflection)
        self.assertIn('Action: click[Back to Search]', reflection)
        self.assertIn('Buy a red cup below 20 dollars.', reflection)
        # A model diagnosis may itself contain the native extraction marker.
        c.plain_history.append('Local diagnosis (model advice): Instruction: NOTE_WITH_MARKER')
        transcript = plain_history.transcript(c)
        self.assertIn(transcript, c._red_reflection_prompt(transcript))

    def test_local_roles_do_not_repeat_observation_or_expose_repeat_matching(self):
        import plain_history
        c, requests, _ = make_controller(
            ['search[cup]', 'click[Next >]', 'LOCAL_NOTE', 'click[A]',
             'click[Buy Now]', 'REFLECTION_NOTE', 'ACTION 3',
             'click[red]', 'click[Buy Now]'], Experiment(ablation='without_event_subtree'))
        c.run()
        self.assert_no_extracted_records([r.prompt for r in requests])
        context = next(r.prompt for r in requests if r.prompt.endswith('Local diagnosis:'))
        self.assertNotIn('Current observation:', context)
        self.assertNotIn('Trigger:', context)
        # Only an actual restore creates a Current observation: record.
        self.assertEqual(requests[-1].prompt.count('Current observation:'), 1)

    def test_web_full_mapping_is_preserved_and_ablation_requires_model_selection(self):
        full, _, _ = make_controller([], Experiment())
        plain, _, _ = make_controller([], Experiment(ablation='without_event_subtree'))
        mapping = {'action_index': 2, 'source': 'TEST_NATIVE_MAPPING'}
        with patch('web_v16_controller.public_fault_action_index', return_value=mapping) as mapper:
            self.assertEqual(full._red_action_index_mapping([], 'blue mismatch'), mapping)
            mapper.assert_called_once_with([], 'blue mismatch')
            mapper.reset_mock()
            self.assertIsNone(plain._red_action_index_mapping([], 'blue mismatch'))
            mapper.assert_not_called()

    def test_game_diagnosis_marker_cannot_cut_earlier_reflection_history(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_event_subtree', yellow_entries=1)
        c, requests = controller(['1 + 1 = 99'] * 4 + [GOOD[-1]],
                                 diagnosis='Here is the task: LITERAL_NOTE', settings=cfg)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        reflection = next(r['prompt'] for r in requests if r['role'] == 'red_reflection')
        self.assertEqual(reflection.count('> 1 + 1 = 99'), 4)
        self.assertIn('Local diagnosis (model advice): Here is the task: LITERAL_NOTE', reflection)

    def test_game_role_wording_changes_do_not_rewrite_model_advice(self):
        cfg = replace(SHORT_SETTINGS, ablation='without_event_subtree', yellow_entries=1)
        advice = 'inspect their lineage'
        c, requests = controller(['1 + 1 = 99'] * 4 + [GOOD[-1]], diagnosis=advice, settings=cfg)
        self.assertTrue(c.run(0, '1 1 4 6').success)
        for role in ('yellow_action', 'red_reflection', 'red_checkpoint'):
            prompt = next(r['prompt'] for r in requests if r['role'] == role)
            self.assertIn('Local diagnosis (model advice): ' + advice, prompt)
            self.assertNotIn('Local diagnosis (model advice): inspect their recorded operations', prompt)


if __name__ == '__main__':
    unittest.main()
