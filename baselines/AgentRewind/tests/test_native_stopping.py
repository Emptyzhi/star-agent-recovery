"""Explicit failure stopping and ordinary-tool availability after candidate listing."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_adaptation import AdaptationTests, CounterEnvironment, ScriptedModel
from agentrewind_bench.controller import Controller
from agentrewind_bench.rejection_context import LEDGER_PREFIX
from agentrewind_bench import author
from replay.backtrack import find_duplicate_memory_overlap


class NativeStoppingTests(unittest.TestCase):
    def settings(self, **overrides):
        base = AdaptationTests().config(extra=['--history-format','dialogue',
            '--recovery-format','lines','--notebook-item-char-limit','4000',
            '--rejected-history-keep','2','--max-recoveries','unlimited',
            '--max-terminal-failures','5','--max-reflection-rejections','5'])
        return replace(base, **overrides)

    def run_case(self, responses, env=None, settings=None):
        model = ScriptedModel(responses)
        with tempfile.TemporaryDirectory() as directory:
            result = Controller(env or CounterEnvironment(), model, settings or self.settings(),
                directory, 'native_stopping').run()
            events = [json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
        return model, result, events

    def test_cli_propagation_and_invalid_limits(self):
        settings = self.settings()
        self.assertEqual(settings.as_dict()['adaptation_protocol'], 'native_act_only_v21_native_failure_stop_1200')
        self.assertEqual((settings.max_terminal_failures, settings.max_reflection_rejections), (5,5))
        self.assertIsNone(settings.as_dict()['model_request_limit'])
        for key in ('max_terminal_failures','max_reflection_rejections'):
            for value in (0,-1,True,1.5):
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    replace(settings, **{key:value})

    def test_fifth_author_rejection_stops_without_sixth_generation(self):
        bad = 'backtrack_commit[rec_000001]\nThe task is not feasible.'
        model, result, events = self.run_case(['bad','backtrack_candidates[Observed failure]',*[bad]*5])
        self.assertEqual(result['status'], 'REFLECTION_REJECTION_LIMIT_REACHED')
        self.assertEqual(result['consecutive_reflection_rejections'], 5)
        self.assertEqual(result['cost']['model_requests'], 7)
        self.assertEqual((result['executed_actions'], result['completed_recoveries']), (1,0))
        self.assertEqual(sum(e['event']=='author_tool_rejected' for e in events), 5)
        self.assertEqual(result['cost']['input_tokens'], 7*31)
        self.assertEqual(result['cost']['output_tokens'], 7*7)
        self.assertIn('The task is not feasible.', model.requests[-1][-2]['content'])

    def test_literal_malformed_commit_rejections_are_charged_and_bounded(self):
        _, result, _ = self.run_case(['bad','backtrack_candidates[failure]',
            *['backtrack_commit[rec_000001]']*5])
        self.assertEqual(result['status'], 'REFLECTION_REJECTION_LIMIT_REACHED')
        self.assertEqual(len(result['protocol_rejections']), 5)
        self.assertEqual(result['cost']['model_requests'], 7)

    def test_native_action_after_rejected_commit_is_executed_and_resets_counter(self):
        bad = 'backtrack_commit[rec_000001]\nThe task is not feasible.'
        _, result, events = self.run_case(['bad','backtrack_candidates[failure]',bad,'good'])
        self.assertTrue(result['success'])
        self.assertEqual(result['consecutive_reflection_rejections'], 0)
        self.assertEqual(result['cost']['model_requests'], 4)
        self.assertEqual((result['executed_actions'],result['completed_recoveries']), (2,0))
        self.assertEqual(sum(e['event']=='rejected_history_phase_finished' for e in events), 1)

    def test_fresh_candidate_listing_keeps_rejection_streak_and_compacted_evidence(self):
        bad = 'backtrack_commit[rec_000001]\nThe task is not feasible.'
        model, result, events = self.run_case(['bad','backtrack_candidates[failure]',*[bad]*4,
            'backtrack_candidates[Reconsider the checkpoint]',bad])
        self.assertEqual(result['status'], 'REFLECTION_REJECTION_LIMIT_REACHED')
        self.assertEqual(result['cost']['model_requests'], 8)
        self.assertEqual(result['consecutive_reflection_rejections'], 5)
        self.assertTrue(any(m['content'].startswith(LEDGER_PREFIX) for m in model.requests[-1]))
        self.assertEqual(sum(e['event']=='rejected_history_phase_started' for e in events), 2)

    def test_successful_restore_resets_rejection_counter(self):
        bad = 'backtrack_commit[rec_000001]\nThe task is not feasible.'
        _, result, _ = self.run_case(['bad','backtrack_candidates[failure]',*[bad]*4,
            'backtrack_commit[rec_000001]\nChoose good after inspecting the counter.','good'])
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'], 1)
        self.assertEqual(result['consecutive_reflection_rejections'], 0)
        self.assertEqual(result['cost']['model_requests'], 8)

    def test_pending_native_progress_keeps_history_valid_for_later_rejections(self):
        bad = 'backtrack_commit[rec_000001]\nThe task is not feasible.'
        _, result, events = self.run_case(['bad','backtrack_candidates[failure]',*[bad]*3,
            'bad',*[bad]*4,'good'])
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'], 11)
        self.assertEqual(result['consecutive_reflection_rejections'], 0)
        self.assertEqual(result['executed_actions'], 3)
        phases = [e for e in events if e['event']=='rejected_history_phase_started']
        self.assertIn('continuation_record_uid', phases[1]['stage'])

    def test_fifth_failed_purchase_stops_and_preserves_maximum_actual_reward(self):
        class Purchases(CounterEnvironment):
            def __init__(self):
                super().__init__()
                self.last_reward = None
            def observe(self):
                return dict(observation='Purchase receipt',done=self.last_reward is not None,
                    accepted=True,reward=self.last_reward,state={})
            def execute(self, action):
                self.actions += 1
                self.last_reward = float(action)
                return dict(self.observe(),executed_action=action)
            def checkpoint(self):
                self.snapshots.append(self.last_reward)
                return len(self.snapshots)-1
            def restore(self, key):
                self.last_reward = self.snapshots[key]
                return dict(restored=True)
        rewards = ['.25','.75','.5','0','.25']
        notes = ['Inspect the red option before purchase.', 'Compare the required size against the listing.',
                 'Check price against the task budget.', 'Review material information on the product page.']
        responses = []
        for i, reward in enumerate(rewards):
            responses.append(reward)
            if i < 4:
                responses.append('backtrack_commit[rec_000001]\n'+notes[i])
        _, result, events = self.run_case(responses, Purchases())
        self.assertEqual(result['status'], 'TERMINAL_FAILURE_LIMIT_REACHED')
        self.assertEqual(result['failed_terminal_submissions'], 5)
        self.assertEqual(result['completed_recoveries'], 4)
        self.assertEqual(result['benchmark_reward'], .75)
        self.assertEqual(result['final_public_reward'], .25)
        self.assertEqual(result['cost']['model_requests'], 9)
        self.assertEqual(sum(e['event']=='native_terminal_evaluated' for e in events), 5)

    def test_action_window_is_not_counted_as_terminal_submission(self):
        _, result, _ = self.run_case(['bad','backtrack_candidates[window exhausted]',
            'backtrack_commit[rec_000001]\nChoose good after restoring.','good'],
            settings=self.settings(initial_actions=1,recovery_actions=1))
        self.assertTrue(result['success'])
        self.assertEqual(result['failed_terminal_submissions'], 0)

    def test_author_duplicate_validator_still_rejects_identical_memory(self):
        memory = 'Inspect the current page and choose a different product before buying.'
        self.assertEqual(find_duplicate_memory_overlap(memory, [{'memory_summary':memory}]), 1)
        self.assertIn('treat that branch as commit-only', author.instruction('lines'))


if __name__=='__main__':
    unittest.main()
