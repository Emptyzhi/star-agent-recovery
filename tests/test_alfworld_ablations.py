"""Exercise ALFWorld ablations through actual CLI, controller and public receipts."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'source')]
import dispatch_star
import run_star
from dataset_runtime import execute_alf, alf_episode
from experiment_settings import Experiment, ABLATIONS
from alf_rebased_v64 import RebasedEpisode
from alf_runtime import ThreadSafeAlfEpisode
from icr_core import Generation
from test_unified_cli import NativeFixture


def execute(outputs, ablation='full', **settings):
    cfg = replace(Experiment.for_dataset('alfworld'), ablation=ablation, **settings)
    requests, events = [], []
    answers = iter(outputs)
    def generate(request):
        requests.append(request)
        return Generation(request, next(answers), 10, 5, {})
    def emit(event, **data):
        events.append(dict(event=event, **data))
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **kw: list(text.encode())
    tokenizer.decode.side_effect = lambda tokens, **kw: bytes(tokens).decode()
    episode = RebasedEpisode(NativeFixture(), 'game', 'ACT EXAMPLE',
        lambda messages: '\n'.join(m['content'] for m in messages),
        action_limit=cfg.initial_actions, emit=emit)
    args = SimpleNamespace(prompt_limit=32000, context_limit=32768, run_namespace='alf_ablation')
    with patch('dataset_runtime.alf_episode', return_value=episode), \
            patch('dataset_runtime.encode_prompt', return_value={'input_ids': [1]}), ExitStack() as stack:
        result = execute_alf(0, 'game', SimpleNamespace(generate=generate), tokenizer,
            {}, {}, cfg, args, emit, stack)
    return result, requests, events, episode


class AlfAblationTests(unittest.TestCase):
    def test_environment_resets_steps_and_replay_do_not_overlap(self):
        counts = dict(active=0, peak=0)
        count_lock = Lock()
        class ConcurrentFixture(NativeFixture):
            def reset(self):
                with count_lock:
                    counts['active'] += 1
                    counts['peak'] = max(counts['peak'], counts['active'])
                time.sleep(.005)
                result = super().reset()
                with count_lock:
                    counts['active'] -= 1
                return result
        def probe(i):
            episode = ThreadSafeAlfEpisode(ConcurrentFixture(), 'game', 'ACT EXAMPLE',
                lambda ms: '', action_limit=49, emit=lambda *a, **kw: None)
            try:
                episode.reset()
                episode.step('look')
                checkpoint = episode.checkpoint()
                episode.restore(checkpoint)
                self.assertEqual(episode.checkpoint(), checkpoint)
            finally:
                episode.close()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(probe, range(8)))
        self.assertEqual(counts, dict(active=0, peak=1))

    def test_native_environment_registration_is_isolated_before_concurrent_actions(self):
        counts = dict(active=0, peak=0, total=0)
        count_lock = Lock()
        class Provider:
            def init_env(self, batch_size):
                with count_lock:
                    counts['active'] += 1
                    counts['peak'] = max(counts['peak'], counts['active'])
                    counts['total'] += 1
                time.sleep(.005)
                with count_lock:
                    counts['active'] -= 1
                return NativeFixture()
        modules = {'alfworld': Mock(), 'alfworld.agents': Mock(),
            'alfworld.agents.environment': SimpleNamespace(AlfredTWEnv=Provider)}
        with patch.dict(sys.modules, modules), ThreadPoolExecutor(max_workers=8) as pool:
            episodes = list(pool.map(lambda i: alf_episode(
                '/data/pick_and_place-Apple/trial%d/game.tw-pddl'%i, {}, Mock(),
                Experiment.for_dataset('alfworld'), lambda *a, **kw: None), range(8)))
        self.assertEqual(counts, dict(active=0, peak=1, total=8))
        self.assertEqual(len({id(e.raw) for e in episodes}), 8)

    def test_all_modes_reach_cli_from_queue_with_alfworld_budgets(self):
        for mode in ABLATIONS:
            request = dict(run_id='alf_test', package=str(PACKAGE), dataset='alfworld',
                manifest='manifest.json', task_ids=list(range(134)), workers_per_gpu=40,
                endpoints=['http://127.0.0.1:41581', 'http://127.0.0.1:41582'],
                experiment=dict(ablation=mode))
            settings, sampling, capacity = dispatch_star.resolve_request(request, {})
            args = SimpleNamespace(model_path='/models/qwen', dtype='bfloat16',
                parent_job_id='3820768', node='gpu13', results_root=Path('results'))
            command = dispatch_star.worker_command(args, request, 0, list(range(0,134,2)),
                settings, sampling, capacity)
            parsed = run_star.build_parser().parse_args(command[2:])
            actual, _, tasks = run_star.resolve(parsed)
            self.assertEqual(actual.ablation, mode)
            self.assertEqual((actual.initial_actions, actual.recovery_actions,
                actual.yellow_actions, actual.model_request_limit), (50,50,4,None))
            self.assertEqual((len(tasks), parsed.workers_per_gpu), (67,40))

    def test_without_yellow_executes_green_repeat_and_preserves_red(self):
        result, requests, events, _ = execute(['bad command', 'bad command',
            'Take the apple.', 'ACTION 0', 'take apple 1 from table 1',
            'put apple 1 in/on table 2'], 'without_yellow', initial_actions=2)
        self.assertTrue(result['success'])
        self.assertFalse(result['state']['yellow_enabled'])
        self.assertFalse(any(e['event']=='yellow_local_diagnosis' for e in events))
        self.assertEqual(result['replay_resets'], 1)
        self.assertEqual(len(requests), 6)

    def test_without_red_keeps_yellow_and_never_restores(self):
        result, requests, events, _ = execute(['bad command', 'bad command',
            'Take the visible apple.', 'take apple 1 from table 1',
            'put apple 1 in/on table 2'], 'without_red')
        self.assertTrue(result['success'])
        self.assertTrue(any(e['event']=='yellow_local_diagnosis' for e in events))
        self.assertEqual(result['replay_resets'], 0)
        self.assertEqual(result['state']['max_red_model_calls'], 0)
        self.assertEqual(result['state']['max_recovery_actions'], 0)
        self.assertFalse(any(e['event']=='red_failure_reflected' for e in events))
        self.assertEqual(len(requests), 5)

    def test_without_red_neither_reassigns_window_nor_recovers_empty_output(self):
        for output, status in [('look','INITIAL_WINDOW_EXHAUSTED'), ('','INITIAL_EMPTY_ACTOR_OUTPUT')]:
            result, requests, _, _ = execute([output], 'without_red', initial_actions=1)
            self.assertEqual(result['status'], status)
            self.assertFalse(result['success'])
            self.assertEqual(result['replay_resets'], 0)
            self.assertEqual(len(requests), 1)

    def test_no_subtree_keeps_red_memory_and_full_ett_without_projection(self):
        outputs = ['look', 'Take the visible apple.', 'ACTION 0',
            'take apple 1 from table 1', 'put apple 1 in/on table 2']
        for mode in ('full', 'without_event_subtree'):
            result, requests, events, _ = execute(outputs, mode, initial_actions=1)
            self.assertTrue(result['success'])
            self.assertEqual(result['replay_resets'], 1)
            self.assertEqual(result['state']['selected_fault_index'], 0)
            self.assertIn('Take the visible apple.', requests[-2].prompt)
            self.assertIn('You pick up the apple', requests[-1].prompt)
            projected = any('[EVENT_SUBTREE]' in r.prompt for r in requests)
            self.assertEqual(projected, mode=='full')
            self.assertEqual(any(e['event']=='alf_event_subtree_delivered' for e in events), mode=='full')
            self.assertTrue(result['state']['occurrences'])

    def test_no_subtree_keeps_local_yellow_feedback_without_historical_recall(self):
        with patch('alf_recovery_projection.recovery_subtree', side_effect=AssertionError('historical recall')):
            result, requests, events, episode = execute(['bad command', 'bad command',
                'Take the visible apple.', 'take apple 1 from table 1',
                'put apple 1 in/on table 2'], 'without_event_subtree')
        self.assertTrue(result['success'])
        self.assertTrue(any(e['event']=='yellow_local_diagnosis' for e in events))
        self.assertTrue(any('Local diagnosis (model advice): Take the visible apple.' in r.prompt for r in requests))
        self.assertTrue(all('LOCAL_REPAIR_FEEDBACK' not in r.prompt for r in requests))
        self.assertEqual(episode.recovery_event_subtree, dict(events=[], nodes=[]))
        self.assertFalse(any('[EVENT_SUBTREE]' in r.prompt for r in requests))

    def test_ablation_flags_do_not_leak_to_following_full_run(self):
        execute(['look'], 'without_red', initial_actions=1)
        result, requests, _, _ = execute(['look', 'Take the visible apple.', 'ACTION 0',
            'take apple 1 from table 1', 'put apple 1 in/on table 2'], initial_actions=1)
        self.assertTrue(result['success'])
        self.assertEqual(result['replay_resets'], 1)
        self.assertEqual(len(requests), 5)


if __name__ == '__main__':
    unittest.main()
