"""Game24 local repair cannot transfer control to Red before the window boundary."""
from dataclasses import replace
import unittest

from test_game24_migration import controller, GOOD, SHORT_SETTINGS
from test_game24.model_api import CallableModelClient, Generation
from test_game24.star_controller import Game24STARController
from source.experiment_settings import Experiment


class Game24LocalOnlyV3Tests(unittest.TestCase):
    def test_backtrack_never_exits_either_twelve_action_window(self):
        calls = []
        def generate(prompt, **kw):
            calls.append(dict(prompt=prompt, **kw))
            role = kw['role']
            if role == 'red_reflection':
                text = 'An earlier operation may need revision; this is unverified.'
            elif role == 'red_checkpoint':
                text = 'ACTION 0'
            elif role == 'yellow_diagnosis':
                text = 'PREFIX_PLAN_NOT_24; earlier choices may need revision.'
            elif role == 'yellow_action':
                text = 'BACKTRACK'
            else:
                text = '1 + 1 = 99'
            return Generation(text, input_tokens=1, output_tokens=1, queue_seconds=0)
        cfg = Experiment.for_dataset('game24')
        result = Game24STARController(CallableModelClient(generate), settings=cfg).run(0,'1 1 4 6')
        self.assertEqual((result.forward_actions, result.recovery_actions), (12,12))
        self.assertEqual(result.red_attempts, 1)
        self.assertEqual(result.status, 'RECOVERY_WINDOW_EXHAUSTED')
        index = next(i for i,e in enumerate(result.events) if e['event']=='red_preparation_started')
        self.assertEqual(sum(e['event']=='action_executed' for e in result.events[:index]),12)
        self.assertEqual(len(result.ett['local_events']),4)
        self.assertTrue(all(e['outcome']=='PREFIX_REVISION_ADVICE_RECORDED' for e in result.ett['local_events']))
        self.assertFalse(any(n['receipt']['action']=='BACKTRACK' for n in result.ett['nodes']))
        reflection = next(q['prompt'] for q in calls if q['role']=='red_reflection')
        self.assertIn('PREFIX_PLAN_NOT_24',reflection)
        self.assertIn('Receipt nodes 0,1,2,3,4,5,6,7,8,9,10,11; count=12', reflection)
        self.assertEqual(len(result.ett['nodes']),24)

    def test_non24_pool_does_not_proactively_enter_yellow_or_red(self):
        actions = ['1 + 1 = 2','4 + 6 = 10','10 + 2 = 12',GOOD[-1]]
        c, requests = controller(actions)
        result = c.run(0,'1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.yellow_entries,0)
        self.assertEqual(result.red_attempts,0)
        self.assertEqual([q['role'] for q in requests],['green']*4)

    def test_repeated_rejection_summary_keeps_all_node_ids_and_one_literal_action(self):
        cfg = replace(SHORT_SETTINGS, yellow_entries=0)
        bad = '1 + 1 = 99'
        c, requests = controller([bad]*4 + [GOOD[-1]], settings=cfg)
        result = c.run(0,'1 1 4 6')
        self.assertTrue(result.success)
        prompt = next(q['prompt'] for q in requests if q['role']=='red_reflection')
        actual = prompt.split('[FAILED_BRANCH_EVENT_SUBTREE]',1)[1].split('Original input occurrences:',1)[0]
        self.assertEqual(actual.count(bad),1)
        self.assertIn('Receipt nodes 0,1,2,3; count=4',actual)
        self.assertEqual(len(result.ett['nodes']),5)

    def test_restore_handoff_uses_actual_checkpoint_and_does_not_execute_red_advice(self):
        c, requests = controller(['1 - 1 = 0','4 + 6 = 10','10 + 0 = 10',
            'answer: (1 - 1) + (4 + 6) = 10',GOOD[-1]],checkpoint='ACTION 1',
            settings=replace(SHORT_SETTINGS,yellow_entries=0))
        result = c.run(0,'1 1 4 6')
        self.assertTrue(result.success)
        recovered = [q['prompt'] for q in requests if q['role']=='green'][4]
        self.assertIn('Actual restored CHECKPOINT 1:',recovered)
        self.assertIn('Current operand occurrences: 4 [constant.3], 6 [constant.4], 0 [operation.1]',recovered)
        self.assertEqual(result.recovery_actions,1)
        self.assertEqual(result.ett['nodes'][-1]['receipt']['action'],GOOD[-1])
        selection = next(q['prompt'] for q in requests if q['role']=='red_checkpoint')
        self.assertIn('Output exactly one listed ACTION n.', selection)


if __name__ == '__main__':
    unittest.main()
