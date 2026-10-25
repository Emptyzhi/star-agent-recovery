"""Red's second call selects an actual action and produces no second reflection."""
import unittest
from dataclasses import replace
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller
from test_game24.model_api import CallableModelClient, Generation
from test_game24.star_controller import Game24STARController
from source.experiment_settings import Experiment


class ReviewV4Tests(unittest.TestCase):
    def run_reply(self, reply):
        actions = iter(BAD+GOOD)
        requests = []
        def generate(prompt, **kw):
            requests.append(dict(prompt=prompt, **kw))
            if kw['role'] == 'red_reflection': text = 'The original proposal is unverified.'
            elif kw['role'] == 'red_checkpoint': text = reply
            else: text = next(actions)
            return Generation(text, input_tokens=1, output_tokens=1, queue_seconds=0)
        result = Game24STARController(CallableModelClient(generate), settings=SHORT_SETTINGS).run(0, '1 1 4 6')
        return result, requests

    def test_actual_action_trie_and_first_reflection_memory(self):
        result, requests = self.run_reply('ACTION 0')
        self.assertTrue(result.success)
        selection = next(q for q in requests if q['role'] == 'red_checkpoint')
        self.assertEqual(selection['commands'], tuple('ACTION '+str(i) for i in range(4)))
        self.assertEqual(selection['max_new_tokens'], 100)
        green = [q['prompt'] for q in requests if q['role'] == 'green'][4]
        self.assertIn('The original proposal is unverified.', green)
        self.assertNotIn('Red checkpoint review', green)
        self.assertNotIn('REVIEW:', selection['prompt'])

    def test_invalid_reply_is_protocol_failure_without_root_fallback(self):
        for reply in ('ACTION 999', 'ACTION 0\nREVIEW: extra', '', 'ACTION 0 ', 'CHECKPOINT 0'):
            with self.subTest(reply=reply):
                result, _ = self.run_reply(reply)
                self.assertEqual(result.status, 'PROTOCOL_ERROR')
                self.assertEqual(result.cost['checkpoint_restores'], 0)

    def test_real_root_selection_preserves_twelve_action_recovery(self):
        settings = replace(Experiment.for_dataset('game24'), ablation='without_yellow')
        model, _ = controller(['1 + 1 = 99']*24, checkpoint='ACTION 0', settings=settings)
        result = model.run(0, '1 1 4 6')
        self.assertEqual((result.forward_actions, result.recovery_actions), (12, 12))
        self.assertEqual(result.cost['checkpoint_restores'], 1)

    def test_empty_reflection_stops_before_selection(self):
        actions = iter(BAD)
        def generate(prompt, **kw):
            self.assertNotEqual(kw['role'], 'red_checkpoint')
            return Generation('' if kw['role'] == 'red_reflection' else next(actions))
        result = Game24STARController(CallableModelClient(generate), settings=SHORT_SETTINGS).run(0, '1 1 4 6')
        self.assertEqual(result.status, 'PROTOCOL_ERROR')
        self.assertEqual(result.cost['checkpoint_restores'], 0)
