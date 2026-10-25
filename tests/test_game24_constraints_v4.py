"""Role isolation and task-scoped rejection evidence, without an answer oracle."""
from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from test_game24_migration import controller, GOOD, SHORT_SETTINGS
from test_game24.action_language import YELLOW_LANGUAGE
from test_game24.model_api import ResidentModelClient


class ConstraintsV4Tests(unittest.TestCase):
    def test_resident_grammar_owns_line_boundary_and_green_stays_plain(self):
        requests=[]
        def generate(request):
            requests.append(request)
            return SimpleNamespace(request=request,text='1 + 1 = 2\n',input_tokens=1,output_tokens=1,
                                   raw_response={'queue_time_seconds':0})
        client=ResidentModelClient(SimpleNamespace(generate=generate),None,
            prompt_limit=32000,context_limit=32768)
        client.count_tokens=lambda prompt:1
        with patch('model_backend.format_model_prompt',side_effect=lambda tokenizer,prompt,mode:prompt):
            client.generate('yellow',role='yellow_action',max_new_tokens=100,stop=('\n',),
                            parameter_language=YELLOW_LANGUAGE)
            client.generate('green',role='green',max_new_tokens=100,stop=('\n',))
            client.generate('red',role='red_checkpoint',max_new_tokens=100,commands=('ACTION 0','ACTION 2'))
        self.assertIsNone(requests[0].stop_mode)
        self.assertEqual(requests[0].parameter_language,YELLOW_LANGUAGE)
        self.assertEqual(requests[1].stop_mode,'first_line')
        self.assertIsNone(requests[1].parameter_language)
        self.assertEqual(requests[2].route,'command_trie')
        self.assertEqual(requests[2].commands,('ACTION 0','ACTION 2'))

    def test_only_yellow_action_requests_arithmetic_language(self):
        model, requests = controller(['1 + 1 = 99', *GOOD],
            settings=replace(SHORT_SETTINGS, ablation='without_red', initial_actions=5))
        self.assertTrue(model.run(0, '1 1 4 6').success)
        for request in requests:
            self.assertEqual(request.get('parameter_language'),
                             YELLOW_LANGUAGE if request['role']=='yellow_action' else None)
            self.assertEqual(request['commands'], ())

    def test_wrong_arithmetic_remains_environment_rejected(self):
        model, _ = controller(['1 + 1 = 99', '1 + 1 = 98', *GOOD],
            settings=replace(SHORT_SETTINGS, initial_actions=6, ablation='without_red'))
        result = model.run(0, '1 1 4 6')
        yellow = [node['receipt'] for node in result.ett['nodes'] if node['scope']=='YELLOW']
        self.assertFalse(yellow[0]['accepted'])
        self.assertEqual(yellow[0]['action'], '1 + 1 = 98')

    def test_final_expression_rejection_visible_after_pool_changes(self):
        wrong = 'answer: 1 + 1 + 4 + 6 = 24'
        model, requests = controller([wrong, '1 - 1 = 0', '4 * 6 = 24', '24 + 0 = 24', GOOD[-1]],
            settings=replace(SHORT_SETTINGS, initial_actions=5, yellow_entries=0, ablation='without_red'))
        self.assertTrue(model.run(0, '1 1 4 6').success)
        # A single rejection is already in ordinary history. On restoration its
        # archived record must also be visible at a different intermediate pool.
        model.archived_paths = [model.active_path[:1]]
        model.active_path = model.active_path[1:]
        projection = model._event_view()
        self.assertIn(wrong, projection)
        self.assertIn('independently of the intermediate pool', projection)
