from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import webshop_methods


class NativeEnvironment:
    reward = 0.5
    def __init__(self, website):
        self.sessions = {}
    def webshop_text(self, **state):
        raise AssertionError('not used in the controlled controller test')
    def step(self, session, action):
        return 'end', self.reward, True


class Model:
    pass


class ReflectionDepthTests(unittest.TestCase):
    def execute(self, reward, record_costs=False):
        NativeEnvironment.reward = reward
        reflections = []
        def bind(author, webshopEnv):
            def trial(path, world, index, configs, memory, model, website):
                _, actual, done = webshopEnv(website).step('fixed_0', 'click[Buy Now]')
                configs[0]['reward'] = max(configs[0]['reward'], actual)
                configs[0]['is_success'] = actual == 1
                return configs
            def reflect(path, configs, model):
                reflections.append(path)
                configs[0]['memory'].append('Observed failure reflection')
                return configs
            return SimpleNamespace(run_trial=trial, update_memory=reflect)
        with tempfile.TemporaryDirectory() as folder, patch.object(webshop_methods, 'bind_module', bind):
            transport=SimpleNamespace(role='actor',trial=0)
            transport.stage_costs=lambda calls:dict(input_tokens=10*(transport.trial+1)+100*len(reflections),environment_calls=calls)
            template=dict(website='http://site',max_reflections=4)
            if record_costs:template['record_stage_costs']=True
            result = webshop_methods.execute('reflexion', 0,
                template,
                dict(reflexion=SimpleNamespace(Local_llm=Model, webshopEnv=NativeEnvironment)),
                None, transport, Path(folder))
        return result, reflections

    def test_four_reflections_and_five_attempts_without_unused_final_reflection(self):
        result, reflections = self.execute(0.5)
        self.assertEqual(len(reflections), 4)
        self.assertEqual(len(result['trials']), 5)
        self.assertEqual([s['recovery_count'] for s in result['recovery_stages']], list(range(5)))
        self.assertEqual([s['best_purchase_reward'] for s in result['recovery_stages']], [0.5]*5)

    def test_initial_success_has_no_reflection(self):
        result, reflections = self.execute(1)
        self.assertEqual(reflections, [])
        self.assertEqual(len(result['trials']), 1)

    def test_cost_snapshot_precedes_the_next_reflection(self):
        result,_=self.execute(.5,True)
        self.assertEqual([s['cost']['input_tokens'] for s in result['recovery_stages']],
                         [10,120,230,340,450])


if __name__ == '__main__':
    unittest.main()
