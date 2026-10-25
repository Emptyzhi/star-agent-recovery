"""Exercise the actual patched author function without loading a model or site."""
import ast
from pathlib import Path
from typing import List,Dict,Tuple
import unittest


SOURCE=Path(__file__).resolve().parent/'vendor/ga/test_webshop/web_reflexion.py'


def author_functions():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    names={'EnvironmentHistory','_get_base_query','webshop_run'}
    extracted=ast.Module(body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names],type_ignores=[])
    namespace=dict(List=List,Dict=Dict,Tuple=Tuple)
    exec(compile(extracted,str(SOURCE),'exec'),namespace)
    return namespace


class Environment:
    def step(self,session,action):
        if action=='reset':return ('Instruction: buy the blue mug',0,False)
        if action=='search[blue mug]':return ('[mug-blue] price $10',0,False)
        if action=='click[mug-blue]':return ('Blue mug [Buy Now]',0,False)
        if action=='click[Buy Now]':return ('Your score: 1',1,True)
        raise AssertionError('invalid test action')


class Actor:
    def __init__(self):
        self.calls=[]
        self.actions=iter(['search[blue mug]','click[mug-blue]','click[Buy Now]'])
    def _generate(self,prompt,stop):
        self.calls.append((prompt,stop))
        return next(self.actions)


class ReflexionMemory(unittest.TestCase):
    def run_trial(self,memory,base='AUTHOR EXAMPLES'):
        ns=author_functions();actor=Actor()
        history,reward=ns['webshop_run']('fixed_0',Environment(),base,memory,actor,to_print=False)
        return actor,history,reward

    def test_memory_reaches_first_and_later_actor_requests_once(self):
        memory='The previous mug was red. Select blue this time.'
        actor,history,reward=self.run_trial([memory])
        self.assertEqual(reward,1)
        self.assertEqual(len(actor.calls),3)
        for prompt,stop in actor.calls:
            self.assertEqual(prompt.count(memory),1)
            self.assertIn('Your memory for the task below:',prompt)
            self.assertIn('Instruction: buy the blue mug',prompt)
            self.assertTrue(prompt.endswith('\nAction:'))
            self.assertEqual(stop,'\n')
        self.assertIn('Action: search[blue mug]',actor.calls[1][0])
        self.assertIn('Observation: [mug-blue] price $10',actor.calls[1][0])

    def test_first_trial_does_not_manufacture_memory(self):
        actor,_,_=self.run_trial([])
        self.assertTrue(all('Your memory for the task below:' not in p for p,_ in actor.calls))

    def test_native_last_three_memory_policy_and_no_6400_character_loss(self):
        memories=['PLAN_OLD','PLAN_A','PLAN_B','PLAN_C']
        actor,_,_=self.run_trial(memories,'X'*7000)
        for prompt,_ in actor.calls:
            self.assertNotIn('PLAN_OLD',prompt)
            for memory in memories[1:]:self.assertEqual(prompt.count(memory),1)
            self.assertTrue(prompt.startswith('X'*7000))


if __name__=='__main__':
    unittest.main()
