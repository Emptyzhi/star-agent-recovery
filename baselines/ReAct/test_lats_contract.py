"""Execute original search functions with scripted environment/model fixtures."""
import ast
import copy
from functools import partial
import logging
from pathlib import Path
from types import ModuleType,SimpleNamespace
import unittest
import numpy as np

from author_binding import bind_module


def author_core():
    path=Path(__file__).resolve().parent/'vendor/lats/webshop/lats.py'
    source=ast.parse(path.read_text())
    module=ModuleType('lats_test_author')
    module.__dict__.update(copy=copy,partial=partial,np=np,logging=logging)
    functions=ast.Module(body=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[])
    exec(compile(functions,str(path),'exec'),module.__dict__)
    return module


class Env:
    def __init__(self):self.sessions={}
    def clone_state(self):return copy.deepcopy(self.sessions)
    def step(self,idx,action):
        self.sessions[idx]={'action':action}
        return ('Task' if action=='reset' else 'Score:1',0 if action=='reset' else 1,action!='reset')


class Task:
    def __init__(self):self.value_cache={}
    def cot_prompt_wrap(self,*args):return 'ACT'
    def value_prompt_wrap(self,*args):return 'VALUE'
    def value_outputs_unwrap(self,values):return 1


class LatsContracts(unittest.TestCase):
    def setUp(self):
        self.args=SimpleNamespace(backend='resident',temperature=.1,log=None,n_generate_sample=5,
            n_evaluate_sample=1,prompt_sample='cot')
    def bound(self):
        def gpt(prompt,n=1,**kwargs):return ['click[Buy Now]']*n
        return bind_module(author_core(),gpt=gpt,env=Env())
    def test_complete_author_search_retains_terminal_reward(self):
        result=self.bound().lats_search(self.args,Task(),'fixed_0',iterations=30,to_print=False)
        self.assertEqual(len(result),4)
        self.assertEqual(result[2],1)
    def test_exhausted_tree_reaches_existing_break_instead_of_infinite_loop(self):
        core=self.bound();core.select_node=lambda node:None
        self.assertEqual(core.lats_search(self.args,Task(),'fixed_0',iterations=30,to_print=False)[2],0)
    def test_early_success_has_the_same_four_field_return_contract(self):
        core=self.bound();winner=core.Node(None,'Task');winner.is_terminal=True;winner.reward=1
        core.select_node=lambda node:winner
        self.assertEqual(len(core.lats_search(self.args,Task(),'fixed_0',iterations=30,to_print=False)),4)
    def test_action_parser_preserves_colon_inside_query(self):
        core=self.bound();core.failed_trajectories=[]
        core.get_samples=lambda *args,**kwargs:['Action: search[clock 12:30]']
        root=core.Node(None,'Task',env_state={})
        nodes=core.generate_new_states(root,self.args,Task(),'fixed_0',1)
        self.assertEqual(nodes[0].state['action'],'search[clock 12:30]')


class NativeImports(unittest.TestCase):
    def test_full_author_modules_import_without_network_or_model_loading(self):
        from lats_webshop import load_lats
        core,task=load_lats()
        self.assertTrue(callable(core.lats_search))
        self.assertTrue(callable(task.WebShopTask))
        self.assertIsNone(task.tokenizer)
        first=bind_module(task,gpt=lambda *a,**k:['first'],tokenizer=SimpleNamespace(encode=lambda x:[]))
        second=bind_module(task,gpt=lambda *a,**k:['second'],tokenizer=SimpleNamespace(encode=lambda x:[]))
        a,b=first.WebShopTask(),second.WebShopTask()
        a.value_cache['one']=1
        self.assertEqual(b.value_cache,{})
        evidence=[dict(trajectory='failed trajectory',r=.5)]
        self.assertEqual(a.generate_self_reflection(evidence,'task')[0]['reflection'],'first')
        self.assertEqual(b.generate_self_reflection(evidence,'task')[0]['reflection'],'second')


if __name__=='__main__':unittest.main()
