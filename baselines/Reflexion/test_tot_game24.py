import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from tot_game24 import load_tot,execute_tot,PACKAGE


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return messages[0]['content']


class ScriptedModel:
    sampling={'temperature':.1}
    def __init__(self):
        # One candidate at each level; the next three calls evaluate it.
        self.texts=iter([s for candidate in ['1 * 1 = 1 (left: 4 6 1)',
            '4 * 6 = 24 (left: 1 24)','24 * 1 = 24 (left: 24)',
            'Answer: (1 * 1) * (4 * 6) = 24'] for s in [candidate,'sure','sure','sure']])
        self.caps=[]
    def generate(self,prompt,cap):
        self.caps.append(cap)
        return {'raw_output':next(self.texts)}


class NativeToT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.author=load_tot(PACKAGE/'vendor/ga/test_game24/24.csv')
    def test_native_bfs_and_evaluator_with_isolated_cache(self):
        model=ScriptedModel()
        result=execute_tot(0,self.author,Tokenizer(),model)
        self.assertTrue(result['success'])
        self.assertEqual(result['candidate_rewards'],[1])
        self.assertEqual(len(result['search']['steps']),4)
        self.assertEqual(model.caps,[1000]*16)
        self.assertEqual(self.author[1].value_cache,{})


if __name__=='__main__':
    unittest.main()
