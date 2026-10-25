import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from concurrent.futures import Future
from unittest.mock import patch
from contextlib import redirect_stdout
import io

from source.lats_webshop import load_lats
from source.round_budget import install_round_search
from source.search_adaptations import AlfWorldActionOnlyTask
from source.score_protocol import recorded_score
from source.search_adaptations import execute_alfworld_actionly
from source.config import SearchConfig
from source.model import Model


class NonfatalModelTests(unittest.TestCase):
    def test_alfworld_mixed_batch_executes_valid_action_only(self):
        executed=[]
        class Pool:
            def submit(self, function, job):
                executed.append(job[0])
                future=Future()
                future.set_result((None,dict(observation='Done',reward=1,terminal=True)))
                return future
        class Transport:
            outputs=iter([' \n','take apple 1 from table 1\n'])
            def generate(self,prompt,cap):
                return dict(raw_output=next(self.outputs))
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            directory=Path(temp)
            initial=dict(observation='task',reward=0,terminal=False)
            world=SimpleNamespace(directory=directory,task_id=0,examples='',
                cache={():initial},gamefile='task',env_config={},horizon=50,
                at=lambda actions:initial,
                record_isolated=lambda actions,receipt:receipt)
            config=SearchConfig(candidates=1,iterations=2,round_actions=50)
            model=Model(None,Transport(),config,.1,directory)
            with patch('source.search_adaptations._ALFWORLD_CANDIDATE_POOL',Pool()):
                result=execute_alfworld_actionly(world,model,config,load_lats())
            events=[json.loads(row) for row in (directory/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(executed,['take apple 1 from table 1'])
        self.assertEqual(events[0]['raw_output'],' \n')

    def test_alfworld_empty_action_is_explicit_skip(self):
        for raw in [' ', '\n', ' \n\n', '> ']:
            self.assertIsNone(AlfWorldActionOnlyTask.parse_action(raw))
        self.assertEqual(AlfWorldActionOnlyTask.parse_action('take apple 1 from table 1'),
                         'take apple 1 from table 1')
        with self.assertRaises(ValueError):
            AlfWorldActionOnlyTask.parse_action(None)

    def test_all_empty_expansions_use_only_normal_iterations(self):
        native, _ = load_lats()
        calls=[]
        core=SimpleNamespace(Node=native.Node,
            env=SimpleNamespace(sessions={},step=lambda *args: ('task',0,False)),
            select_node=lambda root:root,
            expand_node=lambda *args:calls.append(1),
            collect_all_nodes=lambda root:[root])
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            install_round_search(core,50,directory)
            state,value,reward,em=core.lats_search(None,None,0,iterations=2)
            events=[json.loads(row) for row in (directory/'rounds.jsonl').read_text().splitlines()]
        self.assertEqual(calls,[1,1])
        self.assertEqual(reward,0)
        self.assertEqual(state['action'],'')
        self.assertEqual(sum(row['event']=='round_no_progress' for row in events),2)

    def test_unscored_is_below_valid_and_has_original_evidence(self):
        failures=[]
        values=[recorded_score([s],0,10,failures,'test') for s in ['bad','0','10']]
        self.assertEqual(values,[-1,0,1])
        self.assertEqual(values.index(max(values)),2)
        self.assertEqual(failures[0]['outputs'],['bad'])
        self.assertEqual(failures[0]['ranking_value'],-1)
        with self.assertRaises(AttributeError):
            recorded_score([None],0,10,failures,'test')


if __name__=='__main__':
    unittest.main()
