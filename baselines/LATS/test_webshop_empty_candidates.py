from concurrent.futures import Future
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from source.config import SearchConfig
from source.model import Model
from source.lats_webshop import (
    execute_lats, execute_webshop_candidates, load_lats, parse_webshop_action, rollout_webshop,
)


class AuthorEmptyCandidates(unittest.TestCase):
    def test_real_search_stops_empty_rollout_without_executing_empty_actions(self):
        outputs=['search[cup]\n','search[cup]\n','bad score','\n']
        pending=iter(outputs);generation=[];executed=[]
        class Transport:
            def generate(self,prompt,cap):
                generation.append((self.role,cap))
                return dict(raw_output=next(pending))
        class Workers:
            def __init__(self,*args):
                pass
            def execute(self,snapshot,session,action):
                executed.append(action)
                before=copy.deepcopy(snapshot)
                after={session:dict(page_type='init' if action=='reset' else 'search')}
                if action=='reset':
                    receipt=('Purchase a blue cup.',0,False)
                elif action=='search[cup]':
                    receipt=('Available item.',0,False)
                elif action=='click[Buy Now]':
                    receipt=('Your score: 1.0',1,True)
                else:
                    raise AssertionError('unexpected environment action '+repr(action))
                return dict(receipt=receipt,before=before,after=after,
                            invalid=False,server_session=session)
            def close(self):
                pass
        with TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            directory=Path(temporary)
            config=SearchConfig(candidates=1,iterations=1,round_actions=2)
            model=Model(None,Transport(),config,.1,directory)
            with patch('source.lats_webshop.WebShopWorkers',Workers):
                result=execute_lats(0,load_lats(),'http://unused',model,directory,config)
            events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
            raw_samples=[json.loads(line)['raw_output'] for line in (directory/'samples.jsonl').read_text().splitlines()]
        self.assertFalse(result['success'])
        self.assertEqual(result['value_protocol_failures'][0]['outputs'], ['bad score'])
        self.assertEqual(executed,['reset','search[cup]','search[cup]'])
        self.assertEqual(generation,[('candidate',100)]*2+[('evaluation',100)]+[('candidate',100)])
        self.assertEqual(raw_samples,outputs)
        self.assertEqual(sum(e['event']=='rollout_no_progress' for e in events),1)

    def test_empty_candidates_are_skipped_without_environment_calls(self):
        executed=[]
        def submit(ordinal,action):
            executed.append((ordinal,action))
            result=Future();result.set_result(action)
            return result
        outputs=['\n','search[cup]\n','Action: \n','click[item]\n','click[item]\n']
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            rows=execute_webshop_candidates(outputs,submit,directory)
            rejected=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        self.assertEqual(executed,[(1,'search[cup]'),(3,'click[item]'),(4,'click[item]')])
        self.assertEqual([row[0] for row in rows],[1,3,4])
        self.assertEqual([row['raw_output'] for row in rejected],['\n','Action: \n'])
        self.assertEqual([row['ordinal'] for row in rejected],[0,2])

    def test_rollout_empty_batch_returns_without_resampling(self):
        root=SimpleNamespace(depth=4,is_terminal=False,question='task')
        winner=SimpleNamespace(depth=5,is_terminal=True,reward=1)
        trace=[];batches=iter([[],[],[winner]])
        def generate(node,args,task,idx,n):
            self.assertIs(node,root)
            trace.append(n)
            return next(batches)
        bound=SimpleNamespace(iteration_depth_limit=6,generate_new_states=generate)
        args=SimpleNamespace(n_generate_sample=5,n_evaluate_sample=1)
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            result=rollout_webshop(bound,root,args,None,'fixed_0',2,directory)
            events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        self.assertIs(result,root)
        self.assertEqual(trace,[5])
        self.assertEqual(root.depth,4)
        self.assertEqual(events,[dict(event='rollout_no_progress',depth=4,samples=5)])

    def test_multiline_and_incomplete_list_remain_protocol_errors(self):
        with self.assertRaisesRegex(ValueError,'single-line'):
            parse_webshop_action('search[cup]\nclick[item]')
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'generation must finish'):
                execute_webshop_candidates(iter(['search[cup]']),None,Path(temporary))

    def test_generation_errors_propagate_without_resampling(self):
        calls=[]
        def generate(*args):
            calls.append(1)
            raise RuntimeError('HTTP request failed')
        bound=SimpleNamespace(iteration_depth_limit=2,generate_new_states=generate)
        node=SimpleNamespace(depth=0,is_terminal=False,question='task')
        args=SimpleNamespace(n_generate_sample=5,n_evaluate_sample=1)
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError,'HTTP request failed'):
                rollout_webshop(bound,node,args,None,'fixed_0',2,Path(temporary))
        self.assertEqual(calls,[1])


if __name__=='__main__':
    unittest.main()
