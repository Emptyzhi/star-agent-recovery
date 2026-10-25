from concurrent.futures import Future,ThreadPoolExecutor
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
import threading
import time
from unittest.mock import patch
from source.candidate_execution import execute_candidates
from source.config import SearchConfig, resolve
from source.model import Model
from source.lats_webshop import load_lats
from source.search_adaptations import execute_game24_actionly
from source.search_environments import BranchWorld,run_isolated_game24_candidate


class SearchOrder(unittest.TestCase):
    def test_task_pool_bounds_requests_without_parallel_candidates(self):
        lock=threading.Lock();active=peak=0
        class Transport:
            def generate(inner,prompt,cap):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                time.sleep(.01)
                with lock:active-=1
                return dict(raw_output='action\n')
        with tempfile.TemporaryDirectory() as temp,ThreadPoolExecutor(max_workers=4) as pool:
            models=[]
            for task in range(8):
                folder=Path(temp)/str(task);folder.mkdir()
                models.append(Model(None,Transport(),SearchConfig(),.1,folder))
            futures=[pool.submit(model.gpt,'candidate',n=10,stop='\n') for model in models]
            self.assertEqual([future.result() for future in futures],[['action']*10]*8)
            self.assertEqual(peak,4)

    def test_real_search_finishes_generation_then_executes_each_candidate(self):
        trace=[]
        class Transport:
            def generate(inner,prompt,cap):
                trace.append(inner.role)
                if inner.role!='candidate':raise AssertionError('terminal children need no scoring')
                return dict(raw_output='answer: (1 + 2 + 3) * 4 = 24\n')
        real=run_isolated_game24_candidate
        def execute(*args):
            trace.append('env')
            return real(*args)
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            folder=Path(temp)
            world=BranchWorld('game24',0,{'puzzles':['1 2 3 4']},None,None,folder)
            config=SearchConfig(candidates=1,iterations=1,round_actions=12)
            model=Model(None,Transport(),config,.1,folder)
            with patch('source.search_adaptations.run_isolated_game24_candidate',execute):
                result=execute_game24_actionly(world,model,config,load_lats())
            world.close()
        self.assertEqual(trace,['candidate','candidate','env'])
        self.assertTrue(result['success'])

    def test_candidates_execute_sequentially_and_keep_duplicate_policy(self):
        trace=[]
        def submit(ordinal,action):
            trace.append((ordinal,action));future=Future();future.set_result(action);return future
        outputs=[(0,'x'),(1,'x'),(2,'y')]
        rows=execute_candidates(outputs,str,submit)
        self.assertEqual(trace,outputs)
        self.assertEqual([r[2] for r in rows],['x','x','y'])
        with self.assertRaisesRegex(ValueError,'generation must finish'):
            execute_candidates(iter(outputs),str,submit)

    def test_reflection_stays_multiline_and_evaluation_returns_one_score(self):
        records=[]
        class Transport:
            def generate(inner,prompt,cap):
                records.append((inner.role,cap))
                return dict(raw_output='reason\nnext plan' if inner.role=='reflection' else '7\n')
        with tempfile.TemporaryDirectory() as temp:
            model=Model(None,Transport(),SearchConfig(),.1,Path(temp))
            self.assertEqual(model.for_role('reflection')('reflect'),['reason\nnext plan'])
            self.assertEqual(model.for_role('evaluation')('score'),['7\n'])
        self.assertEqual(records,[('reflection',256),('evaluation',100)])

    def test_cli_accepts_http_and_rejects_ipc(self):
        args=['--dataset','webshop','--model-path','/models/m','--website','http://site',
              '--run-id','probe','--endpoints','http://resident:41581']
        _,req=resolve(args)
        self.assertEqual(req['request_limit_per_endpoint'],40)
        with self.assertRaises(SystemExit):resolve(args[:-1]+['unix:///tmp/old.sock'])


if __name__=='__main__':unittest.main()
