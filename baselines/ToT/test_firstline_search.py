from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from unittest.mock import patch
from source.config import SearchConfig, resolve
from source.model import Model
from source.search import load_native, solve
from source.environments import BranchWorld, InteractiveTask


class SearchOrder(unittest.TestCase):
    def test_failed_bfs_never_restarts(self):
        config = SearchConfig('sample',2,2,1,12,100,1,10)
        failed = dict(outputs=['failed'], candidate_rewards=[0], first_ranked_reward=0,
                      first_ranked_success=False, search={'steps':[{}]})
        with tempfile.TemporaryDirectory() as temp, patch('source.search.solve_round', return_value=failed) as once:
            result = solve(0,None,None,None,config,Path(temp),.1,True)
            once.assert_called_once()
            self.assertEqual(result['rounds_executed'],1)
            self.assertFalse(result['first_ranked_success'])

    def test_cli_and_serialized_config_reject_two_attempts(self):
        args=['--dataset','webshop','--model-path','/models/m','--website','http://site',
              '--run-id','probe','--endpoints','http://resident:41581']
        _,req=resolve(args)
        self.assertEqual(req['experiment']['search']['iterations'],1)
        with self.assertRaises(ValueError):
            SearchConfig('sample',2,2,1,12,100,2,10)
        with self.assertRaises(SystemExit):
            resolve(args+['--iterations','2'])

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
                models.append(Model(None,Transport(),SearchConfig('sample',2,2,1,1,100,1,10),.1,folder))
            futures=[pool.submit(model.gpt,'candidate',n=10,stop='\n') for model in models]
            self.assertEqual([future.result() for future in futures],[['action']*10]*8)
            self.assertEqual(peak,4)

    def test_complete_layer_precedes_environment_and_sequential_values(self):
        trace=[]
        class Env:
            def reset(self):
                self.actions=[]
                return dict(observation='root',reward=0,terminal=False)
            def snapshot(self):return list(self.actions)
            def restore(self,state):self.actions=list(state)
            def step(self,action):
                trace.append('env');self.actions.append(action)
                return dict(observation=action,reward=0,terminal=False)
        class Transport:
            count=0
            def generate(inner,prompt,cap):
                trace.append(inner.role)
                if inner.role=='evaluation':return dict(raw_output='likely')
                inner.count+=1
                return dict(raw_output='move'+str(inner.count)+'\n')
        bfs,_=load_native()
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            folder=Path(temp);world=BranchWorld(Env(),'webshop','examples',2,folder)
            config=SearchConfig('sample',2,2,1,2,100,1,2)
            model=Model(None,Transport(),config,.1,folder)
            result=solve(0,InteractiveTask(world),bfs,model,config,folder,.1,True)
        self.assertEqual(trace,['candidate']*2+['env']*2+['evaluation']*2+
                               ['candidate']*4+['env']*4+['evaluation']*4)
        self.assertEqual(len(result['search']['steps']),2)

    def test_candidate_and_evaluator_formats_are_separate(self):
        class Transport:
            def generate(inner,prompt,cap):
                return dict(raw_output='action\n' if inner.role=='candidate' else 'likely\n')
        with tempfile.TemporaryDirectory() as temp:
            config=SearchConfig('sample',2,2,1,1,100,1,2)
            model=Model(None,Transport(),config,.1,Path(temp))
            self.assertEqual(model.gpt('candidate',n=2,stop='\n'),['action','action'])
            self.assertEqual(model.for_role('evaluation')('score'),['likely\n'])
            self.assertEqual(model.role,'candidate')

    def test_cli_accepts_http_and_rejects_ipc(self):
        args=['--dataset','webshop','--model-path','/models/m','--website','http://site',
              '--run-id','probe','--endpoints','http://resident:41581']
        _,req=resolve(args)
        self.assertEqual(req['request_limit_per_endpoint'],40)
        with redirect_stdout(io.StringIO()),self.assertRaises(SystemExit):
            resolve(args[:-1]+['unix:///tmp/old.sock'])


if __name__=='__main__':unittest.main()
