import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from source.candidate_pipeline import execute_candidates
from source.config import SearchConfig
from source.game24_env import Game24Environment
from source.lats_webshop import WebShopWorkers, execute_lats, load_lats, parse_webshop_action
from source.search_environments import BranchWorld, run_isolated_game24_candidate
from source.search_adaptations import execute_game24_actionly
from source.resident_transport import ResidentTransport


class PipelineContracts(unittest.TestCase):
    def test_stream_executes_before_next_sample_and_commits_in_sample_order(self):
        started=threading.Event()
        overlap=threading.Barrier(2)
        def work(action):
            started.set()
            overlap.wait(timeout=3)
            return action
        def samples():
            yield 1,'second'
            self.assertTrue(started.wait(3))
            yield 0,'first'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=execute_candidates(samples(),str,lambda ordinal,action:pool.submit(work,action))
        self.assertEqual(results,[(0,'first','first'),(1,'second','second')])

    def test_errors_are_not_converted_into_receipts(self):
        def broken(action):
            raise RuntimeError('original environment failure')
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaisesRegex(RuntimeError,'original environment failure'):
                execute_candidates(iter([(0,'x')]),str,lambda ordinal,action:pool.submit(broken,action))

    def test_parse_failure_drains_already_submitted_environments(self):
        started=threading.Event();release=threading.Event();finished=threading.Event()
        def work(action):
            started.set()
            if not release.wait(3):raise TimeoutError('test release missing')
            finished.set()
        def samples():
            yield 0,'valid'
            self.assertTrue(started.wait(3))
            release.set()
            yield 1,''
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaises(ValueError):
                execute_candidates(samples(),parse_webshop_action,
                                   lambda ordinal,action:pool.submit(work,action))
            self.assertTrue(finished.is_set())

    def test_game24_workers_restore_private_snapshots_and_reuse_environment(self):
        parent=Game24Environment('1 2 3 4')
        checkpoint=parent.checkpoint()
        barrier=threading.Barrier(2)
        def work(action):
            barrier.wait(timeout=3)
            return run_isolated_game24_candidate(parent.puzzle,copy.deepcopy(checkpoint),action)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(work,['1 + 2 = 3','1 + 3 = 4']))
        self.assertEqual(results[0][0]['observation'],'numbers updated; numbers left: 3 4 3')
        self.assertEqual(results[1][0]['observation'],'numbers updated; numbers left: 2 4 4')
        self.assertEqual(parent.checkpoint(),checkpoint)
        with ThreadPoolExecutor(max_workers=1) as pool, \
             patch('source.game24_env.Game24Environment',wraps=Game24Environment) as constructor:
            for action in ['1 + 2 = 3','1 + 3 = 4']:
                pool.submit(run_isolated_game24_candidate,parent.puzzle,checkpoint,action).result()
            constructor.assert_called_once()

    def test_real_game24_search_streams_before_all_candidates_are_generated(self):
        started=threading.Event()
        real=run_isolated_game24_candidate
        def worker(*args):
            started.set()
            return real(*args)
        class Model:
            temperature=.1
            def gpt_iter(inner,prompt,n,stop):
                self.assertEqual(n,2)
                self.assertIsNone(stop)
                yield 1,'answer: (1 + 2 + 3) * 4 = 24'
                self.assertTrue(started.wait(3))
                yield 0,'answer: (1 * 2 * 3) * 4 = 24'
            def gpt(inner,*args,**kwargs):
                raise AssertionError('terminal candidates need no value or reflection model call')
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            directory=Path(temp)
            world=BranchWorld('game24',0,{'puzzles':['1 2 3 4']},None,None,directory)
            checkpoint=world.env.checkpoint()
            with patch('source.search_adaptations.run_isolated_game24_candidate',side_effect=worker):
                result=execute_game24_actionly(world,Model(),SearchConfig(candidates=1),load_lats())
            self.assertTrue(result['success'])
            self.assertEqual(world.env.checkpoint(),checkpoint)
            tree=json.loads((directory/'search_tree.json').read_text())
            self.assertEqual(tree[1]['state']['action'],'answer: (1 * 2 * 3) * 4 = 24')

    def test_game24_invalid_expansion_is_scored_and_best_branch_continues(self):
        scores_started = threading.Barrier(2)
        class Model:
            temperature=.1
            def gpt_iter(inner,prompt,n,stop):
                self.assertIsNone(stop)
                if 'invalid proposal B' not in prompt:
                    yield 0,'invalid proposal A'
                    yield 1,'invalid proposal B'
                else:
                    self.assertIn('Act 1> invalid proposal B',prompt)
                    self.assertIn('invalid arithmetic action',prompt)
                    yield 0,'answer: (1 + 2 + 3) * 4 = 24'
            def gpt(inner,prompt,**kwargs):
                scores_started.wait(timeout=3)
                return ['9' if 'invalid proposal B' in prompt else '1']
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            directory=Path(temp)
            world=BranchWorld('game24',0,{'puzzles':['1 2 3 4']},None,None,directory)
            result=execute_game24_actionly(world,Model(),
                SearchConfig(candidates=1,iterations=1,round_actions=2),load_lats())
            self.assertTrue(result['success'])
            self.assertEqual(world.cache[('invalid proposal A',)]['reward'],0)
            self.assertEqual(world.cache[('invalid proposal B',)]['reward'],0)

    def test_webshop_real_native_http_environments_are_private_and_reused(self):
        barrier=threading.Barrier(2)
        sessions=[]
        class Http:
            def __init__(self):
                self.urls=[];self.closed=False;sessions.append(self)
            def get(self,url,timeout):
                self.urls.append(url)
                if '/search_results/' in url:
                    barrier.wait(timeout=3)
                    html='<div>Header</div><div>Search</div><a class="product-link">PRODUCT</a>'
                else:
                    html='<div>Same task 8</div>'
                return Mock(text=html,status_code=200)
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as temp,patch('requests.Session',Http):
            workers=WebShopWorkers(load_lats()[0],'http://site',Path(temp),logging,8)
            with ThreadPoolExecutor(max_workers=2) as pool:
                root=pool.submit(workers.execute,{},'fixed_8','reset').result()
                original=copy.deepcopy(root['after'])
                a=pool.submit(workers.execute,original,'fixed_8','search[alpha]')
                b=pool.submit(workers.execute,original,'fixed_8','search[beta]')
                left,right=a.result(),b.result()
                self.assertNotEqual(left['server_session'],right['server_session'])
                self.assertTrue(left['server_session'].endswith('_fixed_8'))
                self.assertTrue(right['server_session'].endswith('_fixed_8'))
                self.assertEqual(left['after']['fixed_8']['query_string'],'alpha')
                self.assertEqual(right['after']['fixed_8']['query_string'],'beta')
                self.assertEqual(original,root['after'])
                self.assertEqual(original['fixed_8']['page_type'],'init')
                self.assertEqual(len(sessions),2)
                for http in sessions:
                    remote_ids=[urlsplit(url).path.split('/')[2] for url in http.urls if '/search_results/' in url]
                    self.assertEqual(len(set(remote_ids)),1)
            workers.close()
            self.assertTrue(all(http.closed for http in sessions))

    def test_webshop_native_invalid_action_keeps_error_evidence(self):
        with tempfile.TemporaryDirectory() as temp,patch('requests.Session') as http:
            http.return_value.get.return_value=Mock(text='<div>Task</div>',status_code=200)
            workers=WebShopWorkers(load_lats()[0],'http://site',Path(temp),logging,8)
            try:
                root=workers.execute({},'fixed_8','reset')
                invalid=workers.execute(root['after'],'fixed_8','click[Buy Now]')
                self.assertEqual(invalid['receipt'],('Invalid action!',-1,False))
                self.assertIn('AssertionError',(Path(temp)/'events.jsonl').read_text())
            finally:
                workers.close()
        with self.assertRaises(ValueError):parse_webshop_action('')
        with self.assertRaises(ValueError):parse_webshop_action('search[x]\nclick[y]')

    def test_webshop_rejects_worker_goal_mismatch_and_propagates_http_errors(self):
        from requests import HTTPError
        with tempfile.TemporaryDirectory() as temp,patch('requests.Session') as http:
            http.return_value.get.return_value=Mock(text='<div>Different goal</div>',status_code=200)
            workers=WebShopWorkers(load_lats()[0],'http://site',Path(temp),logging,8)
            workers.root_observation='Original goal'
            snapshot={'fixed_8':{'session':'fixed_8','page_type':'init'}}
            try:
                with self.assertRaisesRegex(ValueError,'changed task observation'):
                    workers.execute(snapshot,'fixed_8','search[x]')
                http.return_value.get.return_value.raise_for_status.side_effect=HTTPError('503 original')
                with self.assertRaisesRegex(HTTPError,'503 original'):
                    workers.execute({},'fixed_8','reset')
            finally:
                workers.close()

    def test_webshop_search_preserves_duplicate_order_with_pipeline(self):
        started=threading.Event()
        class Env:
            def __init__(self):self.sessions={}
            def clone_state(self):return copy.deepcopy(self.sessions)
            def step(self,session,action):
                self.sessions[session]={'action':action}
                if action=='reset':return 'Task',0,False
                started.set()
                return 'won',1,True
        class Model:
            temperature=.1
            tokenizer=None
            def gpt_iter(inner,prompt,n,stop):
                self.assertEqual(n,4)
                self.assertTrue(prompt.endswith('\nAction: '))
                yield 3,'search[B]'
                self.assertTrue(started.wait(3))
                yield 0,'search[A]'
                yield 2,'search[A]'
                yield 1,'search[B]'
            def gpt(self,*args,**kwargs):return ['10']
        author=load_lats()
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()), \
             patch.object(author[0],'webshopEnv',Env):
            result=execute_lats(8,author,'http://site',Model(),Path(temp),SearchConfig(candidates=2))
            tree=json.loads((Path(temp)/'search_tree.json').read_text())
            self.assertEqual(tree[0]['children'],[3,4])
            self.assertEqual([row['state']['action'] for row in tree[1:]],
                             ['search[A]','search[B]','search[A]','search[B]'])
            self.assertTrue(result['success'])

    def test_concurrent_queue_seconds_are_not_subtracted_from_wall_time(self):
        transport=ResidentTransport.__new__(ResidentTransport)
        transport.records=[{'response':dict(queue_time_seconds=8,model_active_step_seconds=2,
                                          input_tokens=3,output_tokens=4)} for _ in range(3)]
        costs=transport.costs(10)
        self.assertEqual(costs['model_queue_seconds'],24)
        self.assertEqual(costs['elapsed_seconds'],10)
        self.assertIsNone(costs['queue_excluded_elapsed_seconds'])
        transport.records[0]['response']['queue_time_seconds']=-1
        with self.assertRaisesRegex(ValueError,'invalid request duration'):transport.costs(10)


if __name__=='__main__':unittest.main()
