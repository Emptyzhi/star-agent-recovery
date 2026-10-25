from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stdout
import copy
import io
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from source.config import ROOT, SearchConfig
from source.environments import ALFWorld, create_task, load_inputs
from source.isolated_environments import (IsolatedBranchWorld, environment_executor,
                                          process_environment_job)
from source.model import Model
from source.search import load_native, solve
from source.resident_transport import ResidentTransport

_PROCESS_BARRIER = None

class ScriptedSubmission:
    def submit_gpt(self, prompt, **kwargs):
        futures=[]
        for index,text in enumerate(self.gpt(prompt,**kwargs)):
            future=Future();future.set_result((index,text));futures.append(future)
        return futures

class FakeAlfWorld:
    """Spawn-picklable fixture: no TextWorld or benchmark files are required."""
    def __init__(self, gamefile, config, directory):
        if multiprocessing.current_process().name == 'MainProcess':
            raise AssertionError('ALFWorld constructed in parent')
        self.gamefile, self.directory = gamefile, directory
    def reset(self):
        self.actions=[]
        return dict(observation='root '+self.gamefile, reward=0, terminal=False)
    def step(self, action):
        if _PROCESS_BARRIER is not None and action in ('b','c'):
            _PROCESS_BARRIER.wait(timeout=10)
        time.sleep(.05)
        self.actions.append(action)
        return dict(observation=','.join(self.actions), reward=int(action=='win'), terminal=action=='win')
    def close(self):pass


def initialize_fake_alfworld(barrier=None):
    global _PROCESS_BARRIER
    _PROCESS_BARRIER = barrier
    import source.isolated_environments as module
    module.ALFWorld=FakeAlfWorld


class PipelineTests(unittest.TestCase):
    def test_candidate_job_directories_are_reserved_before_worker_execution(self):
        directories=[]
        class Executor:
            def submit(self, operation, directory, *args):
                directory=Path(directory)
                self_outer.assertTrue(directory.is_dir())
                directories.append(directory)
                future=Future()
                future.set_result(dict(receipt={'observation':'root','reward':0,'terminal':False},
                                       snapshot=None,replay_steps=0))
                return future
        self_outer=self
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            world=IsolatedBranchWorld(dict(dataset='webshop',directory=str(directory)),
                                      '',15,directory,Executor())
            try:
                for _ in range(100):
                    world._submit_job((),None,{},None).result()
                world.restart_search()
                self.assertEqual(len(directories),102)
                self.assertEqual(len(set(directories)),102)
                self.assertTrue(all(path.parent==directory/'candidate_envs' for path in directories))
            finally:
                world.close()

    def test_raw_text_and_concurrent_model_calls_for_all_models(self):
        for name in ('Qwen','LLaMA','Mistral'):
            barrier=threading.Barrier(3)
            lock=threading.Lock()
            prompts=[]
            class Transport:
                def generate(self,prompt,cap):
                    with lock:
                        prompts.append(prompt)
                    barrier.wait(timeout=5)
                    return {'raw_output':'act\nobservation'}
            with tempfile.TemporaryDirectory() as temp:
                tokenizer=Mock()
                tokenizer.apply_chat_template.side_effect=AssertionError(name+' template called')
                model=Model(tokenizer,Transport(),SearchConfig('sample',3,1,1,2,77,1),.1,Path(temp))
                try:
                    self.assertEqual(model.gpt('RAW '+name,n=3,stop='\n'),['act']*3)
                    self.assertEqual(prompts,['RAW '+name]*3)
                    tokenizer.apply_chat_template.assert_not_called()
                finally:model.close()

    def test_game24_pipeline_starts_environment_before_second_sample(self):
        started=threading.Event()
        from source.isolated_environments import ThreadEnvironments
        original=ThreadEnvironments.execute
        def execute(worker,directory,actions,*args):
            if actions:started.set()
            return original(worker,directory,actions,*args)
        class ScriptedModel(ScriptedSubmission):
            role='proposal'
            def gpt_iter(inner,prompt,n,**kwargs):
                self.assertEqual(n,2)
                yield 1,'Act 1> 1 + 3 = 4'
                self.assertTrue(started.wait(5))
                yield 0,'Act 1> 1 + 2 = 3'
            def gpt(inner,prompt,**kwargs):return ['likely']
        with tempfile.TemporaryDirectory() as temp,ThreadPoolExecutor(2) as pool, \
             patch.object(ThreadEnvironments,'execute',execute),redirect_stdout(io.StringIO()):
            directory=Path(temp)
            csv=directory/'p.csv';csv.write_text('Puzzles\n1 2 3 4\n')
            req={'dataset':'game24','experiment':{'search':{'max_depth':1}}}
            task,world=create_task(req,0,{'csv':str(csv),'examples':'native examples'},directory,pool)
            try:
                result=solve(0,task,load_native()[0],ScriptedModel(),
                             SearchConfig('sample',2,2,1,1,100,1),directory,.1,True)
                self.assertEqual(result['outputs'],['1 + 2 = 3\n','1 + 3 = 4\n'])
                self.assertEqual(world.cache[('1 + 2 = 3',)]['observation'],
                                 'numbers updated; numbers left: 3 4 3')
                self.assertEqual(world.cache[('1 + 3 = 4',)]['observation'],
                                 'numbers updated; numbers left: 2 4 4')
                self.assertEqual(world.snapshots[()][0],('1','2','3','4'))
                self.assertEqual(world.transition_calls,2)
            finally:world.close()

    def test_proposal_mode_executes_candidates_concurrently_without_extra_generation(self):
        from source.isolated_environments import ThreadEnvironments
        original=ThreadEnvironments.execute
        barrier=threading.Barrier(2)
        def execute(worker,directory,actions,*args):
            if actions:barrier.wait(timeout=5)
            return original(worker,directory,actions,*args)
        class Model(ScriptedSubmission):
            role='proposal'
            def __init__(self):self.proposals=0
            def gpt(self,prompt,**kwargs):
                if 'Propose alternative' in prompt:
                    self.proposals+=1
                    return ['1 + 2 = 3\n1 + 3 = 4']
                return ['likely']
        with tempfile.TemporaryDirectory() as temp,ThreadPoolExecutor(2) as pool, \
             patch.object(ThreadEnvironments,'execute',execute),redirect_stdout(io.StringIO()):
            directory=Path(temp)
            csv=directory/'p.csv';csv.write_text('Puzzles\n1 2 3 4\n')
            req={'dataset':'game24','experiment':{'search':{'max_depth':1}}}
            task,world=create_task(req,0,{'csv':str(csv),'examples':''},directory,pool)
            model=Model()
            try:
                result=solve(0,task,load_native()[0],model,
                    SearchConfig('propose',None,2,1,1,100,1),directory,.1,True)
                self.assertEqual(model.proposals,1)
                self.assertEqual(len(result['outputs']),2)
            finally:world.close()

    def test_real_spawn_alfworld_roots_replay_and_independent_candidates(self):
        context=multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as temp,ProcessPoolExecutor(
                max_workers=2,mp_context=context,
                initializer=initialize_fake_alfworld,initargs=(context.Barrier(2),)) as pool:
            directory=Path(temp)
            spec=dict(dataset='alfworld',task_id=0,directory=str(directory),gamefile='game0',config={})
            world=IsolatedBranchWorld(spec,'examples',3,directory,pool)
            try:
                world.at(('a',))
                futures=[(actions,world.submit(actions)) for actions in [('a','b'),('a','c')]]
                for actions,future in futures:world.commit(actions,future.result())
                self.assertEqual(world.cache[('a','b')]['observation'],'a,b')
                self.assertEqual(world.cache[('a','c')]['observation'],'a,c')
                self.assertEqual(world.replay_calls,2)
                receipts=[json.loads(path.read_text()) for path in directory.glob('candidate_envs/*/receipt.jsonl')]
                self.assertTrue(all(row['pid']!=os.getpid() for row in receipts))
                self.assertEqual(len({row['pid'] for row in receipts}),2)
                world.cache[('a',)]['observation']='corrupt'
                with self.assertRaisesRegex(ValueError,'replay differs'):
                    world.at(('a','d'))
            finally:world.close()
        with self.assertRaisesRegex(RuntimeError,'isolated environment process'):
            ALFWorld('game',{},Path('.'))

    def test_alfworld_full_search_uses_process_pipeline(self):
        class ScriptedModel(ScriptedSubmission):
            role='proposal'
            def gpt_iter(self,prompt,**kwargs):
                yield 1,'go'
                yield 0,'win'
            def gpt(self,*args,**kwargs):return ['likely']
        with tempfile.TemporaryDirectory() as temp,ProcessPoolExecutor(
                max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_fake_alfworld) as pool,redirect_stdout(io.StringIO()):
            directory=Path(temp)
            req={'dataset':'alfworld','experiment':{'search':{'max_depth':1}}}
            template=dict(games=['/data/pick_and_place-test/trial/game'],config={},
                          demos={'act_put_1':'example','act_put_0':'example'})
            task,world=create_task(req,0,template,directory,pool)
            try:
                result=solve(0,task,load_native()[0],ScriptedModel(),
                             SearchConfig('sample',2,1,1,1,100,1),directory,.1,True)
                self.assertTrue(result['first_ranked_success'])
                self.assertEqual(result['outputs'],['win\n'])
            finally:world.close()

    def test_webshop_pipeline_isolates_server_sessions_http_and_parent_options(self):
        from source.isolated_environments import ThreadEnvironments
        original=ThreadEnvironments.execute
        barrier=threading.Barrier(2)
        started=threading.Event()
        def execute(worker,directory,actions,*args):
            if actions:
                started.set();barrier.wait(timeout=5)
            return original(worker,directory,actions,*args)
        sessions=[]
        class Http:
            def __init__(self):self.urls=[];self.closed=False;sessions.append(self)
            def get(self,url,timeout):
                self.urls.append(url)
                html=('<div>Header</div><div>Search</div><a class="product-link">ASIN</a>'
                      if '/search_results/' in url else '<div>Same goal 8</div>')
                return Mock(text=html,status_code=200)
            def close(self):self.closed=True
        class ScriptedModel(ScriptedSubmission):
            role='proposal'
            def gpt_iter(inner,*args,**kwargs):
                yield 1,' Action: search[beta] '
                self.assertTrue(started.wait(5))
                yield 0,' search[alpha] '
            def gpt(self,*args,**kwargs):return ['likely']
        template=load_inputs({'dataset':'webshop','website':'http://site','task_ids':[8]},None)
        with tempfile.TemporaryDirectory() as temp,ThreadPoolExecutor(2) as pool, \
             patch('requests.Session',Http),patch.object(ThreadEnvironments,'execute',execute), \
             redirect_stdout(io.StringIO()):
            directory=Path(temp)
            task,world=create_task({'dataset':'webshop','experiment':{'search':{'max_depth':1}}},
                                   8,template,directory,pool)
            try:
                root=copy.deepcopy(world.snapshots[()])
                result=solve(8,task,load_native()[0],ScriptedModel(),
                    SearchConfig('sample',2,2,1,1,100,1),directory,.1,True)
                self.assertEqual(result['outputs'],['search[alpha]\n','search[beta]\n'])
                self.assertEqual(world.snapshots[()],root)
                self.assertEqual(world.snapshots[('search[alpha]',)]['fixed_8']['query_string'],'alpha')
                self.assertEqual(world.snapshots[('search[beta]',)]['fixed_8']['query_string'],'beta')
                self.assertEqual(len(sessions),2)
                remote=[url.split('/search_results/')[1].split('/')[0]
                        for http in sessions for url in http.urls if '/search_results/' in url]
                self.assertEqual(len(set(remote)),2)
                self.assertTrue(all(sid.endswith('_fixed_8') for sid in remote))
            finally:world.close()
            self.assertTrue(all(http.closed for http in sessions))

    def test_duplicate_cache_restart_and_worker_reuse(self):
        with tempfile.TemporaryDirectory() as temp,ThreadPoolExecutor(1) as pool:
            directory=Path(temp)
            csv=directory/'p.csv';csv.write_text('Puzzles\n1 2 3 4\n')
            spec=dict(dataset='game24',task_id=0,directory=str(directory),csv=str(csv))
            world=IsolatedBranchWorld(spec,'',2,directory,pool)
            try:
                actions=('1 + 2 = 3',)
                a,b=world.submit(actions),world.submit(actions)
                self.assertIs(a,b)
                world.commit(actions,a.result());world.commit(actions,b.result())
                self.assertEqual(world.transition_calls,1)
                world.restart_search()
                self.assertNotIn(actions,world.cache)
                world.at(actions)
                self.assertEqual(world.transition_calls,2)
                self.assertEqual(len(world.worker.resources),1)
                with self.assertRaisesRegex(ValueError,'completed branch'):
                    world.at(actions+('3 + 3 = 6','6 * 4 = 24'))
            finally:world.close()

    def test_environment_cap_and_process_selection(self):
        with patch('source.isolated_environments.ProcessPoolExecutor') as processes, \
             patch('source.isolated_environments.ThreadPoolExecutor') as threads:
            environment_executor('alfworld',400)
            self.assertEqual(processes.call_args.kwargs['max_workers'],400)
            self.assertEqual(processes.call_args.kwargs['mp_context'].get_start_method(),'spawn')
            environment_executor('webshop',400);environment_executor('game24',400)
            self.assertEqual(threads.call_count,2)
            self.assertTrue(all(c.kwargs['max_workers']==400 for c in threads.call_args_list))

    def test_overlapping_request_queue_durations_are_not_wall_time(self):
        transport=ResidentTransport.__new__(ResidentTransport)
        transport.records=[dict(response=dict(queue_time_seconds=9,model_active_step_seconds=1,
                          input_tokens=3,output_tokens=2)) for _ in range(3)]
        cost=transport.costs(10)
        self.assertEqual(cost['model_queue_seconds'],27)
        self.assertIsNone(cost['queue_excluded_elapsed_seconds'])
        transport.records[0]['response']['queue_time_seconds']=-1
        with self.assertRaises(ValueError):transport.costs(10)

    def test_concurrent_transport_ordinals_sessions_and_evidence(self):
        identity=dict(parent_job_id='1',hostname='gpu',gpu_uuid='uuid',service_pid=2,
                      service_instance_id='instance',slurm_step_id='3')
        sampling={'temperature':.1}
        barrier=threading.Barrier(3)
        sessions=[]
        lock=threading.Lock()
        class Http:
            def __init__(self):
                self.closed=False
                with lock:sessions.append(self)
            def post(self,url,json,timeout):
                barrier.wait(timeout=5)
                self.payload=json
                raw=dict(gpu_identity=identity,runtime=identity,effective_seed=None,task_seed=None,
                         text='candidate',raw_output='candidate',output_token_limit=77,
                         output_tokens=1,output_token_ids=[7],input_tokens=5,decoding=sampling,
                         queue_time_seconds=10,model_active_step_seconds=1)
                response=Mock(ok=True);response.json.return_value=raw
                return response
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as temp,patch('requests.Session',Http):
            directory=Path(temp)
            transport=ResidentTransport('http://resident','lease',sampling,identity,directory)
            model=Model(Mock(),transport,SearchConfig('sample',3,1,1,1,77,1),.1,directory)
            try:
                for _ in range(2):
                    self.assertEqual(model.gpt('unchanged prompt',n=3),['candidate']*3)
                self.assertEqual(len(sessions),3)
                requests=[json.loads(line) for line in (directory/'requests.jsonl').read_text().splitlines()]
                responses=[json.loads(line) for line in (directory/'responses.jsonl').read_text().splitlines()]
                self.assertEqual(sorted(row['ordinal'] for row in requests),list(range(6)))
                self.assertEqual(sorted(row['ordinal'] for row in responses),list(range(6)))
                self.assertTrue(all(row['payload']['prompt']=='unchanged prompt' for row in requests))
                self.assertEqual(transport.costs(1)['model_queue_seconds'],60)
            finally:
                model.close();transport.close()
            self.assertTrue(all(http.closed for http in sessions))

    def test_pipeline_environment_failure_is_not_converted_to_a_score(self):
        from source.isolated_environments import ThreadEnvironments
        original=ThreadEnvironments.execute
        def execute(worker,directory,actions,*args):
            if actions:raise RuntimeError('original candidate failure')
            return original(worker,directory,actions,*args)
        class ScriptedModel(ScriptedSubmission):
            role='proposal'
            def gpt_iter(self,*args,**kwargs):yield 0,'1 + 2 = 3'
            def gpt(self,*args,**kwargs):raise AssertionError('must not evaluate a failed environment')
        with tempfile.TemporaryDirectory() as temp,ThreadPoolExecutor(1) as pool, \
             patch.object(ThreadEnvironments,'execute',execute),redirect_stdout(io.StringIO()):
            directory=Path(temp);csv=directory/'p.csv';csv.write_text('Puzzles\n1 2 3 4\n')
            task,world=create_task({'dataset':'game24','experiment':{'search':{'max_depth':1}}},
                                  0,{'csv':str(csv),'examples':''},directory,pool)
            try:
                with self.assertRaisesRegex(RuntimeError,'original candidate failure'):
                    solve(0,task,load_native()[0],ScriptedModel(),
                          SearchConfig('sample',1,1,1,1,100,1),directory,.1,True)
                self.assertEqual(world.transition_calls,0)
                self.assertEqual(list(world.cache),[()])
            finally:world.close()


if __name__=='__main__':unittest.main()
