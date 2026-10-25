import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit

from source.config import ROOT, SearchConfig, resolve, validate_request
from source.datasets import create_world, load_inputs, native_url
from source.lats_webshop import load_lats, execute_lats as native_search
from source.model import Model, validate_health
from source.search_adaptations import (
    DatasetLatsTask,
    Game24ActionOnlyTask,
    AlfWorldActionOnlyTask,
    execute_lats as adapted_search,
)
from source.author_binding import bind_module
from source.search_environments import BranchWorld, LatsEnvironment
from source import runner
from source.resident_transport import ResidentTransport
from source.round_budget import ROUND_ACTIONS, install_round_search


def cli(dataset='game24', extra=()):
    flags = {'game24': ['--csv', 'puzzles.csv'], 'alfworld': ['--manifest', 'tasks.json'],
             'webshop': ['--website', 'http://environment:3001']}[dataset]
    return resolve(['--dataset', dataset, '--model-path', '/models/selected',
                    '--endpoints', 'http://resident:9000', 'http://resident2:9000', '--run-id', 'test', *flags, *extra])


class Tokenizer:
    def __init__(self, name='selected'):
        self.name = name
        self.chat_template_calls = []
    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append((messages, kwargs))
        return self.name+'\n'+messages[0]['content']


class Transport:
    def __init__(self, outputs): self.outputs, self.calls = iter(outputs), []
    def generate(self, prompt, cap):
        self.calls.append((prompt, cap))
        return dict(raw_output=next(self.outputs))


class World:
    dataset = 'webshop'
    horizon = 3
    task_id = 0
    examples = 'Task examples.'
    def __init__(self, directory):
        self.directory = directory
        self.cache = {(): dict(observation='Solve the task.', reward=0, terminal=False)}
    def at(self, actions):
        actions = tuple(actions)
        if actions not in self.cache:
            if self.at(actions[:-1])['terminal']: raise ValueError('action after terminal')
            self.cache[actions] = dict(observation='Receipt '+actions[-1], reward=int(actions[-1]=='finish'),
                terminal=actions[-1] in ('bad', 'finish') or len(actions)==self.horizon)
        return self.cache[actions]
    def trajectory(self, actions):
        return 'Task\n'+'\n'.join('Action: '+a+'\nObservation: '+self.at(actions[:i+1])['observation']
                                  for i, a in enumerate(actions))


class CliContracts(unittest.TestCase):
    def test_endpoint_limit_400_is_preserved_and_validated(self):
        _, req = cli('game24', ['--request-limit-per-endpoint','400',
            '--workers-per-gpu','400','--max-num-seqs','400'])
        validate_request(req)
        self.assertEqual(req['request_limit_per_endpoint'],400)
        req['request_limit_per_endpoint'] = 0
        with self.assertRaisesRegex(ValueError,'endpoint request limit'):
            validate_request(req)

    def test_max_depth_alias_resolves_same_round_budget(self):
        for option in ('--max-depth', '--max_depth', '--round-actions'):
            _, req = cli('webshop', [option, '7', '--iterations', '2'])
            config = validate_request(json.loads(json.dumps(req)))
            self.assertEqual(config.round_actions, 7)
            self.assertEqual(config.iterations, 2)

    def test_all_datasets_models_and_remote_paths_roundtrip(self):
        for dataset, count in [('game24',500), ('webshop',500), ('alfworld',134)]:
            for model in ('qwen', 'llama', 'mistral', 'arbitrary-model'):
                _, req = cli(dataset, ['--local_llm_path', '/models/'+model, '--package', '/remote/lats'])
                self.assertEqual(len(req['task_ids']), count)
                self.assertEqual(req['model_path'], '/models/'+model)
                self.assertEqual(validate_request(json.loads(json.dumps(req))),
                                 SearchConfig(round_actions=ROUND_ACTIONS[dataset]))
                if dataset == 'alfworld':
                    self.assertEqual(req['alfworld_config'], '/remote/lats/vendor/ga/test_alfworld/base_config.yaml')

    def test_remote_dataset_paths_are_not_rewritten_by_windows_path(self):
        _, req = resolve(['--dataset', 'alfworld', '--model-path', '/models/selected',
                          '--endpoints', 'http://resident:9000', 'http://resident2:9000',
                          '--run-id', 'remote-paths', '--manifest', '/data/tasks.json',
                          '--alfworld-data', '/data/alfworld'])
        self.assertEqual(req['manifest'], '/data/tasks.json')
        self.assertEqual(req['alfworld_data'], '/data/alfworld')

    def test_entry_exports_without_runtime_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'request.json'
            code = "import sys;sys.path.insert(0,sys.argv[1]);import run_lats;run_lats.main(sys.argv[2:]);assert 'source.runner' not in sys.modules;assert 'requests' not in sys.modules;assert 'transformers' not in sys.modules"
            subprocess.run([sys.executable, '-S', '-c', code, str(ROOT), '--dataset', 'webshop',
                '--local_llm_path', '/models/new', '--endpoints', 'http://resident:9000', 'http://resident2:9000', '--website',
                'http://site:3001', '--run-id', 'offline', '--write-request', str(target)], check=True, capture_output=True)
            self.assertEqual(json.loads(target.read_text())['model_path'], '/models/new')

    def test_wrong_dataset_unknown_fields_and_wrong_model_are_rejected(self):
        with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
            cli('webshop', ['--csv', 'wrong.csv'])
        _, req = cli()
        req['experiment']['search']['beam_width'] = 4
        with self.assertRaises(ValueError): validate_request(req)
        _, req = cli()
        health = dict(status='READY', fatal_error=None, engine='vllm', model_path='/models/other',
                      max_num_seqs=40, max_new_tokens=100)
        with self.assertRaisesRegex(ValueError, 'model path'): validate_health(health, req)
        health['model_path'] = req['model_path']; health['max_new_tokens'] = 99
        with self.assertRaisesRegex(ValueError, 'capacity'): validate_health(health, req)

    def test_queue_dispatch_preserves_exact_request_in_children(self):
        _, req = cli('webshop', ['--candidates','2','--eval-samples','2','--iterations','4',
                               '--max-output-tokens','77','--task-ids','1','7',
                               '--endpoints','http://one:9000','http://two:9000'])
        commands = []
        def spawn(command, **kwargs):
            commands.append(command)
            received = json.loads(Path(command[command.index('--request')+1]).read_text())
            self.assertEqual(received, req)
            gpu = int(command[command.index('--gpu-index')+1])
            group = Path(command[command.index('--output-dir')+1]); group.mkdir()
            for task_id in req['task_ids'][gpu::2]:
                task = group/f'task_{task_id:03d}'; task.mkdir()
                runner.save(task/'summary.json', dict(complete=True,identity=dict(task_id=task_id), result=dict(reward=1,success=True)))
            return Mock(wait=lambda:0, poll=lambda:0)
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'verify_allocation'), \
             patch.object(runner,'get_health',side_effect=[{'gpu_uuid':'one'},{'gpu_uuid':'two'}]), \
             patch.object(runner.subprocess,'Popen',side_effect=spawn):
            self.assertEqual(runner.dispatch(req,Path(temp)/'results','parent','compute'),0)
        self.assertEqual(len(commands),2)
        self.assertTrue(all(c[0]==sys.executable for c in commands))

    def test_shard_selects_tokenizer_endpoints_and_worker_count(self):
        load_lats()  # Keep native imports outside the temporary sys.modules fixture.
        _,req=cli('game24',['--workers-per-gpu','3','--task-ids','2','4','6','8',
                           '--endpoints','http://one:9000','http://two:9000'])
        transformer=ModuleType('transformers'); transformer.AutoTokenizer=Mock()
        token=transformer.AutoTokenizer.from_pretrained.return_value
        response=Mock(); response.raise_for_status.return_value=None
        def completed(task_id,*args,**kwargs): return dict(complete=True,identity=dict(task_id=task_id))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules,transformers=transformer), \
             patch.object(runner,'verify_allocation'),patch.object(runner,'get_health',return_value={}) as health, \
             patch.object(runner,'load_inputs'),patch.object(runner,'run_task',side_effect=completed) as task, \
             patch('requests.post',return_value=response) as post, \
             patch.object(runner,'ThreadPoolExecutor',wraps=ThreadPoolExecutor) as pool:
            self.assertEqual(runner.run_shard(req,Path(temp)/'group',1,'parent','compute'),0)
            health.assert_called_once_with('http://two:9000',req,'parent','compute')
            transformer.AutoTokenizer.from_pretrained.assert_called_once_with('/models/selected',trust_remote_code=False,local_files_only=True)
            self.assertEqual(pool.call_count,2)
            self.assertEqual([call.kwargs['max_workers'] for call in pool.call_args_list],[3,3])
            self.assertIsNone(task.call_args_list[0].kwargs['request_pool'])
            self.assertIs(task.call_args_list[0].kwargs['request_pool'],
                          task.call_args_list[1].kwargs['request_pool'])
            environment_pools=[call.args[-1] for call in task.call_args_list]
            self.assertIs(environment_pools[0],environment_pools[1])
            self.assertIsInstance(environment_pools[0],ThreadPoolExecutor)
            self.assertIs(task.call_args_list[0].kwargs['request_limiter'],
                          task.call_args_list[1].kwargs['request_limiter'])
            self.assertEqual(sorted(c.args[0] for c in task.call_args_list),[4,8])
            self.assertTrue(all(c.args[5] is token for c in task.call_args_list))
            self.assertEqual([c.args[0] for c in post.call_args_list],['http://two:9000/begin_run','http://two:9000/end_run'])

    def test_resolved_sampling_and_tokens_reach_real_transport_payload(self):
        _,req=cli('webshop',['--temperature','.23','--top-p','.9','--top-k','7',
                           '--repetition-penalty','1.1','--max-output-tokens','81'])
        health=dict(parent_job_id='42',hostname='compute',gpu_uuid='uuid',slurm_step_id='3',
                    service_pid=100,service_instance_id='instance')
        raw=dict(raw_output='first\n',text='first\n',gpu_identity=health,stop_mode='first_line',
            runtime=health,effective_seed=None,task_seed=None,output_token_limit=81,
            output_tokens=2,output_token_ids=[1,2],decoding=req['sampling'],input_tokens=10,
            queue_time_seconds=28,model_active_step_seconds=2)
        session=Mock();session.post.return_value.ok=True;session.post.return_value.json.return_value=raw
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            tokenizer=Tokenizer('arbitrary')
            transport=ResidentTransport(req['endpoints'][0],'lease',req['sampling'],health,directory,session=session)
            model=Model(tokenizer,transport,validate_request(req),.23,directory)
            self.assertEqual(model.gpt('task',stop='\n'),['first'])
            payload=session.post.call_args.kwargs['json']
            self.assertEqual(payload['sampling'],req['sampling'])
            self.assertEqual(payload['max_new_tokens'],81)
            self.assertEqual(payload['prompt'],'task')
            self.assertEqual(tokenizer.chat_template_calls,[])
            self.assertIsNone(payload['task_seed']); self.assertEqual(payload['stop_mode'],'first_line')
            self.assertEqual(transport.costs(32)['queue_excluded_elapsed_seconds'],4)
            self.assertEqual(transport.costs(32)['model_queue_seconds'],28)
            with self.assertRaises(ValueError):transport.costs(10)


class SearchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.author = load_lats()

    def test_native_child_snapshot_accumulates_game24_actions_and_isolates_siblings(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp); csv=directory/'puzzles.csv'
            csv.write_text('Puzzles\n1 2 3 4\n')
            _,req=cli('game24',['--csv',str(csv),'--task-ids','0'])
            world=create_world(req,0,load_inputs(req),None,directory)
            env=LatsEnvironment(world);env.step('fixed_0','reset')
            task=Game24ActionOnlyTask(world, None)
            prompt=task.action_prompt(self.author[0].Node(state=None,question='1 2 3 4'))
            self.assertRegex(prompt,r'(?m)^Act 1>')
            self.assertIn('Return only one action',prompt)
            child=world.at(('1 + 2 = 3',))
            sibling=world.at(('1 + 3 = 4',))
            terminal=world.at(('1 + 2 = 3','3 + 3 = 6','6 * 4 = 24'))
            self.assertEqual(terminal['reward'],0)
            self.assertFalse(terminal['terminal'])
            self.assertEqual(child['observation'],'numbers updated; numbers left: 3 4 3')
            self.assertEqual(sibling['observation'],'numbers updated; numbers left: 2 4 4')
            self.assertEqual(world.cache[('1 + 2 = 3',)]['observation'], child['observation'])

    def test_score_protocol_rejects_puzzle_digits_missing_and_conflicting_scores(self):
        task=DatasetLatsTask(SimpleNamespace(examples=''),None,None)
        with self.assertRaises(ValueError):task.value_outputs_unwrap(['Thus the correctness score is 2.'])
        with self.assertRaises(ValueError):task.value_outputs_unwrap(['Thus the correctness score is 10'])
        self.assertEqual(task.value_outputs_unwrap(['10']),1)
        self.assertEqual(task.value_outputs_unwrap(['2']),.2)
        # Native LATS uses the first evaluation sample, not an adapter average.
        self.assertEqual(task.value_outputs_unwrap(['2','9']),.2)
        for raw in ['The numbers are 1 4 10 10.', 'Thus the correctness score is 11.',
                    'Thus the correctness score is 2.\nThus the correctness score is 10.',
                    'Subtract 4 from 10. Thus the correctness score is 2.']:
            with self.assertRaises(ValueError):task.value_outputs_unwrap([raw])
        self.assertEqual(len(task.score_protocol_failures), 6)
        game_task=Game24ActionOnlyTask(
            SimpleNamespace(examples='', env=SimpleNamespace(puzzle='1 2 3 4')), None)
        with self.assertRaises(ValueError):game_task.value_outputs_unwrap(['Thus the correctness score is 0.'])
        self.assertEqual(game_task.value_outputs_unwrap(['0']),0)
        alf_task=AlfWorldActionOnlyTask(
            SimpleNamespace(examples='', cache={(): {'observation':'Root'}}), None)
        self.assertEqual(alf_task.value_outputs_unwrap(['7']),.7)
        with self.assertRaises(ValueError):alf_task.value_outputs_unwrap(['The correctness score is 7'])
        self.assertEqual(len(alf_task.score_protocol_failures), 1)
        self.assertEqual(alf_task.parse_action('\n  go to desk 1'), 'go to desk 1')

    def test_game24_parser_scans_full_output_for_complete_actions(self):
        self.assertEqual(Game24ActionOnlyTask.parse_action('Action: 1 + 2 = 3'), '1 + 2 = 3')
        self.assertEqual(Game24ActionOnlyTask.parse_action('Act 2> answer: (1 + 2) * (3 + 4) = 21'),
                         'answer: (1 + 2) * (3 + 4) = 21')
        self.assertEqual(Game24ActionOnlyTask.parse_action(
            '\nLet me check the current numbers.\nAction: 1 + 2 = 3\nObservation: ignored'),
            '1 + 2 = 3')
        self.assertEqual(Game24ActionOnlyTask.parse_action(
            'The next candidate is Action: 1 + 2 = 3'), '1 + 2 = 3')
        self.assertEqual(Game24ActionOnlyTask.parse_action(
            '\nexplanation only\nAction: not an action'),
            'explanation only\nAction: not an action')
        self.assertEqual(Game24ActionOnlyTask.parse_action('\n'), '')

    def test_nondefault_cli_reaches_native_expansion_evaluation_and_tokens(self):
        _, req = cli('game24', ['--candidates','2','--eval-samples','2','--iterations','3','--max-output-tokens','73'])
        config = validate_request(json.loads(json.dumps(req)))
        class Responses:
            def __init__(self): self.calls = []; self.actors = 0
            def generate(self, prompt, cap):
                self.calls.append((prompt, cap))
                if prompt.endswith('Score:'): text = '5'
                elif 'This attempt failed.' in prompt: text = 'Try finish after go.'
                else:
                    self.actors += 1
                    text = 'bad' if self.actors==1 else 'finish' if 'Receipt go' in prompt else 'go'
                return dict(raw_output=text)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            directory=Path(temp); transport=Responses()
            tokenizer=Tokenizer('new-model')
            model=Model(tokenizer,transport,config,.1,directory)
            result=adapted_search(World(directory),self.author,model,config)
            self.assertTrue(result['success'])
            self.assertTrue(result['reflection_map'])
            self.assertTrue(any('Try finish after go.' in p and 'Current branch:' in p for p,_ in transport.calls))
            self.assertTrue(all(cap==73 for p,cap in transport.calls))
            self.assertEqual(tokenizer.chat_template_calls,[])
            logical=[json.loads(line) for line in (directory/'logical_calls.jsonl').read_text().splitlines()]
            self.assertEqual(logical[0]['samples'],4)  # Native root multiplier.
            self.assertIn(2,[r['samples'] for r in logical])

    def test_model_samples_use_sequential_transport_calls(self):
        class ConcurrentTransport:
            def __init__(self):
                self.role='search'; self.lock=threading.Lock(); self.active=0; self.maximum=0
            def generate(self, prompt, cap):
                with self.lock:
                    self.active+=1; self.maximum=max(self.maximum,self.active)
                time.sleep(.02)
                with self.lock:
                    index=self.active; self.active-=1
                return {'raw_output':f'sample {index}'}
        with tempfile.TemporaryDirectory() as temp:
            transport=ConcurrentTransport()
            model=Model(Tokenizer(),transport,SearchConfig(),.1,Path(temp))
            outputs=model.gpt('task',n=4)
            self.assertEqual(len(outputs),4)
            self.assertEqual(transport.maximum,1)

    def test_game24_native_environment_and_branch_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp); csv=directory/'puzzles.csv'
            csv.write_text('Puzzles\n1 2 3 4\n2 3 4 5\n')
            _,req=cli('game24',['--csv',str(csv),'--task-ids','0','1'])
            template=load_inputs(req)
            a=create_world(req,0,template,None,directory)
            b=create_world(req,1,template,None,directory)
            self.assertEqual(a.cache[()]['observation'],'1 2 3 4')
            self.assertEqual(b.cache[()]['observation'],'2 3 4 5')
            a.at(('1 + 2 = 3',));a.at(('1 + 3 = 4',))
            self.assertEqual(a.env.public_state()['numbers'], ['2', '4', '4'])
            self.assertEqual(b.env.public_state()['numbers'], ['2', '3', '4', '5'])
            self.assertEqual(a.env.max_actions, 12)
            self.assertEqual(a.horizon, 12)

    def test_alfworld_real_prefix_replay_contract(self):
        class Env:
            def reset(self): self.actions=[]; return ['Header\n\nRoot'],{}
            def step(self, actions):
                self.actions+=actions
                return ['Receipt '+','.join(self.actions)],[0],[False],{'won':[False]}
        from source import search_environments as environments
        from concurrent.futures import Future
        env=Env()
        class Pool:
            def submit(self, function, *args):
                future=Future()
                try: future.set_result(function(*args))
                except Exception as error: future.set_exception(error)
                return future
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(environments,'_worker_environment',return_value=env):
            world=BranchWorld.__new__(BranchWorld)
            world.dataset='alfworld';world.directory=Path(temp);world.horizon=50
            world.environment_pool=Pool();world.gamefile='game';world.env_config={}
            world.cache={():dict(observation='Root',reward=0,terminal=False)};world.snapshots={():None}
            world.replay_calls=world.transition_calls=0
            world.at(('a','b'));world.at(('a','c'))
            self.assertEqual(env.actions,['a','c'])
            self.assertEqual(world.replay_calls,2)
            world.cache[('a',)]['observation']='corrupted'
            with self.assertRaisesRegex(ValueError,'differs'): world.at(('a','d'))

    def test_alfworld_root_and_branches_never_create_environment_in_task_thread(self):
        from source.search_environments import run_isolated_alfworld_root, run_isolated_alfworld_job
        pool=Mock()
        pool.submit.return_value.result.side_effect=[
            'Root', ('look',dict(observation='Room',reward=0,terminal=False))]
        template=dict(games=['/data/pick_and_place-test/trial/game.tw-pddl'],config={})
        with tempfile.TemporaryDirectory() as temp, \
             patch('source.search_adaptations.get_alfworld_candidate_pool',return_value=pool), \
             patch('source.datasets.Environment',side_effect=AssertionError('parent created environment')):
            world=BranchWorld('alfworld',0,template,None,None,Path(temp))
            self.assertEqual(world.cache[()]['observation'],'Root')
            self.assertEqual(world.at(('look',))['observation'],'Room')
            world.close()
        self.assertIs(pool.submit.call_args_list[0].args[0],run_isolated_alfworld_root)
        self.assertIs(pool.submit.call_args_list[1].args[0],run_isolated_alfworld_job)
        self.assertFalse(hasattr(world,'env'))

    def test_alfworld_pool_400_is_shared_and_spawn_isolated(self):
        from source import search_adaptations as adaptation
        with patch.object(adaptation,'_ALFWORLD_CANDIDATE_POOL',None), \
             patch.object(adaptation,'ProcessPoolExecutor') as executor:
            pool=adaptation.configure_alfworld_candidate_pool(400)
            self.assertIs(adaptation.get_alfworld_candidate_pool(),pool)
            self.assertIs(adaptation.configure_alfworld_candidate_pool(400),pool)
            executor.assert_called_once()
            self.assertEqual(executor.call_args.kwargs['max_workers'],400)
            self.assertEqual(executor.call_args.kwargs['mp_context'].get_start_method(),'spawn')
            adaptation.shutdown_alfworld_candidate_pool()
            pool.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
            with self.assertRaises(RuntimeError): adaptation.get_alfworld_candidate_pool()

    def test_webshop_native_search_keeps_reward_and_encoded_query(self):
        query='clock 12:30 / blue + red'
        url=native_url('http://site:3001',dict(session='fixed_8',page_type='search',query_string=query))
        self.assertEqual(unquote(unquote(urlsplit(url).path.split('/')[3])),query)
        class Env:
            def __init__(self): self.sessions={}
            def clone_state(self): return copy.deepcopy(self.sessions)
            def step(self,idx,action):
                self.sessions[idx]={'action':action}
                return ('CURRENT_UNIQUE_SHOPPING_TASK' if action=='reset' else 'Score:1',int(action!='reset'),action!='reset')
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch.object(self.author[0],'webshopEnv',Env):
            directory=Path(temp);config=SearchConfig(candidates=2)
            transport=Transport(['click[Buy Now]']*40)
            result=native_search(8,self.author,'http://site:3001',Model(Tokenizer(),transport,config,.1,directory),directory,config)
            self.assertEqual(result['reward'],1)
            self.assertEqual(result['final_public_reward'],1)
            self.assertTrue((directory/'search_tree.json').exists())
            self.assertIn('CURRENT_UNIQUE_SHOPPING_TASK',transport.calls[0][0])

    def test_webshop_invalid_actions_finish_two_bounded_rounds(self):
        class Env:
            def __init__(self): self.sessions={}
            def clone_state(self): return copy.deepcopy(self.sessions)
            def step(self,idx,action):
                if action=='reset':
                    self.sessions[idx]={'page_type':'init','session':idx}
                    return ('UNIQUE_REAL_TASK',0,False)
                raise AssertionError('invalid action evidence')
        class InvalidModel:
            temperature=.1
            tokenizer=Tokenizer()
            def for_role(inner,role):
                def call(prompt,n=1,**kwargs):
                    if role=='candidate':
                        self.assertIn('UNIQUE_REAL_TASK',prompt)
                        return ['not an action']*n
                    return ['5']*n
                return call
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch.object(self.author[0],'webshopEnv',Env):
            directory=Path(temp)
            result=native_search(8,self.author,'http://site',InvalidModel(),directory,
                                 SearchConfig(candidates=2,iterations=2,round_actions=15))
            rows=[json.loads(x) for x in (directory/'rounds.jsonl').read_text().splitlines()]
            ends=[r for r in rows if r['event']=='round_finished']
            self.assertEqual(len(ends),2)
            self.assertTrue(all(r['path_actions']==15 for r in ends))
            self.assertFalse(result['success'])

    def test_round_budget_counts_expansion_and_renews_from_selected_prefix(self):
        core=bind_module(self.author[0])
        core.env=SimpleNamespace(sessions={},step=lambda *a:('Task',0,False))
        created=[]
        Native=core.Node
        class Node(Native):
            def __init__(self,*args,**kw):
                super().__init__(*args,**kw);created.append(self)
        core.Node=Node
        # Force a non-root recovery prefix so an absolute-depth cap would fail.
        selections=[]
        def select(root):
            node=root if not selections else next(n for n in created if n.depth==2)
            selections.append(node)
            return node
        core.select_node=select
        def expand(node,*args):
            child=Node(state=dict(action='a',observation='receipt'),question='Task',parent=node)
            node.children.append(child)
        core.expand_node=expand
        core.evaluate_node=lambda *a:None
        def rollout(node,*args,max_depth):
            for _ in range(max_depth):
                expand(node);node=node.children[-1]
            return node
        core.rollout=rollout
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);install_round_search(core,5,directory)
            result=core.lats_search(SimpleNamespace(),None,'fixed_0',iterations=2)
            rows=[json.loads(x) for x in (directory/'rounds.jsonl').read_text().splitlines()]
        ends=[r for r in rows if r['event']=='round_finished']
        self.assertEqual([r['path_actions'] for r in ends],[5,5])
        self.assertEqual([r['endpoint_depth'] for r in ends],[5,7])
        self.assertEqual(len(selections),2)
        self.assertEqual(result[2],0)
        self.assertEqual(len(core.failed_trajectories),1)

    def test_game24_budget_cutoff_does_not_poison_environment_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);csv=directory/'p.csv';csv.write_text('Puzzles\n1 2 3 4\n')
            _,req=cli('game24',['--csv',str(csv),'--task-ids','0','--round-actions','3'])
            self.assertEqual(validate_request(req).round_actions,3)
            world=create_world(req,0,load_inputs(req),None,directory)
            prefix=tuple(['bad']*12)
            self.assertFalse(world.at(prefix)['terminal'])
            result=world.at(prefix+('answer: (1 + 2 + 3) * 4 = 24',))
            self.assertEqual(result['reward'],1)

    def test_two_game24_rounds_deliver_reflection_without_turning_value_into_reward(self):
        class Replies:
            def __init__(self):self.calls=[]
            def generate(self,prompt,cap):
                self.calls.append(prompt)
                text=('10' if prompt.endswith('Score:') else
                      'Use two distinct available occurrences.' if 'This attempt failed.' in prompt else
                      'Action: 1 + 1 = 2')
                return dict(raw_output=text)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            directory=Path(temp);csv=directory/'p.csv';csv.write_text('Puzzles\n1 2 3 4\n')
            _,req=cli('game24',['--csv',str(csv),'--task-ids','0'])
            config=validate_request(req);world=create_world(req,0,load_inputs(req),None,directory)
            replies=Replies();model=Model(Tokenizer(),replies,config,.1,directory)
            result=adapted_search(world,self.author,model,config)
            self.assertFalse(result['success']);self.assertEqual(result['reward'],0)
            self.assertTrue(any('Use two distinct available occurrences.' in p and 'Act 13>' in p for p in replies.calls))
            rounds=[json.loads(x) for x in (directory/'rounds.jsonl').read_text().splitlines()]
            self.assertEqual([r['path_actions'] for r in rounds if r['event']=='round_finished'],[12,12])

    def test_concurrent_model_bindings_and_raw_stop_records(self):
        def generate(name):
            with tempfile.TemporaryDirectory() as temp:
                directory=Path(temp);transport=Transport(['Action: '+name+'\nObservation: generated']*2)
                tokenizer=Tokenizer(name)
                model=Model(tokenizer,transport,SearchConfig(),.1,directory)
                outputs=model.gpt('task',n=2,stop='\nObservation:')
                samples=[json.loads(x) for x in (directory/'samples.jsonl').read_text().splitlines()]
                return outputs,transport.calls,samples,tokenizer.chat_template_calls
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(generate,['one','two']))
        for name,(outputs,calls,samples,chat_template_calls) in zip(['one','two'],results):
            self.assertEqual(outputs,['Action: '+name]*2)
            self.assertTrue(all(p=='task' for p,_ in calls))
            self.assertEqual(chat_template_calls,[])
            self.assertIn('Observation: generated',samples[0]['raw_output'])


if __name__=='__main__': unittest.main()
