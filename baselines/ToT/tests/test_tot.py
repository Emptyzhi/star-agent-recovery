from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import subprocess
import sys
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit

from source.author_binding import bind_module
from source.config import ROOT, SearchConfig, resolve, validate_request
from source.environments import BranchWorld, InteractiveTask, WebShop, WebShopTask, native_url, load_inputs
from source.game24_adapter import Game24Environment, Game24Task
from source.model import Model, validate_health
from source.resident_transport import ResidentTransport
from source.search import load_native, solve


def cli(dataset='game24', extra=()):
    data = {'game24': ['--csv', str(ROOT/'vendor/tot/src/tot/data/24/24.csv')],
            'alfworld': ['--manifest', 'tasks.json'], 'webshop': ['--website', 'http://environment:8080']}
    root = '5'
    for index, flag in enumerate(extra):
        if flag == '--candidates':
            root = extra[index+1]
    return resolve(['--dataset', dataset, '--local_llm_path', '/models/selected',
                    '--endpoints', 'http://resident:9000', '--run-id', 'test', '--iterations', '1',
                    '--method-generate','sample','--candidates','5','--beam-width','5',
                    '--eval-samples','3','--max-output-tokens','100',
                    '--root-candidates',root, *data[dataset], *extra])


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        raise AssertionError('ToT must send raw prompts without chat templates')


class SequenceTransport:
    def __init__(self, texts):
        self.texts, self.calls = iter(texts), []

    def generate(self, prompt, cap):
        self.calls.append((prompt, cap))
        return dict(raw_output=next(self.texts))


class FixtureEnv:
    def reset(self):
        self.actions = []
        return dict(observation='Root', reward=0, terminal=False)

    def step(self, action):
        self.actions.append(action)
        return dict(observation=','.join(self.actions), reward=int(action == 'win'),
                    terminal=action in ('win', 'lose'))

    def snapshot(self):
        return list(self.actions)

    def restore(self, actions):
        self.actions = list(actions)


class Contracts(unittest.TestCase):
    def test_webshop_framed_action_matches_lats_whitespace_handling(self):
        for raw in [' search[apple] ', '\tsearch[apple] ', ' Action: search[apple] ', 'action: search[apple]']:
            self.assertEqual(WebShopTask.normalize_action(raw),'search[apple]')
        self.assertEqual(WebShopTask.normalize_action(' click[not a visible product] '),
                         'click[not a visible product]')
        for raw in ['', ' ', 'search[apple]\nclick[x]']:
            with self.assertRaisesRegex(ValueError,'single-line action'):
                WebShopTask.normalize_action(raw)

    def test_shared_request_pool_bounds_many_tasks_without_nested_waits(self):
        lock=threading.Lock()
        active=peak=0
        threads=set()
        class Transport:
            def generate(self,prompt,cap):
                nonlocal active,peak
                with lock:
                    active+=1;peak=max(peak,active)
                    threads.add(threading.get_ident())
                time.sleep(.01)
                with lock:active-=1
                return dict(raw_output='likely')
        with tempfile.TemporaryDirectory() as folder,ThreadPoolExecutor(4) as requests, \
                ThreadPoolExecutor(8) as tasks:
            models=[]
            for i in range(8):
                directory=Path(folder)/str(i);directory.mkdir()
                models.append(Model(Tokenizer(),Transport(),
                    SearchConfig('sample',2,2,3,1,100,1,10),.1,directory,request_pool=requests))
            calls=[tasks.submit(model.gpt,'candidate',n=10) for model in models]
            for call in calls:self.assertEqual(call.result(timeout=10),['likely']*10)
            self.assertEqual(peak,4)
            self.assertEqual(len(threads),4)
            models[0].close()
            models[1].role='evaluation'
            futures=models[1].submit_gpt('score',n=3)
            self.assertEqual([future.result(timeout=5)[1] for future in futures],['likely']*3)
            for model in models:model.close()

    def test_text_only_compatibility_does_not_simulate_visual_models(self):
        env=dict(os.environ, PYTHONPATH=str(ROOT/'runtime_compat'))
        code=('import torchvision; '
              'assert torchvision.__version__ == "0.0-text-only"; '
              'torchvision.models.detection.MaskRCNN()')
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('visual torchvision models are unavailable',result.stderr)

    def test_new_defaults_apply_to_three_datasets(self):
        for dataset, flags in [('game24',['--csv','data.csv']),
                               ('webshop',['--website','http://site']),
                               ('alfworld',['--manifest','tasks.json'])]:
            _,request=resolve(['--dataset',dataset,'--model-path','/models/selected',
                '--endpoints','http://resident','--run-id','defaults',*flags])
            config=validate_request(request)
            self.assertEqual((config.method_generate,config.candidates,config.root_candidates,
                config.beam_width,config.eval_samples,config.max_output_tokens),
                ('sample',2,10,2,1,100))

    def test_propose_root_ten_then_two_per_retained_parent(self):
        class Transport:
            def __init__(self):self.prompts=[]
            def generate(self,prompt,cap):
                self.prompts.append(prompt)
                if prompt.startswith('Assess'):return dict(raw_output='likely')
                count=10 if 'exactly 10 alternative' in prompt else 2
                return dict(raw_output='\n'.join('move'+str(i) for i in range(count)))
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            directory=Path(folder);transport=Transport()
            config=SearchConfig('propose',2,2,1,2,1000,1,10)
            world=BranchWorld(FixtureEnv(),'webshop','',2,directory)
            solve(0,InteractiveTask(world),self.bfs,
                  Model(Tokenizer(),transport,config,.1,directory),config,directory,.1,True)
            expansions=[json.loads(line) for line in (directory/'expansions.jsonl').read_text().splitlines()]
            self.assertEqual([len(row['expanded_candidates']) for row in expansions],[10,2,2])
            self.assertEqual([row['explicit_limit'] for row in expansions],[10,2,2])
            self.assertEqual(sum(not p.startswith('Assess') for p in transport.prompts),3)

    def test_endpoint_limit_is_configured_in_request(self):
        _, req = cli('game24', ['--request-limit-per-endpoint','400'])
        validate_request(req)
        self.assertEqual(req['request_limit_per_endpoint'],400)
        req['request_limit_per_endpoint'] = 0
        with self.assertRaisesRegex(ValueError,'endpoint request limit'):
            validate_request(req)

    @classmethod
    def setUpClass(cls):
        cls.bfs, cls.game_class = load_native()

    def test_three_datasets_and_aliases(self):
        for dataset, depth, mode in [('game24', 12, 'sample'), ('alfworld', 49, 'sample'), ('webshop', 15, 'sample')]:
            _, req = cli(dataset)
            config = validate_request(json.loads(json.dumps(req)))
            self.assertEqual((config.max_depth, config.method_generate, config.beam_width), (depth, mode, 5))
            self.assertEqual(config.max_output_tokens, 100)
        _, req = cli(extra=['--method_generate', 'sample', '--n_select_sample', '2', '--n_evaluate_sample', '1', '--candidates', '3'])
        self.assertEqual(validate_request(req), SearchConfig('sample', 3, 2, 1, 12, 100, 1, 3))

    def test_sample_root_ten_and_parallel_scores_preserve_order(self):
        generation_barrier = threading.Barrier(10)
        scoring_barrier = threading.Barrier(10)
        class Transport:
            def __init__(self):
                self.lock=threading.Lock();self.actions=0;self.evaluations=0
            def generate(self,prompt,cap):
                self_cap=cap
                assert self_cap==100
                if prompt.startswith('Assess'):
                    with self.lock:self.evaluations+=1
                    scoring_barrier.wait(timeout=5)
                    return dict(raw_output='likely')
                with self.lock:
                    ordinal=self.actions;self.actions+=1
                generation_barrier.wait(timeout=5)
                return dict(raw_output='move'+str(ordinal))
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            directory=Path(folder);transport=Transport()
            config=SearchConfig('sample',2,2,1,1,100,1,10)
            world=BranchWorld(FixtureEnv(),'webshop','',1,directory)
            model=Model(Tokenizer(),transport,config,.1,directory)
            try:
                result=solve(0,InteractiveTask(world),self.bfs,model,config,directory,.1,True)
            finally:model.close()
            self.assertEqual((transport.actions,transport.evaluations),(10,10))
            self.assertEqual(len(result['search']['steps'][0]['new_ys']),10)
            logs=[json.loads(line) for line in (directory/'logical_calls.jsonl').read_text().splitlines()]
            self.assertEqual(sorted(row['call_id'] for row in logs),list(range(11)))
            self.assertEqual(logs[0]['samples'],10)
            self.assertEqual(result['search']['steps'][0]['values'],[1]*10)

    def test_cli_defaults_to_two_rounds_and_queue_requires_explicit_rounds(self):
        _, req = resolve(['--dataset', 'game24', '--csv', '/data/game24.csv',
            '--model-path', '/models/selected', '--endpoints', 'http://resident:9000', '--run-id', 'round-default'])
        self.assertEqual(validate_request(req).iterations, 2)
        del req['experiment']['search']['iterations']
        with self.assertRaises(TypeError):
            validate_request(req)

    def test_two_rounds_replay_from_root_with_fresh_path_budget(self):
        _, req = cli('webshop', ['--iterations', '2', '--max_depth', '2',
            '--candidates', '1', '--beam-width', '1', '--eval-samples', '1'])
        config = validate_request(json.loads(json.dumps(req)))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            directory = Path(folder)
            world = BranchWorld(FixtureEnv(), 'webshop', 'Examples', 2, directory)
            transport = SequenceTransport(['a', 'likely', 'b', 'likely', 'a', 'likely', 'win'])
            result = solve(0, InteractiveTask(world), self.bfs,
                Model(Tokenizer(), transport, config, .1, directory), config, directory, .1, True)
            self.assertEqual(result['rounds_executed'], 2)
            self.assertTrue(result['first_ranked_success'])
            self.assertEqual(world.transition_calls, 4)
            self.assertEqual([len(r['search']['steps']) for r in result['search']['rounds']], [2, 2])
            self.assertEqual(result['search']['rounds'][0]['outputs'], ['a\nb\n'])
            self.assertEqual(result['outputs'], ['a\nwin\n'])
            self.assertNotIn(('a', 'b'), world.cache)

    def test_success_stops_before_second_bfs_and_no_reranking(self):
        _, req = cli('webshop', ['--iterations', '2', '--max-depth', '1',
            '--candidates', '1', '--beam-width', '1', '--eval-samples', '1'])
        config = validate_request(req)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            directory = Path(folder)
            world = BranchWorld(FixtureEnv(), 'webshop', '', 1, directory)
            transport = SequenceTransport(['win'])
            result = solve(0, InteractiveTask(world), self.bfs,
                Model(Tokenizer(), transport, config, .1, directory), config, directory, .1, True)
            self.assertEqual(result['rounds_executed'], 1)
            self.assertEqual(len(transport.calls), 1)

    def test_remote_paths_and_round_settings_survive_export(self):
        _, req = cli('alfworld', ['--package', '/remote/tot', '--iterations', '2',
            '--alfworld-runtime', '/remote/runtime', '--max-depth', '49'])
        self.assertEqual(req['alfworld_config'], '/remote/tot/vendor/ga/test_alfworld/base_config.yaml')
        self.assertEqual(req['alfworld_runtime'], '/remote/runtime')
        self.assertEqual(validate_request(json.loads(json.dumps(req))).iterations, 2)

    def test_invalid_and_unknown_request_fields_rejected(self):
        _, req = cli()
        for key, value in [('task_ids', [0, 0]), ('workers_per_gpu', 0), ('dataset', 'unknown')]:
            bad = copy.deepcopy(req)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_request(bad)
        req['experiment']['search']['hidden_fallback'] = True
        with self.assertRaises(TypeError):
            validate_request(req)
        _, req = cli()
        req['sampling']['top_k'] = 0
        with self.assertRaises(ValueError):
            validate_request(req)

    def test_export_and_dry_run_do_not_contact_services(self):
        import run_tot
        with tempfile.TemporaryDirectory() as folder, patch('requests.get', side_effect=AssertionError('network')), patch('requests.post', side_effect=AssertionError('network')):
            for dataset in ('game24', 'alfworld', 'webshop'):
                args, req = cli(dataset)
                with patch('run_tot.resolve', return_value=(SimpleNamespace(write_request=None, dry_run=True), req)), redirect_stdout(io.StringIO()):
                    self.assertEqual(run_tot.main(), 0)
            path = Path(folder)/'request.json'
            with patch('run_tot.resolve', return_value=(SimpleNamespace(write_request=path, dry_run=False), req)), redirect_stdout(io.StringIO()):
                run_tot.main()
            self.assertEqual(json.loads(path.read_text()), req)

    def test_health_model_and_capacity_rejection(self):
        _, req = cli()
        health = dict(status='READY', fatal_error=None, engine='vllm', model_path='/models/selected',
                      max_num_seqs=40, max_new_tokens=1000)
        validate_health(health, req)
        for key, value in [('model_path', '/models/other'), ('max_num_seqs', 20), ('max_new_tokens', 99), ('status', 'LOADING')]:
            with self.assertRaises(ValueError):
                validate_health(dict(health, **{key: value}), req)

    def test_game24_uses_ga_action_only_protocol(self):
        csv_path = str(ROOT/'vendor/tot/src/tot/data/24/24.csv')
        env = Game24Environment(csv_path, 0)
        with tempfile.TemporaryDirectory() as folder:
            world = BranchWorld(env, 'game24', 'Act 1> example', 12, Path(folder))
            task = Game24Task(world)
            self.assertIn('Act 1>', task.cot_prompt_wrap('', ''))
            self.assertEqual(task.normalize_action('Act 2> 4 + 6 = 10'), '4 + 6 = 10')
            self.assertEqual(world.at(('1 + 1 = 2',))['observation'], 'numbers updated; numbers left: 4 6 2')
            self.assertEqual(world.at(('1 + 1 = 2', '2 * 6 = 12', '4 * 12 = 48'))['reward'], 0)

    def test_game24_sampling_normalizes_only_new_action_after_parent(self):
        _, req = cli('game24', ['--iterations', '1', '--max-depth', '2', '--candidates', '1',
                                '--beam-width', '1', '--eval-samples', '1'])
        config = validate_request(req)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            directory = Path(folder)
            world = BranchWorld(Game24Environment(str(ROOT/'vendor/tot/src/tot/data/24/24.csv'), 0),
                                'game24', '', 2, directory)
            transport = SequenceTransport([' \nExplanation\nAction: 1 + 1 = 2', 'likely',
                                           '\nAct 2> 2 * 6 = 12\nObservation: invented', 'likely'])
            result = solve(0, Game24Task(world), self.bfs,
                           Model(Tokenizer(), transport, config, .1, directory),
                           config, directory, .1, False)
            self.assertEqual(result['outputs'], ['1 + 1 = 2\n2 * 6 = 12\n'])

    def test_game24_empty_and_multiline_rejections_remain_single_edges(self):
        for raw in ['', 'Explanation only\nNo arithmetic action']:
            with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                directory=Path(folder)
                world=BranchWorld(Game24Environment(str(ROOT/'vendor/tot/src/tot/data/24/24.csv'),0),
                                  'game24','',2,directory)
                task=Game24Task(world)
                config=SearchConfig('sample',1,1,1,2,100,1)
                transport=SequenceTransport([raw,'likely','1 + 1 = 2','likely'])
                result=solve(0,task,self.bfs,Model(Tokenizer(),transport,config,.1,directory),
                             config,directory,.1,True)
                actions=task.actions(result['outputs'][0])
                self.assertEqual(actions,(raw,'1 + 1 = 2'))
                self.assertIn('invalid arithmetic action',world.cache[(raw,)]['observation'])
                self.assertEqual(world.transition_calls,2)

    def test_cli_controls_real_branching_evaluation_and_output_limit(self):
        class BranchTransport:
            def __init__(self):
                self.actor_calls, self.value_calls, self.caps = 0, 0, []

            def generate(self, prompt, cap):
                self.caps.append(cap)
                if prompt.startswith('Assess'):
                    self.value_calls += 1
                    return dict(raw_output='sure' if 'Observation: bad' in prompt else 'likely')
                self.actor_calls += 1
                if 'Observation: good' in prompt:
                    text = 'win'
                elif 'Observation: bad' in prompt:
                    text = 'lose'
                else:
                    text = 'bad' if self.actor_calls == 1 else 'good'
                return dict(raw_output=text+'\nDO NOT EXECUTE')

        for beam, success in [(1, False), (2, True)]:
            _, req = cli('webshop', ['--beam-width', str(beam), '--candidates', '2', '--eval-samples', '1',
                                     '--max-depth', '3', '--max-output-tokens', '77', '--temperature', '0.3'])
            config = validate_request(json.loads(json.dumps(req)))
            with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                directory = Path(folder)
                world = BranchWorld(FixtureEnv(), 'webshop', 'Examples', 3, directory)
                transport = BranchTransport()
                model = Model(Tokenizer(), transport, config, .3, directory)
                result = solve(0, InteractiveTask(world), self.bfs, model, config, directory, .3, True)
                self.assertEqual(result['first_ranked_success'], success)
                self.assertEqual(transport.value_calls, 2)
                self.assertEqual(transport.actor_calls, 2+2*beam)
                self.assertEqual(set(transport.caps), {77})
                self.assertEqual(len(result['search']['steps'][0]['select_new_ys']), beam)
                self.assertTrue(all('DO NOT' not in y for y in result['outputs']))

    def test_proposal_cap_is_explicit_and_uncapped_preserves_all_lines(self):
        for cap, count in [(None, 3), (2, 2)]:
            with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                directory = Path(folder)
                world = BranchWorld(FixtureEnv(), 'webshop', '', 1, directory)
                config = SearchConfig('propose', cap, 5, 1, 1, 31, 1)
                model = Model(Tokenizer(), SequenceTransport(['a\nb\nc']+['likely']*count), config, .1, directory)
                result = solve(0, InteractiveTask(world), self.bfs, model, config, directory, .1, True)
                self.assertEqual(len(result['outputs']), count)
                expansion = json.loads((directory/'expansions.jsonl').read_text())
                self.assertEqual(len(expansion['all_candidates']), 3)
                self.assertEqual(expansion['explicit_limit'], cap)

    def test_branch_siblings_snapshot_and_replay(self):
        for dataset in ('webshop', 'alfworld'):
            with tempfile.TemporaryDirectory() as folder:
                env = FixtureEnv()
                world = BranchWorld(env, dataset, '', 3, Path(folder))
                world.at(('a', 'b'))
                world.at(('a', 'c'))
                self.assertEqual(env.actions, ['a', 'c'])
                self.assertEqual(world.cache[('a', 'b')]['observation'], 'a,b')
                self.assertEqual(world.replay_calls, 2 if dataset == 'alfworld' else 0)
                if dataset == 'alfworld':
                    world.cache[('a',)]['terminal'] = True
                    with self.assertRaisesRegex(ValueError, 'replay differs'):
                        world.restore(('a',))

    def test_terminal_cannot_be_extended_and_budget_is_not_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            world = BranchWorld(FixtureEnv(), 'webshop', '', 1, Path(folder))
            self.assertFalse(world.at(('a',))['terminal'])
            with self.assertRaises(ValueError):
                world.at(('a', 'b'))
            self.assertTrue(world.at(('win',))['terminal'])
            with self.assertRaises(ValueError):
                world.at(('win', 'b'))

    def test_native_webshop_invalid_action_is_not_repaired(self):
        _, req = cli('webshop')
        template = load_inputs(req, self.game_class)
        self.assertNotIn('think[', template['examples'])
        with tempfile.TemporaryDirectory() as folder:
            env = WebShop(template['native_class'], req['website'], 1, Path(folder))
            env.raw.webshop_text = lambda **state: ('page', {})
            try:
                env.reset()
                before = env.snapshot()
                receipt = env.step('Action: search[not an accepted action prefix]')
                self.assertEqual(receipt['observation'], 'Invalid action!')
                self.assertEqual(env.snapshot(), before)
                env.step('search[red/blue 50% & small]')
                self.assertEqual(env.snapshot()['fixed_1']['query_string'], 'red/blue 50% & small')
            finally:
                env.close()

    def test_webshop_route_double_query_encoding(self):
        query = 'red/blue 50% & small'
        url = native_url('http://web', dict(page_type='search', session='fixed_3', query_string=query, page_num=1))
        encoded = urlsplit(url).path.split('/')[3]
        self.assertEqual(unquote(unquote(encoded)), query)
        self.assertIn('%252F', encoded)

    def test_resident_wire_sampling_seed_and_cost(self):
        _, req = cli(extra=['--top-k', '7', '--top-p', '0.8'])
        identity = dict(parent_job_id='1', hostname='gpu', gpu_uuid='UUID', service_pid=2,
                        service_instance_id='instance', slurm_step_id='3')
        raw = dict(gpu_identity={k: identity[k] for k in ('parent_job_id', 'hostname', 'gpu_uuid', 'slurm_step_id')},
                   runtime={k: identity[k] for k in ('service_pid', 'service_instance_id')},
                   effective_seed=None, task_seed=None, text='sure', raw_output='sure', output_token_limit=77,
                   output_tokens=1, output_token_ids=[23], input_tokens=5, decoding=req['sampling'],
                   queue_time_seconds=28, model_active_step_seconds=2)
        session = Mock()
        session.post.return_value.ok = True
        session.post.return_value.json.return_value = raw
        with tempfile.TemporaryDirectory() as folder:
            transport = ResidentTransport(req['endpoints'][0], 'lease', req['sampling'], identity, Path(folder), session=session)
            transport.generate('prompt', 77)
            payload = session.post.call_args.kwargs['json']
            self.assertEqual(payload['sampling']['top_k'], 7)
            self.assertEqual(payload['sampling']['top_p'], .8)
            self.assertIsNone(payload['task_seed'])
            self.assertIsNone(transport.costs(32)['queue_excluded_elapsed_seconds'])
            self.assertEqual(transport.costs(10)['model_queue_seconds'], 28)
            raw['effective_seed'] = 1
            with self.assertRaisesRegex(ValueError, 'reseeded'):
                transport.generate('prompt', 77)

    def test_parallel_native_bindings_do_not_mutate_shared_bfs(self):
        original = self.bfs.gpt
        def execute(marker):
            core = bind_module(self.bfs, gpt=lambda *args, **kwargs: [marker])
            return core.get_proposals(SimpleNamespace(propose_prompt_wrap=lambda *args: 'p'), '', '')
        with ThreadPoolExecutor(2) as pool:
            outputs = list(pool.map(execute, ['one', 'two']))
        self.assertEqual(outputs, [['one\n'], ['two\n']])
        self.assertIs(self.bfs.gpt, original)

    def test_compute_guard_rejects_local_execution(self):
        from source.runner import verify_allocation
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError):
                verify_allocation('123', 'gpu13')

    def test_cli_json_shard_worker_and_http_integration(self):
        from source.runner import run_shard
        from source.isolated_environments import environment_executor
        _, req = cli(extra=['--task-count', '2', '--workers-per-gpu', '3', '--max-num-seqs', '12',
                           '--candidates', '1', '--beam-width', '1', '--eval-samples', '1', '--max-depth', '1',
                           '--max-output-tokens', '37', '--top-k', '7', '--top-p', '0.8'])
        req = json.loads(json.dumps(req))
        health = dict(status='READY', fatal_error=None, engine='vllm', model_path=req['model_path'],
                      max_num_seqs=12, max_new_tokens=1000, parent_job_id='1', hostname='gpu',
                      gpu_uuid='UUID', service_pid=2, service_instance_id='instance', slurm_step_id='3')
        from openai.types.chat import ChatCompletion
        clients = []

        def client_factory(**options):
            self.assertEqual(options['max_retries'], 0)
            client = Mock()
            client.__enter__ = Mock(return_value=client)
            client.__exit__ = Mock(return_value=False)

            def create(**kwargs):
                output = 'likely' if kwargs['messages'][0]['content'].startswith('Evaluate') else '1 + 1 = 2'
                raw = dict(
                    gpu_identity={k: health[k] for k in ('parent_job_id', 'hostname', 'gpu_uuid', 'slurm_step_id')},
                    runtime=dict({k: health[k] for k in ('service_pid', 'service_instance_id')}, chat_template_applied=True),
                    effective_seed=None, task_seed=None, text=output, raw_output=output,
                    output_token_limit=kwargs['max_tokens'], output_tokens=1, output_token_ids=[23],
                    input_tokens=5, decoding=req['sampling'], queue_time_seconds=0, model_active_step_seconds=0,
                    finish_reason='eos', openai_finish_reason='stop')
                return ChatCompletion.model_validate(dict(id='test', object='chat.completion', created=1,
                    model=req['model_path'], choices=[dict(index=0, message=dict(role='assistant', content=output),
                    finish_reason='stop')], usage=dict(prompt_tokens=5, completion_tokens=1, total_tokens=6),
                    resident_response=raw))

            client.chat.completions.create.side_effect = create
            clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch('source.runner.verify_allocation'), patch('requests.get') as get, \
                patch('requests.post') as post, patch('openai.OpenAI', side_effect=client_factory), \
                patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()) as tokenizer, \
                patch('source.runner.environment_executor', wraps=environment_executor) as environments, \
                patch('source.runner.ThreadPoolExecutor', wraps=ThreadPoolExecutor) as pool:
            get.return_value.json.return_value = health
            directory = Path(folder)/'shard'
            self.assertEqual(run_shard(req, directory, 0, '1', 'gpu'), 0)
            self.assertEqual(pool.call_count,2)
            self.assertEqual(pool.call_args_list[0].kwargs,
                dict(max_workers=60,thread_name_prefix='tot_requests'))
            self.assertEqual(pool.call_args_list[1].kwargs,dict(max_workers=3))
            environments.assert_called_once_with('game24',3)
            tokenizer.assert_called_once_with('/models/selected', trust_remote_code=False, local_files_only=True)
            self.assertEqual(len(clients), 1)
            self.assertEqual(clients[0].chat.completions.create.call_count, 4)
            for call in clients[0].chat.completions.create.call_args_list:
                self.assertEqual(call.kwargs['max_tokens'], 37)
                self.assertEqual(call.kwargs['extra_body']['top_k'], 7)
                self.assertEqual(call.kwargs['top_p'], .8)
                self.assertEqual(call.kwargs['messages'][0]['role'], 'user')
            self.assertEqual([call.args[0].split('/')[-1] for call in post.call_args_list], ['begin_run', 'end_run'])
            self.assertEqual(len(list(directory.glob('task_*/summary.json'))), 2)
            self.assertTrue(json.loads((directory/'run_done.json').read_text())['complete'])
            metadata=json.loads((directory/'run_config.json').read_text())
            self.assertEqual(metadata['prompt_transport'],'openai_sdk_chat_completions')
            self.assertEqual(metadata['environment_workers'],3)


if __name__ == '__main__':
    unittest.main()
