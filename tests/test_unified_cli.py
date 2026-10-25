"""Dataset defaults and nondefault options reach actual worker adapters."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE/'source')]
import run_star
import dispatch_star
from experiment_settings import Experiment, Sampling, settings_from_args
from scoped_recovery_batch import build_parser as web_parser
from test_game24.model_api import ResidentModelClient
from test_game24.prompts import author_reflection
from test_game24.game24_env import Game24Environment
from dataset_runtime import execute_alf, run_task, run_batch
from alf_rebased_v64 import RebasedEpisode
from icr_core import Generation


def arguments(dataset):
    flags = ['--dataset', dataset, '--model-path', '/models/other-model', '--endpoint', 'http://127.0.0.1:41581',
        '--output-dir', 'out', '--run-namespace', 'test', '--task-ids', '2,7',
        '--parent-job-id', 'test-job', '--node', 'test-node', '--gpu-index', '0']
    return flags + {'webshop':['--website','http://127.0.0.1:3001'],
                    'game24':['--csv','24.csv'], 'alfworld':['--manifest','manifest.json']}[dataset]


class NativeFixture:
    def __init__(self):
        self.stage = 0
        self.batch_env = self
    def info(self):
        return {'extra.gamefile':['game'], 'admissible_commands':[['look', 'take apple 1 from table 1',
                  'put apple 1 in/on table 2']], 'won':[self.stage == 2]}
    def reset(self):
        self.stage = 0
        return ['HEADER\n\nYou are in a room. You see table 1 and table 2.\nYour task is to: put a apple in/on table 2.'], self.info()
    def step(self, actions):
        action = actions[0]
        if action == 'take apple 1 from table 1':
            self.stage = 1
            text = 'You pick up the apple 1 from the table 1.'
        elif action == 'put apple 1 in/on table 2' and self.stage == 1:
            self.stage = 2
            text = 'You put the apple 1 in/on the table 2.'
        elif action == 'look': text = 'You see an apple 1 on table 1.'
        else: text = 'Nothing happens.'
        return [text], [0], [self.stage == 2], self.info()
    def close(self): pass


class UnifiedCLITests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('STAR_ALFWORLD_BASELINE_ROOT'), 'set ALFWorld archived source for parity')
    def test_alfworld_v16_is_preserved_except_budget_wiring_and_ablation_gates(self):
        baseline=Path(os.environ['STAR_ALFWORLD_BASELINE_ROOT'])/'source'
        for name in ('alf_goal_grounded_events.py','alf_operation_available.py','alf_discovery_controller.py',
                     'alf_event_timing_v64.py','alf_episode.py','alf_sparse_events.py','alf_discovery_handoff.py'):
            self.assertEqual((PACKAGE/'source'/name).read_text(), (baseline/name).read_text())
        expected=(baseline/'alf_simple_v64.py').read_text()
        expected=expected.replace('    def __init__(self, episode, oracle, ledger, *, task, emit):',
            '    RECOVERY_WINDOW = WINDOW\n    YELLOW_ACTION_BUDGET = YELLOW_ACTIONS\n'
            '    RED_ENABLED = True\n    EVENT_SUBTREE_ENABLED = True\n'
            '    def __init__(self, episode, oracle, ledger, *, task, emit):')
        for before,after in (
            ('action_limit=YELLOW_ACTIONS,','action_limit=self.YELLOW_ACTION_BUDGET,'),
            ('scope.action_serial < YELLOW_ACTIONS','scope.action_serial < self.YELLOW_ACTION_BUDGET'),
            ('k+WINDOW','k+self.RECOVERY_WINDOW'),
            ('fault_occurrence_id,k+self.RECOVERY_WINDOW,WINDOW,','fault_occurrence_id,k+self.RECOVERY_WINDOW,self.RECOVERY_WINDOW,'),
            ('new_actions=WINDOW','new_actions=self.RECOVERY_WINDOW'),
            ('max_recovery_actions=WINDOW','max_recovery_actions=self.RECOVERY_WINDOW')):
            expected=expected.replace(before,after)
        expected=expected.replace('self.episode.recovery_event_subtree = recovery_subtree(self)',
            "self.episode.recovery_event_subtree = (recovery_subtree(self)\n"
            "            if self.EVENT_SUBTREE_ENABLED else {'events': [], 'nodes': []})")
        expected=expected.replace('        self._reflect_and_restore()\n',
            "        if not self.RED_ENABLED:\n"
            "            self.status = ('INITIAL_EMPTY_ACTOR_OUTPUT' if self.empty_actor_failure is not None else\n"
            "                           'TERMINAL_FAILURE' if self.episode.snapshot()['done'] else\n"
            "                           'INITIAL_WINDOW_EXHAUSTED')\n"
            "            self.emit('red_disabled_initial_attempt_finished', status=self.status)\n"
            "            return self.to_dict()\n"
            "        self._reflect_and_restore()\n")
        self.assertEqual((PACKAGE/'source/alf_native_v16.py').read_text(),expected)

    def test_dataset_irrelevant_options_are_not_silently_ignored(self):
        args=run_star.build_parser().parse_args(arguments('game24')+['--website','http://localhost:3001'])
        with self.assertRaisesRegex(ValueError,'does not apply'):run_star.resolve(args)

    def test_concurrent_runner_keeps_task_clients_independent(self):
        from threading import Barrier, Lock
        barrier, lock = Barrier(2), Lock()
        active = 0
        peak = 0
        def one(task, *unused):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=5)
            with lock:active -= 1
            return dict(success=True, final_public_reward=1)
        with tempfile.TemporaryDirectory() as temp:
            args=run_star.build_parser().parse_args(arguments('game24'))
            args.output_dir=Path(temp)/'run'
            args.workers_per_gpu=2
            cfg,sampling,tasks=run_star.resolve(args)
            with patch('dataset_runtime.bind_runtime',return_value=({}, {}, None, {})), \
                    patch('dataset_runtime.load_inputs',return_value=({2:'1 1 4 6',7:'1 1 4 6'},None)), \
                    patch('dataset_runtime.run_task',side_effect=one):
                self.assertEqual(run_batch(args,cfg,sampling,tasks),0)
            self.assertEqual(peak,2)
            done=json.loads((args.output_dir/'run_done.json').read_text())
            self.assertEqual(done['completed'],2)

    def test_game24_task_runner_writes_costs_and_releases_lease(self):
        def generated(request):
            return Generation(request,'answer: (1 - 1) + 4 * 6 = 24',10,12,
                              {'queue_time_seconds':0,'model_active_step_seconds':0})
        for cleanup_error in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                args=run_star.build_parser().parse_args(arguments('game24'))
                args.output_dir=Path(temp)
                cfg,sampling,_=run_star.resolve(args)
                client=Mock()
                client.begin_run.return_value={}
                client.generate.side_effect=generated
                if cleanup_error:client.close.side_effect=RuntimeError('lease cleanup failed')
                with patch('dataset_runtime.BoundResidentClient',return_value=client), \
                        patch('model_backend.encode_prompt',return_value={'input_ids':[1]*10}):
                    if cleanup_error:
                        with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
                            run_task(0,'1 1 4 6',args,cfg,sampling,{}, {},Mock(),None)
                        self.assertFalse((args.output_dir/'task_000/summary.json').exists())
                        self.assertTrue((args.output_dir/'task_000/fatal.json').exists())
                    else:
                        row=run_task(0,'1 1 4 6',args,cfg,sampling,{}, {},Mock(),None)
                        self.assertTrue(row['success'])
                        self.assertEqual(row['cost']['input_tokens'],10)
                        self.assertEqual(row['cost']['output_tokens'],12)
                        self.assertEqual(row['elapsed_seconds'],row['queue_excluded_elapsed_seconds'])
                client.close.assert_called_once()

    def test_dataset_defaults_and_common_model_flags(self):
        for dataset, window, yellow in [('webshop',15,1),('game24',12,1),('alfworld',50,4)]:
            args = run_star.build_parser().parse_args(arguments(dataset))
            cfg, sampling, tasks = run_star.resolve(args)
            self.assertEqual((cfg.initial_actions, cfg.recovery_actions, cfg.yellow_actions), (window,window,yellow))
            self.assertEqual(args.model_path, '/models/other-model')
            self.assertEqual(tasks, [2,7])
            self.assertEqual(sampling.top_k,20)

    def test_webshop_cli_forwards_complete_settings_to_existing_runner(self):
        args = run_star.build_parser().parse_args(arguments('webshop')+['--top-k','-1','--temperature','.3',
            '--actor-output-tokens','77','--red-selection-tokens','51','--yellow-actions','2','--workers-per-gpu','7'])
        cfg, sampling, _ = run_star.resolve(args)
        worker = web_parser().parse_args(run_star.webshop_arguments(args,cfg,sampling))
        self.assertEqual(settings_from_args(worker,Experiment),cfg)
        self.assertEqual(settings_from_args(worker,Sampling),sampling)
        self.assertEqual(worker.workers_per_gpu,7)

    def test_queue_dispatch_for_all_datasets_preserves_nondefault_values(self):
        for dataset in ('webshop','game24','alfworld'):
            request = dict(dataset=dataset,run_id='test',package=str(PACKAGE),task_ids=[2,7],
                endpoints=['http://127.0.0.1:41581','http://127.0.0.1:41582'],
                workers_per_gpu=8,environment_seed=99,
                experiment=dict(red_reflection_tokens=190,yellow_output_tokens=73),sampling=dict(top_k=-1))
            request.update({'webshop':dict(website='http://127.0.0.1:3001'),
                            'game24':dict(csv='24.csv'),'alfworld':dict(manifest='manifest.json')}[dataset])
            cfg,sampling,capacity = dispatch_star.resolve_request(request,dict(experiment={'initial_actions':15}))
            args = SimpleNamespace(model_path='/models/model2',dtype='bfloat16',parent_job_id='test-job',node='test-node',results_root=Path('out'))
            command = dispatch_star.worker_command(args,request,1,[7],cfg,sampling,capacity)
            actual = run_star.build_parser().parse_args(command[2:])
            parsed, sample, tasks = run_star.resolve(actual)
            self.assertEqual(parsed,cfg)
            self.assertEqual(sample,sampling)
            self.assertEqual(tasks,[7])
            self.assertEqual((actual.gpu_index,actual.workers_per_gpu),(1,8))
            if dataset=='game24':self.assertEqual((parsed.initial_actions,parsed.recovery_actions),(12,12))

    def test_dry_run_does_not_connect_or_load_model(self):
        with patch('model_backend.load_model_bundle',side_effect=AssertionError('unexpected model access')), redirect_stdout(StringIO()) as stream:
            self.assertEqual(run_star.main(arguments('game24')+['--dry-run']),0)
        data=json.loads(stream.getvalue())
        self.assertEqual(data['experiment']['initial_actions'],12)
        self.assertEqual(data['experiment']['recovery_actions'],12)
        self.assertIsNone(data['model_request_seed'])

    def test_game24_reflection_binds_current_task_and_preserves_failure_evidence(self):
        log='Here is the task:\nInput: 1 1 4 6\nAction: 1 + 1 = 2\nSTATUS: FAIL'
        prompt = author_reflection(log, task='1 1 4 6')
        evidence, objective = prompt.split('[CURRENT_GAME24_TASK]', 1)
        self.assertIn(log.split('Here is the task:', 1)[1].strip(), evidence)
        self.assertTrue(objective.startswith('\nInput: 1 1 4 6\n'))
        self.assertIn('Do not select a checkpoint here.', evidence)
        self.assertIn('using each original occurrence once', evidence)
        self.assertIn('using all four original occurrences', objective)
        self.assertTrue(objective.endswith('\nNew plan>\n'))

    def test_game24_native_direct_final_answer_is_not_forced_to_wait(self):
        e=Game24Environment('1 1 4 6')
        self.assertEqual(e.execute('answer: (1 - 1) + 4 * 6 = 24').reward,1)

    def test_resident_adapter_uses_existing_protocol_and_real_queue_measurement(self):
        client=Mock()
        def generate(request):return Generation(request,'1 - 1 = 0',10,6,{'queue_time_seconds':.25})
        client.generate.side_effect=generate
        adapter=ResidentModelClient(client,Mock(),prompt_limit=1000,context_limit=2000)
        adapter.count_tokens=lambda p:10
        result=adapter.generate('Action:',role='yellow_action',max_new_tokens=77,stop=('\n',))
        request=client.generate.call_args.args[0]
        self.assertEqual((request.route,request.stop_mode,request.output_limit),('normal','first_line',77))
        self.assertEqual(result.queue_seconds,.25)
        adapter.generate('choose',role='red_checkpoint',max_new_tokens=51,stop=('\n',),commands=('CHECKPOINT 0',))
        self.assertEqual(client.generate.call_args.args[0].route,'command_trie')

    def test_alfworld_config_reaches_native_controller_and_restoration(self):
        cfg=Experiment(initial_actions=1,recovery_actions=3,yellow_actions=2,
            actor_output_tokens=77,red_reflection_tokens=190,red_selection_tokens=51,
            environment_call_limit=100)
        requests=[]
        outputs=iter(['look','Take the apple and put it on table 2.','ACTION 0',
                      'take apple 1 from table 1','put apple 1 in/on table 2'])
        def generate(request):
            requests.append(request)
            return Generation(request,next(outputs),10,5,{})
        client=SimpleNamespace(generate=generate)
        tokenizer=Mock()
        tokenizer.encode.side_effect=lambda text,**kw:list(text.encode())
        tokenizer.decode.side_effect=lambda tokens,**kw:bytes(tokens).decode()
        formatter=lambda messages:'\n'.join(m['content'] for m in messages)
        episode=RebasedEpisode(NativeFixture(),'game','ACT EXAMPLE',formatter,action_limit=1,emit=lambda *a,**k:None)
        args=SimpleNamespace(prompt_limit=24000,context_limit=32768,run_namespace='alf_test')
        with patch('dataset_runtime.alf_episode',return_value=episode), \
                patch('dataset_runtime.encode_prompt',side_effect=lambda t,p:{'input_ids':list(p.encode())}), ExitStack() as stack:
            result=execute_alf(0,'game',client,tokenizer,{}, {},cfg,args,lambda *a,**k:None,stack)
        self.assertTrue(result['success'])
        self.assertEqual([r.output_limit for r in requests],[77,190,51,77,77])
        self.assertEqual(result['state']['forward_trials'][0]['forward_actions_granted'],3)
        self.assertEqual(result['state']['effective_yellow_action_budget'],2)

    def test_alfworld_empty_actor_root_restore_preserves_v16_fix(self):
        cfg=Experiment.for_dataset('alfworld')
        outputs=iter(['','Use the visible apple.','take apple 1 from table 1','put apple 1 in/on table 2'])
        client=SimpleNamespace(generate=lambda request:Generation(request,next(outputs),10,5,{}))
        episode=RebasedEpisode(NativeFixture(),'game','ACT EXAMPLE',
            lambda ms:'\n'.join(m['content'] for m in ms),action_limit=49,emit=lambda *a,**k:None)
        args=SimpleNamespace(prompt_limit=24000,context_limit=32768,run_namespace='empty_alf')
        with patch('dataset_runtime.alf_episode',return_value=episode), \
                patch('dataset_runtime.encode_prompt',return_value={'input_ids':[1]}), ExitStack() as stack:
            result=execute_alf(0,'game',client,Mock(),{}, {},cfg,args,lambda *a,**k:None,stack)
        self.assertTrue(result['success'])
        self.assertEqual(result['state']['selected_fault_index'],0)

    def test_alfworld_yellow_custom_tokens_reach_actual_requests(self):
        cfg=Experiment.for_dataset('alfworld')
        from dataclasses import replace
        cfg=replace(cfg,yellow_diagnosis_tokens=61,yellow_output_tokens=73)
        calls=[]
        outputs=iter(['bad command','bad command','Take the visible apple.',
                      'take apple 1 from table 1','put apple 1 in/on table 2'])
        def generate(request):
            calls.append(request)
            return Generation(request,next(outputs),10,5,{})
        tokenizer=Mock()
        tokenizer.encode.side_effect=lambda text,**kw:list(text.encode())
        tokenizer.decode.side_effect=lambda tokens,**kw:bytes(tokens).decode()
        episode=RebasedEpisode(NativeFixture(),'game','ACT EXAMPLE',
            lambda ms:'\n'.join(m['content'] for m in ms),action_limit=49,emit=lambda *a,**k:None)
        args=SimpleNamespace(prompt_limit=24000,context_limit=32768,run_namespace='yellow_alf')
        with patch('dataset_runtime.alf_episode',return_value=episode), \
                patch('dataset_runtime.encode_prompt',return_value={'input_ids':[1]}), ExitStack() as stack:
            result=execute_alf(0,'game',SimpleNamespace(generate=generate),tokenizer,{}, {},cfg,args,lambda *a,**k:None,stack)
        self.assertTrue(result['success'])
        self.assertEqual([r.output_limit for r in calls],[100,100,61,73,100])


if __name__=='__main__':unittest.main()
