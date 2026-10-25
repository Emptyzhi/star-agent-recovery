"""Run task-isolated LATS searches on an existing, verified vLLM resident."""
from .executors import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from .config import ROOT, validate_request
from .datasets import create_world, load_inputs
from .model import Model, validate_health
from .resident_transport import ResidentTransport, ROLE_STOP_MODES
from .http_control import exchange
from .lats_webshop import load_lats, execute_lats as execute_webshop
from .search_adaptations import (execute_lats as execute_adapted,
                                  configure_alfworld_candidate_pool,
                                  shutdown_alfworld_candidate_pool)


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def verify_allocation(parent, node):
    if os.environ.get('SLURM_JOB_ID') != parent or socket.gethostname() != node:
        raise RuntimeError('runner must execute inside the selected Slurm compute allocation')
    if not os.environ.get('SLURM_STEP_ID', '').isdecimal():
        raise RuntimeError('a numeric Slurm compute step is required')
    state = subprocess.check_output(['squeue', '--jobs', parent, '--noheader', '--format=%T'], text=True).strip()
    if state != 'RUNNING':
        raise RuntimeError('parent allocation is not RUNNING')


def get_health(endpoint, req, parent, node):
    health = exchange(endpoint, 'health', {})
    validate_health(health, req)
    if str(health['parent_job_id']) != parent or health['hostname'] != node:
        raise ValueError('resident is outside the verified allocation/node')
    if not str(health['slurm_step_id']).isdecimal():
        raise ValueError('resident is not inside a compute step')
    return health


def run_task(task_id, req, directory, template, author, tokenizer, health, endpoint, lease,
             environment_pool=None, request_limiter=None, request_pool=None, api_client=None):
    directory.mkdir()
    started = time.perf_counter()
    config = validate_request(req)
    transport = ResidentTransport(endpoint, lease, req['sampling'], health, directory,
                                  request_limiter=request_limiter, session=api_client)
    identity = dict(task_id=task_id, run_id=req['run_id'], method='lats', dataset=req['dataset'],
                    model_path=req['model_path'], resident=health)
    save(directory/'identity.json', identity)
    world = None
    try:
        model = Model(tokenizer, transport, config, req['sampling']['temperature'], directory,
                      request_pool=request_pool)
        if req['dataset'] == 'webshop':
            result = execute_webshop(task_id, author, template['website'], model, directory, config,
                                     environment_pool)
        else:
            world = create_world(req, task_id, template, transport, directory)
            result = execute_adapted(world, author, model, config, environment_pool)
            result.update(round_path_actions=config.round_actions or world.horizon,
                          unique_environment_transitions=world.transition_calls,
                          environment_replay_steps=world.replay_calls)
        result['dataset_adaptation'] = req['dataset'] != 'webshop'
        result['budget_policy'] = 'fresh_path_budget_per_iteration_including_expansion'
        result['search_iterations_limit'] = config.iterations
        result['round_path_actions'] = config.round_actions
        summary = dict(complete=True, identity=identity, result=result,
                       success=result['success'],benchmark_reward=result['reward'],
                       cost=transport.costs(time.perf_counter()-started), logical_model_calls=model.logical_calls)
        save(directory/'summary.json', summary)
        print(json.dumps(dict(event='task_complete', task=task_id,
                              reward=result['reward'])), flush=True)
        return summary
    except Exception as error:
        # Preserve the failure, then propagate it; never fabricate a task score.
        save(directory/'fatal.json', dict(identity=identity, error=repr(error),
             traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started,
             completed_model_records=transport.records))
        raise
    finally:
        try:
            if world is not None:
                world.close()
        finally:
            transport.close()


def run_shard(req, directory, gpu_index, parent, node):
    import requests
    validate_request(req)
    verify_allocation(parent, node)
    endpoint = req['endpoints'][gpu_index]
    health = get_health(endpoint, req, parent, node)
    tasks = req['task_ids'][gpu_index::len(req['endpoints'])]
    if not tasks:
        raise ValueError('empty shard')
    directory.mkdir(parents=True, exist_ok=False)
    author = load_lats()
    template = load_inputs(req)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(req['model_path'], trust_remote_code=req['trust_remote_code'], local_files_only=True)
    save(directory/'run_config.json', dict(request=req, health=health, tasks=tasks,
         dataset_adaptation=req['dataset'] != 'webshop', prompt_transport='vllm_http_raw_text',
         candidate_pipeline=False,role_stop_modes=ROLE_STOP_MODES,
         model_request_pool_scope='one sequential request per task; no candidate request pool',
         environment_seed_scope='environment initialization; no model/task reseeding',
         search_author_revision='853d81614607dd27433faf17c7b0a7d660f95d22'))
    lease = req['run_id']+'_g'+str(gpu_index)
    exchange(endpoint, 'begin_run', dict(experiment_id=lease))
    failures, results = [], []
    candidate_pool_configured=False
    if req['dataset']=='alfworld':
        configure_alfworld_candidate_pool(min(req['workers_per_gpu'], 32))
        candidate_pool_configured=True
    try:
        with ExitStack() as stack:
            api_client = None
            environment_pool=None
            if req['dataset'] in ('game24','webshop'):
                environment_pool=stack.enter_context(ThreadPoolExecutor(
                    max_workers=req['workers_per_gpu'],thread_name_prefix='candidate_env'))
            pool=stack.enter_context(ThreadPoolExecutor(max_workers=req['workers_per_gpu']))
            request_pool=None
            request_limiter = BoundedSemaphore(req['workers_per_gpu'])
            futures = {pool.submit(run_task, task, req, directory/f'task_{task:03d}', template,
                       author, tokenizer, health, endpoint, lease, environment_pool,
                       request_limiter=request_limiter, request_pool=request_pool,
                       api_client=api_client): task for task in tasks}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    failures.append(dict(task=futures[future], error=repr(error)))
        save(directory/'run_done.json', dict(complete=not failures, tasks=tasks,
             completed=len(results), failures=failures))
    finally:
        if candidate_pool_configured:
            shutdown_alfworld_candidate_pool()
        exchange(endpoint, 'end_run', dict(experiment_id=lease))
    return int(bool(failures))


def dispatch(req, output, parent, node):
    validate_request(req)
    verify_allocation(parent, node)
    health = [get_health(endpoint, req, parent, node) for endpoint in req['endpoints']]
    if len({h['gpu_uuid'] for h in health}) != len(health):
        raise ValueError('each endpoint must belong to a different physical GPU')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'request.json', req)
    processes, logs = [], []
    try:
        for gpu in range(len(req['endpoints'])):
            if not req['task_ids'][gpu::len(req['endpoints'])]:
                continue
            log = (output/f'runner_{gpu}.log').open('x', encoding='utf-8')
            logs.append(log)
            command = [sys.executable, '-X', 'utf8', '-B', str(ROOT/'run_shard.py'),
                       '--request', str(output/'request.json'), '--output-dir', str(output/f'group_{gpu}'),
                       '--gpu-index', str(gpu), '--parent-job-id', parent, '--node', node]
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    summaries = [json.loads(p.read_text(encoding='utf-8')) for p in output.glob('group_*/task_*/summary.json')]
    ids = [s['identity']['task_id'] for s in summaries]
    complete = not any(codes) and sorted(ids) == sorted(req['task_ids']) and all(s['complete'] for s in summaries)
    count = len(summaries)
    metrics = None
    if count:
        metrics = {key: sum(s['result'][key] for s in summaries)/count for key in
                   ('reward', 'success')}
    benchmark_metrics = metrics if complete else None
    save(output/'summary.json', dict(complete=complete, run_id=req['run_id'], exit_codes=codes,
         requested_tasks=len(req['task_ids']), scored_tasks=count, metrics_over_scored_tasks=metrics,
         metrics_over_requested_tasks=benchmark_metrics,
         fatal_tasks=len(list(output.glob('group_*/task_*/fatal.json')))))
    save(output/'dispatch_done.json', dict(run_id=req['run_id'], dataset=req['dataset'],
         method='lats', exit_codes=codes, complete=complete))
    return int(not complete)
