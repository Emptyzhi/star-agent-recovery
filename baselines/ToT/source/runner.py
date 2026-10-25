"""Run task-isolated BFS searches on an existing, verified vLLM resident."""
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
from .environments import create_task, load_inputs
from .model import Model, validate_health
from .resident_transport import ResidentTransport, ROLE_STOP_MODES
from .http_control import exchange
from .search import load_native, solve
from .isolated_environments import environment_executor


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


def run_task(task_id, req, directory, template, bfs, tokenizer, health, endpoint, lease,
             environment_pool, request_limiter=None, request_pool=None, api_client=None):
    directory.mkdir()
    started = time.perf_counter()
    config = validate_request(req)
    transport = ResidentTransport(endpoint, lease, req['sampling'], health, directory,
                                  request_limiter=request_limiter, session=api_client)
    identity = dict(task_id=task_id, run_id=req['run_id'], dataset=req['dataset'],
                    model_path=req['model_path'], resident=health)
    save(directory/'identity.json', identity)
    env = model = None
    try:
        task, env = create_task(req, task_id, template, directory, environment_pool)
        model = Model(tokenizer, transport, config, req['sampling']['temperature'], directory,
                      request_pool=request_pool)
        result = solve(task_id, task, bfs, model, config, directory,
                       req['sampling']['temperature'], interactive=env is not None)
        save(directory/'search_tree.json', result.pop('search'))
        if env is not None:
            result.update(unique_environment_transitions=task.world.transition_calls,
                          environment_replay_steps=task.world.replay_calls,
                          final_receipt=task.receipt(result['outputs'][0]))
        summary = dict(complete=True, identity=identity, result=result,
                       success=result['first_ranked_success'],benchmark_reward=result['first_ranked_reward'],
                       cost=transport.costs(time.perf_counter()-started), logical_model_calls=model.logical_calls)
        save(directory/'summary.json', summary)
        print(json.dumps(dict(event='task_complete', task=task_id,
                              first_ranked_reward=result['first_ranked_reward'])), flush=True)
        return summary
    except Exception as error:
        # Preserve the failure, then propagate it; never fabricate a task score.
        save(directory/'fatal.json', dict(identity=identity, error=repr(error),
             traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started,
             completed_model_records=transport.records))
        raise
    finally:
        try:
            if model is not None:
                model.close()
            if env is not None:
                env.close()
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
    bfs, game24_class = load_native()
    template = load_inputs(req, game24_class)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(req['model_path'], trust_remote_code=req['trust_remote_code'], local_files_only=True)
    save(directory/'run_config.json', dict(request=req, health=health, tasks=tasks,
         dataset_adaptation=True, environment_seed_scope='environment initialization; no model/task reseeding',
         prompt_transport='vllm_http_raw_text', candidate_pipeline=False,role_stop_modes=ROLE_STOP_MODES,
         environment_workers=min(req['workers_per_gpu'], 32) if req['dataset']=='alfworld' else req['workers_per_gpu'],
         model_request_workers=req['workers_per_gpu'],
         model_request_pool_scope='one sequential request per task; no candidate request pool',
         environment_isolation='spawn_process' if req['dataset']=='alfworld' else 'private_coroutine_state'))
    lease = req['run_id']+'_g'+str(gpu_index)
    exchange(endpoint, 'begin_run', dict(experiment_id=lease))
    failures, results = [], []
    try:
        with ExitStack() as stack:
            api_client = None
            environment_pool = stack.enter_context(environment_executor(req['dataset'],
                min(req['workers_per_gpu'], 32) if req['dataset']=='alfworld' else req['workers_per_gpu']))
            request_pool = None
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=req['workers_per_gpu']))
            # One shard owns one endpoint; all task candidates share its admission cap.
            request_limiter = BoundedSemaphore(req['workers_per_gpu'])
            futures = {pool.submit(run_task, task, req, directory/f'task_{task:03d}', template,
                       bfs, tokenizer, health, endpoint, lease, environment_pool,
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
    complete = not any(codes) and sorted(ids) == sorted(req['task_ids'])
    count = len(summaries)
    metrics = None
    if count:
        metrics = {key: sum(s['result'][key] for s in summaries)/count for key in
                   ('first_ranked_reward', 'first_ranked_success', 'candidate_mean', 'any_candidate_success')}
    save(output/'summary.json', dict(complete=complete, run_id=req['run_id'], exit_codes=codes,
         requested_tasks=len(req['task_ids']), scored_tasks=count, metrics_over_scored_tasks=metrics,
         fatal_tasks=len(list(output.glob('group_*/task_*/fatal.json')))))
    return int(not complete)
