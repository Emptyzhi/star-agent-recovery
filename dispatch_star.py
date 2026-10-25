"""Dispatch a versioned dataset request inside an existing two-GPU resident."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE/'source'))
from experiment_settings import Experiment, Sampling, settings_flags
from model_backend import add_model_arguments


def resolve_request(request, defaults):
    required = {'run_id', 'package', 'task_ids', 'dataset'}
    optional = {'environment_seed', 'workers_per_gpu', 'sampling', 'experiment',
                'context_limit', 'prompt_limit', 'csv', 'manifest', 'environment_config',
                'split', 'website', 'endpoints', 'prompt_format'}
    if not required <= set(request) or set(request)-required-optional:
        raise ValueError('dataset request has missing or unknown fields')
    if not re.fullmatch('[a-z0-9_]{1,100}', request['run_id']):
        raise ValueError('invalid run ID')
    tasks = request['task_ids']
    if request.get('prompt_format','plain') not in ('plain','native_chat'):
        raise ValueError('invalid model prompt format')
    if (type(tasks) is not list or not tasks or any(type(t) is not int or t < 0 for t in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError('task IDs must be unique original indices')
    values = Experiment.for_dataset(request['dataset']).as_dict()
    values.update(request.get('experiment', {}))
    if 'environment_seed' in request:
        if ('environment_seed' in request.get('experiment', {})
                and request['environment_seed'] != values['environment_seed']):
            raise ValueError('conflicting environment seeds')
        values['environment_seed'] = request['environment_seed']
    settings = Experiment(**values)
    sampling = Sampling(**dict(defaults.get('sampling', {}), **request.get('sampling', {})))
    capacity = {k: request[k] if k in request else defaults.get(k, v)
                for k, v in (('workers_per_gpu', 40), ('context_limit', 32768), ('prompt_limit', 32000))}
    if any(type(v) is not int or v < 1 for v in capacity.values()):
        raise ValueError('invalid capacity')
    if 'endpoints' not in request or len(request['endpoints']) != 2:
        raise ValueError('declare both loaded resident endpoints')
    return settings, sampling, capacity


def worker_command(args, request, gpu, tasks, settings, sampling, capacity):
    command = [sys.executable, str(Path(request['package'])/'run_star.py'),
        '--dataset', request['dataset'], '--model-path', args.model_path, '--dtype', args.dtype,
        '--endpoint', request['endpoints'][gpu], '--output-dir',
        str(args.results_root/request['run_id']/('group_'+str(gpu))),
        '--parent-job-id', args.parent_job_id, '--node', args.node,
        '--gpu-index', str(gpu), '--run-namespace', request['run_id']+'_g'+str(gpu),
        '--prompt-format', request.get('prompt_format','plain'),
        '--task-ids', ','.join(map(str, tasks))]
    for key, value in capacity.items():
        command += ['--'+key.replace('_', '-'), str(value)]
    for key in ('csv', 'manifest', 'environment_config', 'split', 'website'):
        if key in request:
            command += ['--'+key.replace('_', '-'), str(request[key])]
    return command+settings_flags(settings)+settings_flags(sampling)


def main():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--defaults', type=Path)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--parent-job-id', required=True)
    parser.add_argument('--node', required=True)
    args = parser.parse_args()
    if (os.environ.get('SLURM_JOB_ID') != args.parent_job_id or socket.gethostname() != args.node
            or not os.environ.get('SLURM_STEP_ID', '').isdecimal()):
        raise RuntimeError('dispatcher must execute inside the declared compute step')
    request = json.loads(args.request.read_text(encoding='utf-8'))
    defaults = json.loads(args.defaults.read_text(encoding='utf-8')) if args.defaults else {}
    settings, sampling, capacity = resolve_request(request, defaults)
    package = Path(request['package']).resolve(strict=True)
    if not (package/'run_star.py').is_file():
        raise ValueError('request package lacks unified entry point')
    result = args.results_root/request['run_id']
    result.mkdir(parents=True, exist_ok=False)
    (result/'resolved_settings.json').write_text(json.dumps(dict(request=request,
        experiment=settings.as_dict(), sampling=sampling.as_dict(), **capacity), indent=2), encoding='utf-8')
    workers, logs = [], []
    try:
        for gpu in (0, 1):
            tasks = request['task_ids'][gpu::2]
            if not tasks:
                continue
            log = (result/('runner_%d.log'%gpu)).open('x', encoding='utf-8')
            logs.append(log)
            workers.append(subprocess.Popen(worker_command(args, request, gpu, tasks, settings, sampling, capacity),
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                         PYTHONPATH=str(package/'source')+os.pathsep+os.environ.get('STAR_RUNTIME_PYTHONPATH', '')),
                stdout=log, stderr=subprocess.STDOUT))
        codes = [worker.wait() for worker in workers]
        (result/'dispatch_done.json').write_text(json.dumps(dict(exit_codes=codes,
            run_id=request['run_id'], dataset=request['dataset'], resident_reused=True), indent=2), encoding='utf-8')
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                worker.wait()
        for log in logs:
            log.close()
    return int(any(codes))


if __name__ == '__main__': raise SystemExit(main())
