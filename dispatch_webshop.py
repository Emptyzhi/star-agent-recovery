"""Dispatch a controller package without loading or replacing resident weights."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import fields

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE/'source'))
from model_backend import add_model_arguments
from experiment_settings import (Experiment,Sampling,add_experiment_arguments,
                                 add_sampling_arguments,settings_flags)


def worker_command(package, args, run_id, tasks, gpu, result, *, experiment=None,sampling=None,
                   workers_per_gpu=40,context_limit=32768,prompt_limit=32000):
    experiment=Experiment() if experiment is None else experiment
    sampling=Sampling() if sampling is None else sampling
    return [sys.executable, str(package/'run_webshop.py'),
            '--local_llm_path', args.model_path, '--dtype', args.dtype,
            '--website', 'http://127.0.0.1:3001',
            '--endpoints', 'http://127.0.0.1:41581', 'http://127.0.0.1:41582',
            '--output-dir', str(result/('group_%d'%gpu)),
            '--parent-job-id', args.parent_job_id, '--node', args.node,
            '--run-namespace', run_id+'_g%d'%gpu, '--task-ids', ','.join(map(str,tasks)),
            '--gpu-index', str(gpu), '--workers-per-gpu', str(workers_per_gpu),
            '--context-limit', str(context_limit), '--prompt-limit', str(prompt_limit)
            ]+settings_flags(experiment)+settings_flags(sampling)


def resolve_run_settings(args,request):
    defaults=json.loads(args.defaults.read_text(encoding='utf-8')) if args.defaults else {}
    # Precedence is explicit: library defaults < resident defaults < CLI < queue.
    resolved=[]
    for cls,key in ((Experiment,'experiment'),(Sampling,'sampling')):
        values=cls().as_dict()
        values.update(defaults.get(key,{}))
        values.update({f.name:getattr(args,f.name) for f in fields(cls) if getattr(args,f.name) is not None})
        values.update(request.get(key,{}))
        if cls is Experiment and 'environment_seed' in request:
            if 'environment_seed' in request.get(key,{}) and request[key]['environment_seed'] != request['environment_seed']:
                raise ValueError('conflicting queue environment seeds')
            values['environment_seed']=request['environment_seed']
        resolved.append(cls(**values))
    capacity={}
    for key,default in (('workers_per_gpu',40),('context_limit',32768),('prompt_limit',32000)):
        value=defaults.get(key,default)
        if getattr(args,key) is not None:value=getattr(args,key)
        if key in request:value=request[key]
        if type(value) is not int or value<1:raise ValueError('invalid '+key)
        capacity[key]=value
    return *resolved,capacity


def build_parser():
    parser=argparse.ArgumentParser()
    add_model_arguments(parser)
    add_experiment_arguments(parser)
    add_sampling_arguments(parser)
    for key in ('workers_per_gpu','context_limit','prompt_limit'):
        parser.add_argument('--'+key.replace('_','-'),type=int)
    parser.set_defaults(**{f.name:None for cls in (Experiment,Sampling) for f in fields(cls)})
    parser.add_argument('--defaults',type=Path)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--parent-job-id',required=True)
    parser.add_argument('--node',required=True)
    return parser


def delegate_package_dispatch(package):
    target = package/'dispatch_webshop.py'
    if target.resolve() != Path(__file__).resolve():
        if not target.is_file():
            raise ValueError('controller package lacks its versioned dispatcher')
        # Each package owns its settings schema; loaded model services stay put.
        os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])


def main():
    args = build_parser().parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    required={'run_id','package','task_ids'}
    optional={'environment_seed','workers_per_gpu','experiment','sampling','prompt_limit','context_limit',
              'dataset','csv','manifest','environment_config','split','website','endpoints'}
    if not required<=set(request) or set(request)-required-optional:
        raise ValueError('queue request has unknown or missing fields')
    if 'dataset' in request:
        target = Path(request['package']).resolve(strict=True)/'dispatch_star.py'
        if not target.is_file():
            raise ValueError('dataset request lacks its unified dispatcher')
        os.execv(sys.executable, [sys.executable, str(target), *sys.argv[1:]])
    run_id, tasks = request['run_id'], request['task_ids']
    if not re.fullmatch('[a-z0-9_]{1,100}', run_id):
        raise ValueError('invalid run_id')
    if (type(tasks) is not list or len(tasks) < 2
            or any(type(t) is not int or not 0 <= t < 500 for t in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError('dual-GPU request requires unique WebShop task IDs')
    package = Path(request['package']).resolve(strict=True)
    if not (package/'run_webshop.py').is_file():
        raise ValueError('controller package lacks the unified WebShop entry point')
    delegate_package_dispatch(package)
    experiment,sampling,capacity=resolve_run_settings(args,request)
    result = args.results_root/run_id
    result.mkdir(parents=True, exist_ok=False)
    (result/'resolved_settings.json').write_text(json.dumps(dict(
        experiment=experiment.as_dict(),sampling=sampling.as_dict(),**capacity),indent=2),encoding='utf-8')
    processes, logs = [], []
    try:
        for gpu in (0,1):
            log = (result/('runner_%d.log'%gpu)).open('x', encoding='utf-8')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            # Do not inherit another controller package's source path.
            runtime_libraries = os.environ.get('STAR_RUNTIME_PYTHONPATH','')
            env['PYTHONPATH'] = str(package/'source') + (os.pathsep+runtime_libraries if runtime_libraries else '')
            command = worker_command(package,args,run_id,tasks[gpu::2],gpu,result,
                                     experiment=experiment,sampling=sampling,**capacity)
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        (result/'dispatch_done.json').write_text(json.dumps(dict(
            run_id=run_id, exit_codes=codes, model_path=args.model_path,
            dtype=args.dtype, resident_reused=True), indent=2), encoding='utf-8')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    if any(codes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
