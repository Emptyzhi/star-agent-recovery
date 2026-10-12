"""Run the WebShop v16 controller through an existing vLLM resident."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import socket
from urllib.parse import urlparse

from web_v16_run import run_case, save, verify_backend, verify_runtime
from model_backend import add_model_arguments, load_model_bundle, check_model_health
from experiment_settings import (Experiment, Sampling, add_experiment_arguments,
                                 add_sampling_arguments, settings_from_args, validate_runtime_settings)


def task_ids(value):
    rows = [int(item) for item in value.split(',') if item]
    if not rows or len(rows) != len(set(rows)) or any(item < 0 or item >= 500 for item in rows):
        raise RuntimeError('invalid exact task manifest')
    return rows


def select_shard(rows, shard_index, shard_count):
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise RuntimeError('invalid task shard coordinates')
    selected = rows[shard_index::shard_count]
    if not selected:
        raise RuntimeError('task shard is empty')
    return selected


def endpoint_port(value):
    parsed = urlparse(value)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.port is None:
        raise RuntimeError('resident endpoint must be an explicit loopback HTTP URL')
    return parsed.port


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--website', required=True)
    parser.add_argument('--endpoints', nargs=2, required=True)
    add_model_arguments(parser)
    add_experiment_arguments(parser)
    add_sampling_arguments(parser)
    parser.add_argument('--vendor-dir', type=Path)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--parent-job-id', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--run-namespace', required=True)
    parser.add_argument('--task-ids', required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--workers-per-gpu', type=int, default=40)
    parser.add_argument('--context-limit', type=int, default=32768)
    parser.add_argument('--prompt-limit', type=int, default=32000)
    parser.add_argument('--step-id', default=None)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--exact-tasks', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    settings, sampling = settings_from_args(args,Experiment), settings_from_args(args,Sampling)
    if socket.gethostname() != args.node or os.environ.get('SLURM_JOB_ID') != args.parent_job_id:
        raise RuntimeError('v16 runner is outside the declared Slurm allocation')
    args.step_id = args.step_id or os.environ.get('SLURM_STEP_ID')
    if not args.step_id or not str(args.step_id).isdecimal() or args.gpu_index not in (0, 1):
        raise RuntimeError('v16 runner lacks the declared resident step or GPU')
    if args.workers_per_gpu < 1 or not 0 < args.prompt_limit < args.context_limit:
        raise ValueError('invalid concurrency or context limits')
    manifest = task_ids(args.task_ids)
    tasks = select_shard(manifest, args.shard_index, args.shard_count)
    output = Path(args.output_dir)
    package_root = Path(__file__).resolve().parent.parent
    if args.vendor_dir is not None and args.vendor_dir.resolve() != package_root / 'vendor':
        raise ValueError('vendor prompts must come from this versioned package')
    config, tokenizer, eos_ids = load_model_bundle(args.model_path, args.dtype)
    binding = json.loads(json.dumps({
        'parent_job_id': args.parent_job_id,
        'step_id': str(args.step_id), 'hostname': args.node, 'gpu': args.gpu_index,
        'gpu_uuid': None, 'pid': None,
        'port': endpoint_port(args.endpoints[args.gpu_index]), 'instance': None}))
    from requests import get
    response = get(args.endpoints[args.gpu_index] + '/health', timeout=30)
    response.raise_for_status()
    health = response.json()
    check_model_health(health, config)
    validate_runtime_settings(health,settings,sampling,args.prompt_limit)
    if (args.workers_per_gpu > health['max_num_seqs']
            or args.context_limit != health['model_context_tokens']
            or args.prompt_limit > health['max_prompt_tokens']
            or tuple(health['decoding']['eos_token_id']) != eos_ids):
        raise ValueError('runner settings exceed or differ from the resident contract')
    binding.update(gpu_uuid=health['gpu_uuid'], pid=health['service_pid'], instance=health['service_instance_id'])
    verify_runtime(binding, os.environ, socket.gethostname())
    endpoint_port(args.website)
    response = get(args.website.rstrip('/') + '/__star_interface', timeout=30)
    response.raise_for_status()
    backend = response.json()
    verify_backend(backend, binding, settings.environment_seed)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'run_config.json', dict(model=config, eos_token_ids=eos_ids,
         binding=binding, backend=backend, tasks=tasks, workers_per_gpu=args.workers_per_gpu,
         prompt_limit=args.prompt_limit, context_limit=args.context_limit,
         source_baseline='r10_p12_repeat2_full500_20261002',
         sampling=sampling.as_dict(),experiment_settings=settings.as_dict(),
         green_generation_policy='free raw action; first_line during generation',
         green_command_trie=False,yellow_red_command_constraints='unchanged',
         resident_default_sampling=health['decoding']))
    print('RUNNER_START', json.dumps({'run_id': args.run_namespace, 'gpu_index': args.gpu_index,
          'shard_index': args.shard_index, 'shard_count': args.shard_count,
          'task_ids': tasks, 'output_dir': str(output),
          'website': args.website,
          'model_endpoint': args.endpoints[args.gpu_index]}), flush=True)
    rows, failures = [], []
    def one(task):
        return run_case(task, binding, backend, tokenizer, config, output,
                        args.run_namespace, args.website, prompt_limit=args.prompt_limit,
                        settings=settings,sampling=sampling)
    with ThreadPoolExecutor(max_workers=min(args.workers_per_gpu, len(tasks))) as pool:
        futures = {pool.submit(one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                failure = {'task': task, 'error_type': type(error).__name__, 'error': str(error)}
                failures.append(failure)
                print('TASK_FATAL', json.dumps(failure, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: row['identity']['task_id'])
    print('RUNNER_SUMMARY', json.dumps({
        'run_id': args.run_namespace, 'gpu_index': args.gpu_index,
        'completed': len(rows), 'failures': failures,
        'successes': sum(row['success'] for row in rows),
        'scoring_policy': 'webshop_best_observed_purchase_v1',
        'benchmark_rewards': [row['benchmark_reward'] for row in rows],
        'null_final_public_reward': sum(row['final_public_reward'] is None for row in rows),
    }, ensure_ascii=False), flush=True)
    save(output / 'run_done.json', {
        'complete': not failures, 'run_id': args.run_namespace,
        'scoring_policy': 'webshop_best_observed_purchase_v1',
        'R_percent': 100*sum(row['benchmark_reward'] for row in rows)/len(tasks) if not failures else None,
        'SR_percent': 100*sum(row['success'] for row in rows)/len(tasks) if not failures else None,
        'task_manifest': manifest, 'tasks': tasks,
        'completed': len(rows), 'failures': failures,
        'successes': sum(row['success'] for row in rows),
        'null_final_public_reward': sum(row['final_public_reward'] is None for row in rows),
        'source_run_id': 'r10_p12_repeat2_full500_20261002',
        'controller_version': rows[0]['identity']['controller_version'] if rows else None,
        'environment_initialization_seed': settings.environment_seed,
        'model_rng': 'resident_continuous_vllm_rng',
    })
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
