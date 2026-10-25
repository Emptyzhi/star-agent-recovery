"""Single CLI, explicit dataset selection, no model loading in experiment runners."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE/'source'))
sys.path.insert(0, str(PACKAGE))
from experiment_settings import (Experiment, Sampling, add_experiment_arguments,
                                 add_sampling_arguments, settings_from_args, settings_flags)
from model_backend import add_model_arguments


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('alfworld', 'game24', 'webshop'))
    add_model_arguments(parser)
    parser.add_argument('--prompt-format',choices=('plain','native_chat'),default='plain',
                        help='Game24 model adapter framing; plain preserves the author completion interface')
    add_experiment_arguments(parser)
    parser.set_defaults(**{f.name: None for f in fields(Experiment)})
    add_sampling_arguments(parser)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-namespace', required=True)
    parser.add_argument('--task-ids', required=True, help='exact original task IDs, comma-separated')
    parser.add_argument('--workers-per-gpu', type=int, default=40)
    parser.add_argument('--parent-job-id', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--step-id')
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--context-limit', type=int, default=32768)
    parser.add_argument('--prompt-limit', type=int, default=32000)
    parser.add_argument('--website')
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--environment-config', type=Path)
    parser.add_argument('--split', choices=('test', 'development'), default='test')
    parser.add_argument('--dry-run', action='store_true', help='show resolved settings; no model/environment calls')
    return parser


def resolve(args):
    if args.dataset != 'game24' and args.prompt_format != 'plain':
        raise ValueError('--prompt-format native_chat currently applies to the Game24 model adapter')
    base = Experiment.for_dataset(args.dataset).as_dict()
    base.update({f.name: getattr(args, f.name) for f in fields(Experiment) if getattr(args, f.name) is not None})
    settings = Experiment(**base)
    sampling = settings_from_args(args, Sampling)
    tasks = [int(t) for t in args.task_ids.split(',')]
    if not tasks or len(set(tasks)) != len(tasks) or min(tasks) < 0:
        raise ValueError('task IDs must be explicit unique nonnegative indices')
    if args.workers_per_gpu < 1 or args.gpu_index < 0 or not 0 < args.prompt_limit < args.context_limit:
        raise ValueError('invalid runtime capacity')
    required = {'webshop': 'website', 'game24': 'csv', 'alfworld': 'manifest'}[args.dataset]
    if getattr(args, required) is None:
        raise ValueError('--'+required+' is required for '+args.dataset)
    allowed = {'webshop': {'website'}, 'game24': {'csv'}, 'alfworld': {'manifest','environment_config'}}[args.dataset]
    for name in ('website','csv','manifest','environment_config'):
        if name not in allowed and getattr(args,name) is not None:
            raise ValueError('--'+name.replace('_','-')+' does not apply to '+args.dataset)
    if args.dataset != 'alfworld' and args.split != 'test':
        raise ValueError('--split applies only to ALFWorld')
    if args.dataset == 'webshop' and any(t >= 500 for t in tasks):
        raise ValueError('WebShop task ID must be in 0..499')
    return settings, sampling, tasks


def webshop_arguments(args, settings, sampling):
    flags = ['--website', args.website, '--endpoints', args.endpoint, args.endpoint,
        '--model-path', args.model_path, '--dtype', args.dtype,
        '--output-dir', str(args.output_dir), '--parent-job-id', args.parent_job_id,
        '--node', args.node, '--run-namespace', args.run_namespace,
        '--task-ids', args.task_ids, '--gpu-index', str(args.gpu_index),
        '--workers-per-gpu', str(args.workers_per_gpu), '--context-limit', str(args.context_limit),
        '--prompt-limit', str(args.prompt_limit)]
    if args.step_id is not None:
        flags += ['--step-id', args.step_id]
    return flags+settings_flags(settings)+settings_flags(sampling)


def main(argv=None):
    args = build_parser().parse_args(argv)
    settings, sampling, tasks = resolve(args)
    if args.dry_run:
        print(json.dumps(dict(dataset=args.dataset, model_path=args.model_path, dtype=args.dtype,
            prompt_format=args.prompt_format,
            tasks=tasks, workers_per_gpu=args.workers_per_gpu,
            experiment=settings.as_dict(), sampling=sampling.as_dict(),
            model_request_seed=None, model_rng='continuous_resident_rng',
            environment_seed_scope='deterministic_arithmetic_no_task_reseeding' if args.dataset=='game24'
                                   else 'environment_initialization'), indent=2))
        return 0
    if args.dataset == 'webshop':
        from scoped_recovery_batch import main as run_webshop
        previous = sys.argv
        try:
            sys.argv = [str(PACKAGE/'run_webshop.py'), *webshop_arguments(args, settings, sampling)]
            run_webshop()
        finally:
            sys.argv = previous
        return 0
    from dataset_runtime import run_batch
    return run_batch(args, settings, sampling, tasks)


if __name__ == '__main__':
    raise SystemExit(main())
