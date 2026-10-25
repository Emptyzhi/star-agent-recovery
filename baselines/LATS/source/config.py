"""Resolve CLI options once and serialize the exact search configuration."""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import math
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {'game24': 500, 'alfworld': 134, 'webshop': 500}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


@dataclass(frozen=True)
class SearchConfig:
    candidates: int = 5
    eval_samples: int = 1
    iterations: int = 2
    max_output_tokens: int = 100
    round_actions: int | None = None

    def __post_init__(self):
        for value in (self.candidates, self.eval_samples, self.iterations, self.max_output_tokens):
            if type(value) is not int or value < 1:
                raise ValueError('search limits must be positive integers')
        if self.round_actions is not None and (type(self.round_actions) is not int or self.round_actions < 1):
            raise ValueError('round_actions must be a positive integer')


def parser():
    p = argparse.ArgumentParser(description='LATS search using an existing resident model; no model is loaded by this runner.')
    p.add_argument('--dataset', '--task', choices=COUNTS, required=True)
    p.add_argument('--local_llm_path', '--model-path', required=True, help='Exact model path reported by the resident /health endpoint')
    p.add_argument('--endpoints', nargs='+', required=True, help='One resident generation URL per GPU, in shard order')
    p.add_argument('--run-id', required=True)
    p.add_argument('--output-dir', type=Path, help='Fresh result directory; defaults to results/RUN_ID')
    p.add_argument('--package', default=str(ROOT), help='Deployed code path for queue requests')
    p.add_argument('--write-request', type=Path, help='Write a queue request only; do not contact services or submit it')
    p.add_argument('--dry-run', action='store_true', help='Print resolved configuration without loading datasets or contacting services')
    # Dataset paths may belong to the remote Linux host even when the CLI is
    # assembled on Windows; keep them as POSIX strings through serialization.
    p.add_argument('--csv')
    p.add_argument('--manifest', help='ALFWorld JSON containing the ordered test task paths')
    p.add_argument('--alfworld-config', help='Default: PACKAGE/vendor/ga/test_alfworld/base_config.yaml')
    p.add_argument('--alfworld-data')
    p.add_argument('--alfworld-runtime', help='ALFWorld 0.3.5 site-packages path')
    p.add_argument('--website', help='WebShop HTTP environment URL')
    p.add_argument('--task-start', type=int, default=0)
    p.add_argument('--task-count', type=positive)
    p.add_argument('--task-ids', nargs='+', type=int)
    defaults = SearchConfig()
    p.add_argument('--candidates', '--n_generate_sample', type=positive, default=defaults.candidates,
                   help='Expansion samples; the native root expansion doubles this count')
    p.add_argument('--eval-samples', '--n_evaluate_sample', type=positive, default=defaults.eval_samples)
    p.add_argument('--iterations', type=positive, default=defaults.iterations)
    p.add_argument('--round-actions', '--max-depth', '--max_depth', dest='round_actions', type=positive,
                   help='Fresh path actions per iteration, including expansion; dataset default 12/50/15')
    p.add_argument('--max-output-tokens', type=positive, default=defaults.max_output_tokens)
    p.add_argument('--workers-per-gpu', type=positive, default=40)
    p.add_argument('--request-limit-per-endpoint', type=positive, default=40)
    p.add_argument('--max-num-seqs', type=positive, default=40, help='Required resident engine capacity, not a service reconfiguration command')
    p.add_argument('--environment-seed', type=int, default=1)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--repetition-penalty', type=float, default=1.05)
    p.add_argument('--trust-remote-code', action='store_true', help='Explicitly allow custom code when loading the selected tokenizer')
    return p


def resolve(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', args.run_id):
        p.error('run-id must be a simple identifier of at most 100 characters')
    if args.task_start < 0 or (args.task_ids is not None and (args.task_count is not None or args.task_start != 0)):
        p.error('use task-ids or task-start/task-count, not both')
    ids = args.task_ids if args.task_ids is not None else list(range(args.task_start, args.task_start+(args.task_count or COUNTS[args.dataset])))
    if any(i < 0 for i in ids) or len(ids) != len(set(ids)):
        p.error('task IDs must be distinct and nonnegative')
    if len(set(args.endpoints)) != len(args.endpoints) or not all(urlsplit(x).scheme in ('http','https') and urlsplit(x).netloc for x in args.endpoints):
        p.error('endpoints must be distinct HTTP URLs')
    if not (args.temperature > 0 and 0 < args.top_p <= 1 and (args.top_k == -1 or args.top_k >= 1) and args.repetition_penalty > 0):
        p.error('invalid sampling settings')
    required = {'game24': args.csv, 'alfworld': args.manifest, 'webshop': args.website}[args.dataset]
    if required is None:
        p.error('dataset requires '+{'game24': '--csv', 'alfworld': '--manifest', 'webshop': '--website'}[args.dataset])
    from .round_budget import ROUND_ACTIONS
    search = SearchConfig(args.candidates, args.eval_samples, args.iterations, args.max_output_tokens,
                          args.round_actions or ROUND_ACTIONS[args.dataset])
    allowed = {'game24': {'csv'}, 'alfworld': {'manifest', 'alfworld_data', 'alfworld_runtime'}, 'webshop': {'website'}}[args.dataset]
    for name in ('csv', 'manifest', 'alfworld_data', 'alfworld_runtime', 'website'):
        if getattr(args, name) is not None and name not in allowed:
            p.error('--'+name.replace('_', '-')+' does not apply to '+args.dataset)
    if args.dataset != 'alfworld' and args.alfworld_config is not None:
        p.error('--alfworld-config only applies to ALFWorld')
    req = dict(run_id=args.run_id, package=args.package, dataset=args.dataset, task_ids=ids,
               request_limit_per_endpoint=args.request_limit_per_endpoint,
               model_path=args.local_llm_path, endpoints=args.endpoints, workers_per_gpu=args.workers_per_gpu,
               max_num_seqs=args.max_num_seqs, environment_seed=args.environment_seed,
               trust_remote_code=args.trust_remote_code,
               sampling=dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                             top_k=args.top_k, repetition_penalty=args.repetition_penalty),
               experiment=dict(method='lats', search=asdict(search)))
    if args.dataset == 'game24':
        req['csv'] = str(args.csv)
    elif args.dataset == 'alfworld':
        req['manifest'] = str(args.manifest)
        req['alfworld_config'] = args.alfworld_config or args.package.rstrip('/\\')+'/vendor/ga/test_alfworld/base_config.yaml'
        if args.alfworld_data is not None:
            req['alfworld_data'] = str(args.alfworld_data)
        if args.alfworld_runtime is not None:
            req['alfworld_runtime'] = args.alfworld_runtime
    else:
        req['website'] = args.website
    args.output_dir = args.output_dir or Path('results')/args.run_id
    try:
        validate_request(req)
    except ValueError as error:
        p.error(str(error))
    return args, req


def validate_request(req):
    required = {'run_id', 'package', 'dataset', 'task_ids', 'model_path', 'endpoints',
                'workers_per_gpu', 'max_num_seqs', 'environment_seed', 'trust_remote_code',
                'sampling', 'experiment'}
    datasets = {'game24': {'csv'}, 'alfworld': {'manifest', 'alfworld_config'}, 'webshop': {'website'}}
    if req.get('dataset') not in datasets:
        raise ValueError('unsupported dataset')
    required |= datasets[req['dataset']]
    optional = {'alfworld_data', 'alfworld_runtime'} if req['dataset'] == 'alfworld' else set()
    optional = optional | {'request_limit_per_endpoint'}
    if required - req.keys() or req.keys() - required - optional:
        raise ValueError('request has missing or unknown fields')
    if not isinstance(req['run_id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', req['run_id']):
        raise ValueError('invalid run ID')
    for key in ('package', 'model_path', *datasets[req['dataset']], *(optional-{'request_limit_per_endpoint'}).intersection(req)):
        if not isinstance(req[key], str) or not req[key]:
            raise ValueError('missing path/URL: '+key)
    for key in ('workers_per_gpu', 'max_num_seqs'):
        if type(req[key]) is not int or req[key] < 1:
            raise ValueError('invalid concurrency: '+key)
    if 'request_limit_per_endpoint' in req and (type(req['request_limit_per_endpoint']) is not int or req['request_limit_per_endpoint'] < 1):
        raise ValueError('invalid endpoint request limit')
    if type(req['environment_seed']) is not int or type(req['trust_remote_code']) is not bool:
        raise ValueError('invalid seed or tokenizer trust setting')
    ids = req['task_ids']
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('invalid task IDs')
    endpoints = req['endpoints']
    if not isinstance(endpoints, list) or not endpoints or any(not isinstance(x, str) for x in endpoints):
        raise ValueError('invalid endpoints')
    if len(endpoints) != len(set(endpoints)) or any(urlsplit(x).scheme not in ('http','https') or not urlsplit(x).netloc for x in endpoints):
        raise ValueError('endpoints must be distinct HTTP URLs')
    if req['dataset'] == 'webshop' and (urlsplit(req['website']).scheme not in ('http', 'https') or not urlsplit(req['website']).netloc):
        raise ValueError('website must be an HTTP URL')
    sampling = req['sampling']
    if set(sampling) != {'do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty'} or sampling['do_sample'] is not True:
        raise ValueError('unsupported sampling schema')
    if any(type(sampling[k]) not in (float, int) or not math.isfinite(sampling[k]) for k in ('temperature', 'top_p', 'repetition_penalty')):
        raise ValueError('sampling must contain finite numbers')
    if not (sampling['temperature'] > 0 and 0 < sampling['top_p'] <= 1 and sampling['repetition_penalty'] > 0):
        raise ValueError('invalid sampling values')
    if type(sampling['top_k']) is not int or (sampling['top_k'] != -1 and sampling['top_k'] < 1):
        raise ValueError('top-k must be -1 or a positive integer')
    if set(req['experiment']) != {'method', 'search'} or req['experiment']['method'] != 'lats':
        raise ValueError('request must select LATS')
    if set(req['experiment']['search']) != set(asdict(SearchConfig())):
        raise ValueError('request must contain the exact resolved LATS search fields')
    search = SearchConfig(**req['experiment']['search'])
    if max(ids) >= COUNTS[req['dataset']]:
        raise ValueError(f'{req["dataset"]} task ID exceeds its dataset')
    return search
