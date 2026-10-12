"""Shared resident binding, concurrency and per-task accounting for native adapters."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import time
import traceback
from urllib.parse import urlparse

from bound_resident_client import BoundResidentClient
from experiment_settings import validate_runtime_settings
from model_backend import load_model_bundle, check_model_health, encode_prompt
from web_v16_run import save, verify_runtime

PACKAGE = Path(__file__).resolve().parent.parent


def bind_runtime(args, settings, sampling):
    import requests
    if os.environ.get('SLURM_JOB_ID') != args.parent_job_id or socket.gethostname() != args.node:
        raise RuntimeError('experiments must run inside their declared compute allocation')
    step = args.step_id or os.environ.get('SLURM_STEP_ID')
    if step is None or not str(step).isdecimal():
        raise ValueError('a compute workload step is required')
    url = urlparse(args.endpoint)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or url.port is None:
        raise ValueError('resident must be a node-local explicit HTTP endpoint')
    model, tokenizer, eos = load_model_bundle(args.model_path, args.dtype)
    response = requests.get(args.endpoint.rstrip('/')+'/health', timeout=30)
    response.raise_for_status()
    health = response.json()
    check_model_health(health, model)
    validate_runtime_settings(health, settings, sampling, args.prompt_limit)
    if (health['max_num_seqs'] < args.workers_per_gpu or health['model_context_tokens'] != args.context_limit
            or tuple(health['decoding']['eos_token_id']) != eos):
        raise ValueError('requested capacity/tokenization differs from resident')
    binding = dict(parent_job_id=args.parent_job_id, step_id=str(step), hostname=args.node,
                   gpu=args.gpu_index, gpu_uuid=health['gpu_uuid'], pid=health['service_pid'],
                   instance=health['service_instance_id'], port=url.port)
    verify_runtime(binding, os.environ, socket.gethostname())
    return binding, model, tokenizer, health


def load_inputs(args, tasks, settings):
    if args.dataset == 'game24':
        import csv
        with args.csv.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or 'Puzzles' not in reader.fieldnames:
                raise ValueError('Game24 CSV requires Puzzles column')
            rows = [row['Puzzles'] for row in reader]
        if any(t >= len(rows) for t in tasks):
            raise ValueError('task ID is outside original CSV rows')
        from test_game24.game24_env import Game24Environment
        for t in tasks:
            Game24Environment(rows[t])
        return {t: rows[t] for t in tasks}, None
    import yaml
    if importlib.metadata.version('alfworld') != '0.3.5':
        raise ValueError('ALFWorld runner requires the archived 0.3.5 environment')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    games = manifest[args.split]
    if not isinstance(games, list) or len(set(games)) != len(games) or any(t >= len(games) for t in tasks):
        raise ValueError('invalid ALFWorld exact task manifest')
    path = args.environment_config or PACKAGE/'vendor/base_config.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    config['general']['random_seed'] = settings.environment_seed
    return {t: games[t] for t in tasks}, config


def alf_episode(gamefile, config, tokenizer, settings, emit):
    from alfworld.agents.environment import AlfredTWEnv
    from alf_runtime import ALF_ENV_LOCK, ThreadSafeAlfEpisode
    provider = object.__new__(AlfredTWEnv)
    provider.config = deepcopy(config)
    provider.train_eval = 'eval_out_of_distribution'
    provider.game_files = [gamefile]
    provider.num_games = 1
    prefixes = {'pick_and_place':'put', 'pick_clean_then_place':'clean',
        'pick_heat_then_place':'heat', 'pick_cool_then_place':'cool', 'look_at_obj':'examine', 'pick_two_obj':'puttwo'}
    name = Path(gamefile).parent.parent.name
    matches = [v for k, v in prefixes.items() if name.startswith(k)]
    if len(matches) != 1:
        raise ValueError('task family has no unique native demonstration set')
    demos = json.loads((PACKAGE/'prompts/alfworld/alfworld_base_prompts.json').read_text(encoding='utf-8'))
    family = matches[0]
    base = 'Interact with a household to solve a task. Here are two examples.\n'+demos['act_'+family+'_1']+demos['act_'+family+'_0']
    # TextWorld registration/parsing is process-global; model requests remain concurrent.
    with ALF_ENV_LOCK:
        environment = provider.init_env(batch_size=1)
    return ThreadSafeAlfEpisode(environment, gamefile, base,
        lambda messages: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        action_limit=settings.initial_actions, emit=emit)


def execute_alf(task, item, client, tokenizer, health, config, settings, args, emit, stack):
    from alf_configured import ConfiguredAlfController
    from icr_core import Ledger, ProposalOracle, BudgetExhausted, ProtocolError
    episode = alf_episode(item, config, tokenizer, settings, emit)
    stack.callback(episode.close)
    ledger = Ledger(settings.ledger_limits(args.prompt_limit))
    def commands(values, limit):
        rows = tuple(tuple(tokenizer.encode(v, add_special_tokens=False)) for v in values)
        for value, tokens in zip(values, rows):
            if not tokens or len(tokens)+1 > limit or tokenizer.decode(tokens, clean_up_tokenization_spaces=False) != value:
                raise ProtocolError('native command tokenization does not roundtrip')
        return rows
    oracle = ProposalOracle(client.generate, master_seed=1, emit=emit,
        token_counter=lambda p: len(encode_prompt(tokenizer, p)['input_ids']),
        context_limit=args.context_limit, prompt_limit=args.prompt_limit,
        request_namespace=args.run_namespace+'/task'+str(task), generation_versions=health,
        command_language_validator=commands, ordinary_output_limit=settings.actor_output_tokens)
    controller = ConfiguredAlfController(episode, oracle, ledger, task=task, emit=emit, settings=settings)
    stack.callback(controller.close)
    controller.start()
    try:
        state = controller.run()
    except BudgetExhausted as error:
        controller.interrupt('RESOURCE_LIMIT', str(error))
        state = controller.to_dict()
    score = episode.snapshot()['public_terminal_score']
    return dict(complete=True, status=state['status'], success=score == 1,
        final_public_reward=score, cost=ledger.to_dict(), state=state,
        replay_actions=episode.replay_actions, replay_resets=episode.replay_resets,
        terminal_rewards=[o.receipt['public_terminal_score'] for o in controller.occurrences.values() if o.receipt['done']])


def run_task(task, item, args, settings, sampling, binding, model, tokenizer, config):
    directory = args.output_dir/('task_%03d'%task)
    directory.mkdir(exist_ok=False)
    identity = dict(binding, dataset=args.dataset, task_id=task, run_id=args.run_namespace,
        package=str(PACKAGE), source_version='STAR_game24_native_v31_20261003' if args.dataset=='game24' else 'STAR_unified_20261002',
        client_prompt_format=args.prompt_format,
        model_config=model, sampling=sampling.as_dict(), experiment=settings.as_dict(),
        model_request_seed=None, input_identity=item)
    save(directory/'identity.json', identity)
    started = time.monotonic()
    usage = []
    with ExitStack() as stack:
        stream = stack.enter_context((directory/'events.jsonl').open('x', encoding='utf-8'))
        def emit(event, **data):
            stream.write(json.dumps(dict(event=event, time=time.time(), **data), ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
        client = BoundResidentClient(binding, args.run_namespace+'_task'+str(task), model,
            sampling=sampling, request_timeout=settings.request_timeout)
        stack.callback(client.close)
        try:
            health = client.begin_run()
            save(directory/'health_before.json', health)
            original_generate = client.generate
            def measured(request, **kwargs):
                result = original_generate(request, **kwargs)
                usage.append(dict(request_id=request.request_id, input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens, queue_seconds=result.raw_response['queue_time_seconds'],
                    model_active_seconds=result.raw_response['model_active_step_seconds']))
                return result
            client.generate = measured
            if args.dataset == 'game24':
                from test_game24.model_api import ResidentModelClient
                from test_game24.star_controller import Game24STARController
                adapter = ResidentModelClient(client, tokenizer, prompt_limit=args.prompt_limit, context_limit=args.context_limit,
                                              prompt_format=args.prompt_format)
                controller = Game24STARController(adapter, settings=settings, emit=emit)
                result = controller.run(task, item).to_dict()
                save(directory/'scoped_state.json', result)
                if result['status'] == 'PROTOCOL_ERROR':
                    raise ValueError(result['error'])
                summary = dict(result, complete=True, final_public_reward=result['reward'])
            else:
                summary = execute_alf(task, item, client, tokenizer, health, config, settings, args, emit, stack)
                save(directory/'scoped_state.json', summary.pop('state'))
            # A failed environment/lease cleanup must not leave a nominal success summary.
            stack.close()
            elapsed = time.monotonic()-started
            queue = sum(row['queue_seconds'] for row in usage)
            if queue > elapsed:
                raise ValueError('measured queue time exceeds task wall time')
            summary.update(identity=identity, elapsed_seconds=elapsed, model_usage=usage,
                queue_excluded_elapsed_seconds=elapsed-queue, model_queue_seconds=queue)
            save(directory/'summary.json', summary)
            print('TASK_COMPLETE', json.dumps(dict(task=task, success=summary['success'],
                                                  score=summary['final_public_reward'])), flush=True)
            return summary
        except Exception as error:
            # Retain technical failures and return a nonzero batch outcome, never a fabricated reward.
            save(directory/'fatal.json', dict(identity=identity, error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), completed_model_usage=usage,
                token_accounting_complete=False, timing_accounting_complete=False,
                elapsed_seconds=time.monotonic()-started))
            raise


def run_batch(args, settings, sampling, tasks):
    if re.fullmatch(r'[A-Za-z0-9_-]+', args.run_namespace) is None:
        raise ValueError('run namespace must be a safe unique identifier')
    binding, model, tokenizer, health = bind_runtime(args, settings, sampling)
    inputs, config = load_inputs(args, tasks, settings)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save(args.output_dir/'run_config.json', dict(dataset=args.dataset, binding=binding, model=model,
        tasks=tasks, workers_per_gpu=args.workers_per_gpu, experiment=settings.as_dict(),
        sampling=sampling.as_dict(), environment_config=config,
        model_request_seed=None, model_rng='continuous_resident_rng',
        seed_scope='task_order_fixed_no_arithmetic_rng' if args.dataset=='game24' else 'general.random_seed',
        client_prompt_format=args.prompt_format,
        source_version='STAR_game24_native_v31_20261003' if args.dataset=='game24' else 'STAR_unified_20261002'))
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=min(args.workers_per_gpu, len(tasks))) as pool:
        futures = {pool.submit(run_task, t, inputs[t], args, settings, sampling,
                               binding, model, tokenizer, config): t for t in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(dict(task=futures[future], type=type(error).__name__, error=str(error)))
    save(args.output_dir/'run_done.json', dict(complete=not failures, tasks=tasks, completed=len(rows),
        successes=sum(r['success'] for r in rows), failures=failures,
        null=sum(r['final_public_reward'] is None for r in rows)))
    return int(bool(failures))
