"""Run WebShop v16 with an existing Slurm resident; never load model weights."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import socket
import time
import traceback

import requests

from bound_resident_client import BoundResidentClient
from icr_core import BudgetExhausted, Ledger, Limits, ProposalOracle, ProtocolError
from model_backend import encode_prompt, add_model_arguments, load_model_bundle
from web_v16_controller import VERSION, WebV16Controller
from web_v16_episode import PACKAGE, WebV16Episode
from experiment_settings import (Experiment, Sampling, add_experiment_arguments,
                                 add_sampling_arguments, settings_from_args, validate_runtime_settings)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def task_limits(prompt_limit=24000, settings=None):
    return (Experiment() if settings is None else settings).ledger_limits(prompt_limit)


def benchmark_reward(final_public_score):
    # An episode without a purchase earns zero; its native score stays null.
    return 0 if final_public_score is None else final_public_score


def purchase_score_report(final_public_score, terminal_rewards):
    import math
    if not all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
               for value in terminal_rewards):
        raise ProtocolError('purchase rewards must be actual finite scores in [0,1]')
    best = max([0, *terminal_rewards])
    return dict(success=best == 1, benchmark_reward=best, final_public_reward=final_public_score,
                terminal_rewards=list(terminal_rewards), scoring_policy='webshop_best_observed_purchase_v1',
                benchmark_reward_semantics='maximum actual executed purchase reward; zero without a purchase',
                native_final_benchmark_reward=benchmark_reward(final_public_score),
                native_final_success=final_public_score == 1)


def run_controller(controller):
    try:
        return controller.run()
    except BudgetExhausted as error:
        controller.interrupt('RESOURCE_LIMIT', str(error))
        return controller.to_dict()


def verify_runtime(binding, environ, hostname):
    if (environ['SLURM_JOB_ID'] != binding['parent_job_id']
            or environ['SLURM_STEP_ID'] != str(binding['step_id'])
            or hostname != binding['hostname']
            or environ['CUDA_VISIBLE_DEVICES'] != str(binding['gpu'])):
        raise ProtocolError('worker is not inside its bound resident allocation/step/GPU')


def verify_backend(record, binding, environment_seed=1):
    if record['initialization_seed'] != environment_seed:
        raise ProtocolError('WebShop environment seed mismatch: restart the website with the requested seed; model reload is unnecessary')
    if (record['parent_job_id'] != binding['parent_job_id']
            or record['step_id'] != str(binding['step_id'])
            or record['hostname'] != binding['hostname']
            or record['interface'] != 'table1_webshop_attributes_v1'
            or record['show_attributes'] is not True):
        raise ProtocolError('WebShop backend differs from the declared interface/allocation/seed')


def run_case(task, binding, backend, tokenizer, model, output, run_id, website, *, prompt_limit=32000,
             settings=None, sampling=None):
    settings = Experiment() if settings is None else settings
    sampling = Sampling() if sampling is None else sampling
    directory = output / ('task_%03d' % task)
    directory.mkdir(parents=True, exist_ok=False)
    identity = dict(binding, run_id=run_id, task_id=task, controller_version=VERSION,
        model_config=model, backend=backend, environment_initialization_seed=settings.environment_seed,
        per_task_model_seed=None, sampling='inherited_continuous_resident_rng',
        actor_output_tokens=settings.actor_output_tokens, yellow_diagnosis_tokens=settings.yellow_diagnosis_tokens,
        red_reflection_tokens=settings.red_reflection_tokens,
        initial_action_budget=settings.initial_actions, recovery_action_budget=settings.recovery_actions,
        experiment_settings=settings.as_dict(), sampling_settings=sampling.as_dict(),
        empty_actor_policy='ONE_RED_REFLECTION_RESTORE_PRE_GENERATION_CHECKPOINT',
        green_generation_policy='free raw action; first_line during generation',
        green_command_trie=False,yellow_red_command_constraints='unchanged')
    save(directory/'identity.json', identity)
    ledger = Ledger(task_limits(prompt_limit,settings))
    controller = None
    started = time.monotonic()
    with ExitStack() as stack:
        events = stack.enter_context((directory/'events.jsonl').open('x', encoding='utf-8', buffering=1))
        def emit(event, **data):
            events.write(json.dumps(dict(event=event, time=time.time(), **data),
                                    ensure_ascii=False, allow_nan=False)+'\n')
        client = BoundResidentClient(binding, '%s_task%d' % (run_id, task), model,
                                     sampling=sampling, request_timeout=settings.request_timeout)
        stack.callback(client.close)
        try:
            health = client.begin_run()
            stack.callback(client.end_run)
            validate_runtime_settings(health,settings,sampling,prompt_limit)
            if (health['model_class'] != model['model_class'] or health['torch_dtype'] != model['torch_dtype']
                    or 'first_nonempty_line' not in health['stop_modes']):
                raise ProtocolError('resident lacks declared model identity or nonempty-line framing')
            save(directory/'health_before.json', health)
            base = (PACKAGE/'prompts/webshop/webshop_reflexion_actionly_prompt.txt').read_text(encoding='utf-8')
            episode = WebV16Episode(website, '%s_fixed_%d' % (run_id, task), task, base,
                lambda messages: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            stack.callback(episode.close)
            def command_tokens(commands, limit):
                encoded = tuple(tuple(tokenizer.encode(c, add_special_tokens=False)) for c in commands)
                for command, tokens in zip(commands, encoded):
                    if not tokens or len(tokens)+1 > limit or tokenizer.decode(tokens, clean_up_tokenization_spaces=False) != command:
                        raise ProtocolError('command tokenization does not roundtrip')
                return encoded
            oracle = ProposalOracle(client.generate, master_seed=1, emit=emit,
                token_counter=lambda p: len(encode_prompt(tokenizer, p)['input_ids']),
                context_limit=health['model_context_tokens'], prompt_limit=min(prompt_limit, health['max_prompt_tokens']),
                request_namespace=run_id+'/task%d' % task, generation_versions=health,
                command_language_validator=command_tokens)
            controller = WebV16Controller(episode, oracle, ledger, task=task, emit=emit, settings=settings)
            stack.callback(controller.close)
            controller.start()
            state = run_controller(controller)
            save(directory/'scoped_state.json', state)
            if state['status'] == 'RESOURCE_LIMIT':
                save(directory/'resource_limit.json', dict(identity=identity,
                    error_type='BudgetExhausted',error=state['resource_limit'],cost=ledger.to_dict()))
            score = episode.snapshot()['public_terminal_score']
            terminal_rewards = [o.receipt['public_terminal_score'] for o in controller.occurrences.values() if o.receipt['done']]
            summary = dict(identity=identity, complete=True, status=state['status'],
                **purchase_score_report(score, terminal_rewards),
                forward_actions=len(controller.occurrences), replay_actions=0,
                cost=ledger.to_dict(), elapsed_seconds=time.monotonic()-started,
                resource_limit=state.get('resource_limit'))
            if settings.max_recoveries != 1:
                summary['recovery_stages'] = list(controller.recovery_stages)
            save(directory/'summary.json', summary)
            print('TASK_COMPLETE', json.dumps(dict(task=task, success=summary['success'],
                  score=summary['benchmark_reward'], final_public_reward=score)), flush=True)
            return summary
        except Exception as error:
            save(directory/'fatal.json', dict(identity=identity, error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), cost=ledger.to_dict()))
            if controller is not None:
                save(directory/'interrupted_state.json', controller.to_dict())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True, help='one resident GPU binding JSON object')
    parser.add_argument('--tasks', type=Path, required=True, help='JSON list of unique task IDs, 0..499')
    add_model_arguments(parser)
    add_experiment_arguments(parser)
    add_sampling_arguments(parser)
    parser.add_argument('--prompt-limit', type=int, default=32000)
    parser.add_argument('--website', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings, sampling = settings_from_args(args,Experiment), settings_from_args(args,Sampling)
    if not re.fullmatch('[A-Za-z0-9_-]+', args.run_id):
        raise ProtocolError('run ID must be a WebShop-safe unique session prefix')
    binding = json.loads(args.binding.read_text())
    verify_runtime(binding, os.environ, socket.gethostname())
    tasks = json.loads(args.tasks.read_text())
    if (type(tasks) is not list or not tasks or any(type(t) is not int or not 0 <= t < 500 for t in tasks)
            or len(tasks) != len(set(tasks))):
        raise ProtocolError('task manifest must be unique integer IDs in [0,499]')
    if not args.website.startswith(('http://127.0.0.1:', 'http://localhost:')):
        raise ProtocolError('WebShop must run on allocated node loopback')
    response = requests.get(args.website.rstrip('/')+'/__star_interface', timeout=30)
    response.raise_for_status()
    backend = response.json()
    verify_backend(backend, binding, settings.environment_seed)
    model, tokenizer, _ = load_model_bundle(args.model_path, args.dtype)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'run_config.json', dict(run_id=args.run_id, binding=binding,
        backend=backend, tasks=tasks, model_config=model, version=VERSION,
        experiment_settings=settings.as_dict(),sampling_settings=sampling.as_dict()))
    rows = [run_case(task, binding, backend, tokenizer, model, args.output, args.run_id,
                     args.website, prompt_limit=args.prompt_limit,settings=settings,sampling=sampling) for task in tasks]
    save(args.output/'run_done.json', dict(complete=True, completed=len(rows),
        successes=sum(r['success'] for r in rows), success_rate=sum(r['success'] for r in rows)/len(rows),
        R_percent=100*sum(r['benchmark_reward'] for r in rows)/len(rows),
        SR_percent=100*sum(r['success'] for r in rows)/len(rows),
        benchmark_rewards=[r['benchmark_reward'] for r in rows],
        scoring_policy='webshop_best_observed_purchase_v1',
        final_public_rewards=[r['final_public_reward'] for r in rows],
        budget_exhausted_without_terminal=sum(r['final_public_reward'] is None for r in rows)))


if __name__ == '__main__':
    main()
