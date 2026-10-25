"""AgentRewind with selectable models/datasets and explicit action/recovery budgets."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import re
import sys
import traceback

from agentrewind_bench.config import resolve
from agentrewind_bench.environments import EnvironmentProcess
from agentrewind_bench.models import OpenAIModel,ResidentModel,prepare_resident

ROOT = Path(__file__).resolve().parent


def recovery_limit(value):
    if value == 'unlimited':
        return None
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('use a nonnegative integer or unlimited') from error
    if limit < 0:
        raise argparse.ArgumentTypeError('use a nonnegative integer or unlimited')
    return limit


def task_deadline(value):
    if value=='unlimited':
        return None
    return float(value)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=('game24','alfworld','webshop'))
    p.add_argument('--model','--model-path','--local_llm_path',dest='model',required=True)
    p.add_argument('--backend',choices=('resident','openai'),default='resident')
    p.add_argument('--endpoint',required=True,help='Resident root URL, or OpenAI-compatible /v1 base URL')
    p.add_argument('--api-key-env',help='Name of the environment variable containing the API key')
    p.add_argument('--dtype',choices=('bfloat16','float16'),default='bfloat16')
    p.add_argument('--native-adapter-root',type=Path,default=ROOT.parents[1])
    p.add_argument('--environment-python',type=Path,default=Path(sys.executable))
    p.add_argument('--task-ids',required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--initial-actions',type=int)
    p.add_argument('--recovery-actions',type=int)
    p.add_argument('--action-budget-policy',choices=('trajectory','segment'),default='trajectory',
                   help='trajectory: retained prefix plus new native actions shares the limit; segment: legacy fresh window after restore')
    p.add_argument('--max-terminal-failures',type=int,default=None,
        help='Stop after this many actual unsuccessful native terminal submissions; independent adaptation.')
    p.add_argument('--max-reflection-rejections',type=int,default=None,
        help='Stop after this many rejected commit submissions without a native action or successful restore.')
    p.add_argument('--recovery-patience',type=int,choices=(2,),default=None,
        help='After two recovery decisions without restore/accepted action, require an accepted native action before reopening recovery; end if no action can run.')
    p.add_argument('--max-recoveries',type=recovery_limit,default=1,
                   help='Nonnegative count or unlimited; action allowance follows action-budget-policy')
    p.add_argument('--max-output-tokens',type=int,default=1200)
    p.add_argument('--history-format',choices=('flat','dialogue'),default='flat',
                   help='flat: legacy native transcript; dialogue: actual assistant response and observation turns')
    p.add_argument('--prompt-format',choices=('plain','native-chat'),default='native-chat',
                   help='plain: direct native text continuation for every request; native-chat: explicit historical chat-framing variant')
    p.add_argument('--candidate-history', choices=('full','once'), default='full',
                   help='once: latest full candidate receipt and lossless references in earlier receipts; original logs remain complete')
    p.add_argument('--runtime-prompt',choices=('full','compact'),default='full',
                   help='compact: remove duplicate adapter observations and repeated runtime prose; preserve author guidance and public state')
    p.add_argument('--recovery-format',choices=('bracket-json','python','lines'),default='bracket-json',
                   help='Transport author arguments as bracketed JSON or literal named Python function calls; neither executes model code')
    p.add_argument('--interface-format',choices=('native','sectioned','json-tools'),default='native',
                   help='Sectioned plain text separates examples, actual feedback and controller state; commits require an explicit final boundary')
    p.add_argument('--actor-output-tokens',type=int,default=100,
                   help='Native action output cap; combined recovery commit uses max-output-tokens')
    p.add_argument('--text-output-boundary',choices=('eos','command'),default='eos',
                   help='command: resident stops when the sampled text completes one recovery command; requires sectioned and a supporting resident')
    p.add_argument('--text-history-format',choices=('sectioned','native'),default='sectioned',
                   help='native: preserve dataset-native action/observation labels and place actual current commands immediately before the next decision cue')
    p.add_argument('--recovery-prompt-policy',choices=('full','staged-compact'),default='full',
                   help='staged-compact: short action guidance, full recovery guidance and correction-first concise memory; preserves author notebook limits')
    p.add_argument('--task-timeout',type=task_deadline,default=None,
                   help='Seconds, or unlimited (author default: no episode deadline)')
    p.add_argument('--request-timeout',type=float,default=600)
    p.add_argument('--response-grace-seconds',type=float,default=60,
                   help='Resident HTTP receipt wait after the server generation deadline; no extra inference budget')
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--top-p',type=float,default=1)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--repetition-penalty',type=float,default=1.05)
    p.add_argument('--environment-seed',type=int,default=1)
    p.add_argument('--candidate-limit',type=int,default=80)
    p.add_argument('--rejected-history-keep',type=int,choices=(0,2),default=0,
                   help='0: complete dialogue; 2: project older rejected proposals within one pending candidate phase')
    p.add_argument('--memory-limit',type=int,default=20)
    p.add_argument('--notebook-item-char-limit',type=int,choices=(900,4000),default=900,
                   help='Author notebook item rendering: original 900, or explicit 4000 lines-protocol variant; total remains 8000')
    p.add_argument('--context-limit',type=int,default=32768)
    p.add_argument('--prompt-limit',type=int,default=31000)
    p.add_argument('--context-overflow',choices=('stop','head-tail'),default='stop',
                   help='head-tail: on overflow only, omit middle dialogue and bound candidate display previews; no model summarization')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--run-namespace',required=True)
    p.add_argument('--csv',type=Path)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--environment-config',type=Path)
    p.add_argument('--split',choices=('test','development'),default='test')
    p.add_argument('--website')
    p.add_argument('--parent-job-id')
    p.add_argument('--node')
    p.add_argument('--step-id')
    p.add_argument('--resident-step-id',
                   help='Actual existing model-service step; defaults to the current runner step')
    p.add_argument('--gpu-index',type=int)
    p.add_argument('--dry-run',action='store_true')
    return p


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def environment_config(args,settings,task,output):
    return dict(dataset=args.dataset,task_id=task,native_adapter_root=str(args.native_adapter_root.resolve()),
                csv=None if args.csv is None else str(args.csv.resolve()),
                manifest=None if args.manifest is None else str(args.manifest.resolve()),
                environment_config=None if args.environment_config is None else str(args.environment_config.resolve()),
                split=args.split,website=args.website,environment_seed=settings.environment_seed,
                initial_actions=settings.initial_actions,session_id=args.run_namespace+'_fixed_'+str(task),
                environment_trace=str((output/'environment_events.jsonl').resolve()))


def run_task(args,settings,task,bundle):
    from agentrewind_bench.controller import Controller
    output = args.output_dir/('task_%03d'%task)
    output.mkdir(exist_ok=False)
    environment = model = controller = None
    identity = dict(dataset=args.dataset,task_id=task,run_id=args.run_namespace,experiment=settings.as_dict())
    save(output/'identity.json',identity)
    try:
        environment = EnvironmentProcess(args.environment_python,environment_config(args,settings,task,output),output)
        model = (OpenAIModel(args,settings,args.run_namespace+'_task'+str(task)) if args.backend=='openai' else
                 ResidentModel(args,settings,args.run_namespace+'_task'+str(task),bundle))
        identity['model_identity'] = model.identity
        save(output/'identity.json',identity)
        controller = Controller(environment,model,settings,output,args.run_namespace+'_task'+str(task))
        result = controller.run()
        environment.close()
        model.close()
        save(output/'summary.json',dict(identity=identity,**result))
        return result
    except Exception as error:
        save(output/'fatal.json',dict(identity=identity,error_type=type(error).__name__,error=str(error),
             traceback=traceback.format_exc(),cost=None if controller is None else controller.cost()))
        raise
    finally:
        if environment is not None:
            environment.close()
        if model is not None:
            model.close()


def main(argv=None):
    args = parser().parse_args(argv)
    settings,tasks = resolve(args)
    if re.fullmatch(r'[A-Za-z0-9_-]{1,100}',args.run_namespace) is None:
        raise ValueError('run namespace must be a safe unique identifier')
    resolved = dict(experiment=settings.as_dict(),tasks=tasks,workers=args.workers,
                    native_adapter_root=str(args.native_adapter_root.resolve()),
                    environment_python=str(args.environment_python),model_rng='continuous resident RNG; no request reseeding',
                    upstream_root=str(ROOT/'vendor/upstream'),selection_policy='agent-direct',
                    context_policy='rollback',memory_policy='cumulative',
                    environment_restore='native benchmark adapter; cumulative costs are never rewound')
    resolved['runtime'] = dict(backend=args.backend,endpoint=args.endpoint,dtype=args.dtype,
        prompt_limit=args.prompt_limit if args.backend=='resident' else None,
        context_limit=args.context_limit if args.backend=='resident' else None,
        context_enforcement='local tokenizer' if args.backend=='resident' else 'API server',
        parent_job_id=args.parent_job_id,node=args.node,step_id=args.step_id,
        resident_step_id=args.resident_step_id,gpu_index=args.gpu_index)
    resolved['environment'] = dict(website=args.website,split=args.split,
        csv=None if args.csv is None else str(args.csv.resolve()),
        manifest=None if args.manifest is None else str(args.manifest.resolve()),
        environment_config=None if args.environment_config is None else str(args.environment_config.resolve()))
    if args.dry_run:
        print(json.dumps(resolved,ensure_ascii=False,indent=2))
        return 0
    bundle = prepare_resident(args,settings) if args.backend=='resident' else None
    args.output_dir.mkdir(parents=True,exist_ok=False)
    save(args.output_dir/'resolved_settings.json',resolved)
    rows,failures = [],[]
    with ThreadPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:
        futures = {pool.submit(run_task,args,settings,task,bundle):task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(dict(task_id=futures[future],type=type(error).__name__,error=str(error)))
    report = dict(complete=not failures,expected_tasks=len(tasks),completed=len(rows),fatal=len(failures),
                  failures=failures,successful_tasks=sum(r['success'] for r in rows))
    save(args.output_dir/'run_done.json',report)
    print(json.dumps(report,ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
