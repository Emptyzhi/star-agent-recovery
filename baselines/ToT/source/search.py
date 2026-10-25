"""Author BFS order: complete layer generation, then values, then selection."""
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from .author_binding import bind_module
from .resident_transport import append

ROOT=Path(__file__).resolve().parents[1]


def unbound(*args, **kwargs):
    raise RuntimeError('ToT model must be bound to its task')


def load_native():
    sys.path.insert(0,str(ROOT/'vendor/tot/src'))
    if 'tot.models' in sys.modules and sys.modules['tot.models'].gpt is not unbound:
        raise RuntimeError('another ToT model module is already loaded')
    if 'tot.models' not in sys.modules:
        module=ModuleType('tot.models');module.gpt=unbound;sys.modules['tot.models']=module
    return importlib.import_module('tot.methods.bfs'),importlib.import_module('tot.tasks.game24').Game24Task


def solve_round(task_id, task, native_bfs, model, config, directory, temperature, interactive=False):
    if config.method_generate!='sample':
        raise ValueError('multiline proposals conflict with individual first-line candidates; select sample')
    args=SimpleNamespace(backend='resident',temperature=temperature,method_generate='sample',
        method_evaluate='value',method_select='greedy',n_generate_sample=config.candidates,
        n_evaluate_sample=config.eval_samples,n_select_sample=config.beam_width,prompt_sample='cot')
    task.steps=config.max_depth if interactive else min(config.max_depth,4)
    task.stops=['\n']*config.max_depth
    task.value_cache={}
    core=bind_module(native_bfs,gpt=model.gpt)
    native_value,native_values=core.get_value,core.get_values

    def terminal(y):
        return interactive and task.receipt(y)['terminal']

    def samples(task,x,y,n_generate_sample,prompt_sample,stop):
        if terminal(y):return [y]
        prompt=(task.cot_prompt_wrap(x,y) if prompt_sample=='cot' else task.standard_prompt_wrap(x,y))
        count=config.root_candidates if not y and config.root_candidates is not None else n_generate_sample
        model.role='candidate'
        action_stop=None if hasattr(task,'serialize_action') else stop
        outputs=model.gpt(prompt,n=count,stop=action_stop,model=args.backend,temperature=args.temperature)
        candidates=[]
        for raw in outputs:
            action=task.normalize_action(raw) if hasattr(task,'normalize_action') else raw
            if hasattr(task,'serialize_action'):action=task.serialize_action(action)
            candidates.append(y+action+'\n')
        append(directory/'expansions.jsonl',dict(mode='sample',parent=y,expanded_candidates=candidates))
        return candidates

    def value(task,x,y,*pos,**kwargs):
        if terminal(y):return 20*config.eval_samples*task.receipt(y)['reward']
        model.role='evaluation'
        return native_value(task,x,y,*pos,**kwargs)

    def values(task,x,ys,n_evaluate_sample,cache_value=True):
        # Unmodified BFS has collected candidates from all parents before this call.
        if interactive:
            for y in ys:task.receipt(y)
        result=native_values(task,x,ys,n_evaluate_sample,cache_value=cache_value)
        append(directory/'layers.jsonl',dict(event='layer_scored',candidates=ys,values=result,scoring='author_sequential'))
        return result

    core.get_samples,core.get_value,core.get_values=samples,value,values
    outputs,tree=core.solve(args,task,task_id,to_print=False)
    if not outputs:raise ValueError('ToT search returned no final candidates')
    scores=[task.test_output(task_id,y)['r'] for y in outputs]
    return dict(outputs=outputs,search=tree,candidate_rewards=scores,candidate_mean=sum(scores)/len(scores),
        any_candidate_success=any(s==1 for s in scores),first_ranked_reward=scores[0],
        first_ranked_success=scores[0]==1,selection='first ranked final candidate',dataset_adaptation=interactive)


def solve(task_id,task,native_bfs,model,config,directory,temperature,interactive=False):
    if config.iterations != 1:
        raise ValueError('ToT permits exactly one BFS search')
    append(directory/'rounds.jsonl',dict(event='round_started',iteration=1,
        selected_depth=0,action_budget=config.max_depth,policy='single_bfs'))
    result=solve_round(task_id,task,native_bfs,model,config,directory,temperature,interactive)
    append(directory/'rounds.jsonl',dict(event='round_finished',iteration=1,
        search_depth=len(result['search']['steps']),first_ranked_reward=result['first_ranked_reward'],
        first_ranked_success=result['first_ranked_success']))
    result.update(rounds_executed=1,search_iterations_limit=1,
        round_path_actions=config.max_depth,budget_policy='single_bfs',
        search_protocol='ga_act_only' if interactive else 'native')
    return result
