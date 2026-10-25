"""Author ToT BFS with task-local model binding and original candidate metrics."""
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from author_binding import bind_module

PACKAGE=Path(__file__).resolve().parent


def unbound_gpt(*args, **kwargs):
    raise RuntimeError('ToT generation must be bound to this task resident transport')


def load_tot_core():
    sys.path.insert(0,str(PACKAGE/'vendor/tot/src'))
    # Replace only the network transport import; all task/search source stays intact.
    if 'tot.models' in sys.modules and sys.modules['tot.models'].gpt is not unbound_gpt:
        raise RuntimeError('ToT model module was already imported before transport binding')
    if 'tot.models' not in sys.modules:
        module=ModuleType('tot.models')
        module.gpt=unbound_gpt
        sys.modules['tot.models']=module
    return importlib.import_module('tot.methods.bfs')


def load_tot(csv):
    bfs=load_tot_core()
    task_module=importlib.import_module('tot.tasks.game24')
    return bfs,task_module.Game24Task(file=str(csv))


def execute_tot(task_id, author, tokenizer, transport):
    bfs,template=author
    task=copy.copy(template)
    task.value_cache={}

    def gpt(prompt,model=None,temperature=.1,max_tokens=1000,n=1,stop=None):
        if temperature != transport.sampling['temperature']:
            raise ValueError('ToT sampling differs from declared benchmark sampling')
        if stop is not None:
            raise ValueError('propose/value ToT profile does not use text stop sequences')
        request=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
        return [transport.generate(request,max_tokens)['raw_output'] for _ in range(n)]

    bound=bind_module(bfs,gpt=gpt)
    args=SimpleNamespace(backend='resident',temperature=transport.sampling['temperature'],
        method_generate='propose',method_evaluate='value',method_select='greedy',
        n_generate_sample=1,n_evaluate_sample=3,n_select_sample=5,prompt_sample=None)
    outputs,tree=bound.solve(args,task,task_id,to_print=False)
    scores=[task.test_output(task_id,y)['r'] for y in outputs]
    if not scores:
        raise ValueError('author ToT search returned no final candidates')
    # The author reports both candidate mean and any-candidate success. Keep both;
    # a single-answer score uses the top-ranked candidate, never the best oracle score.
    return dict(success=bool(scores[0]),reward=scores[0],candidate_rewards=scores,
        candidate_mean=sum(scores)/len(scores),any_candidate_success=any(scores),
        selection='first LLM-ranked final candidate',outputs=outputs,search=tree)
