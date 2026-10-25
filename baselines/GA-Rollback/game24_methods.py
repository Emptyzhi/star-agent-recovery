"""Execute GA repository's Game24 method bodies with isolated task state."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time

from author_binding import bind_module
from resident_transport import PipelineTransport, GARollbackTransport, append

PACKAGE = Path(__file__).resolve().parent


def load_authors():
    directory = PACKAGE/'vendor/ga/test_game24'
    os.chdir(directory)
    sys.path.insert(0,str(directory))
    return {name:importlib.import_module(module) for name,module in
            [('react','g24_react'),('reflexion','g24_reflexion'),('ga_rollback','g24_rollback')]}


def environment_for(template, task_id, directory, transport):
    env = copy.copy(template)
    env.idx, env.step_cnt, env.value_cache = task_id-1, 0, []
    native_step, native_get_case = env.step, env.get_case

    def step(action):
        before = dict(numbers=list(env.value_cache), step_count=env.step_cnt)
        result = native_step(action)
        append(directory/'events.jsonl', dict(event='environment_step', time=time.time(),
            trial=transport.trial, role=transport.role, action=action, before=before,
            after=dict(numbers=list(env.value_cache),step_count=env.step_cnt), receipt=result))
        return result

    def get_case(reset=False):
        result = native_get_case(reset=reset)
        append(directory/'events.jsonl', dict(event='environment_reset', time=time.time(),
            trial=transport.trial, reset=reset, task_id=env.idx, puzzle=result))
        return result

    env.step, env.get_case = step, get_case
    return env


def execute(method, task_id, template, authors, tokenizer, transport, directory, mode='act'):
    if method == 'tot':
        from tot_game24 import execute_tot
        return execute_tot(task_id,authors['tot'],tokenizer,transport)
    author = authors[method]
    if method == 'reflexion':
        def factory(file_path, seed, sample_num):
            if seed != template.seed or sample_num != template.sample_num:
                raise ValueError('author changed environment selection')
            return environment_for(template,task_id,directory,transport)
        author = bind_module(author, Game24Task=factory)
        model = author.Local_llm.__new__(author.Local_llm)
        model.tokenizer = tokenizer
        model.generation_pipe = PipelineTransport(transport)
        configs = [dict(name='env_'+str(task_id),memory=[],is_success=False,skip=False)]
        trials = []
        for trial in range(2):
            transport.trial, transport.role = trial, 'actor'
            path = directory/('trial_%d.log'%trial)
            author.run_trial(template.seed,template.sample_num,str(path),str(directory/'world.log'),
                             trial,mode,configs,True,model)
            trials.append(dict(trial=trial,success=bool(configs[0]['is_success'])))
            transport.role = 'reflection'
            configs = author.update_memory(str(path),configs,model)
            (directory/('trial_%d.json'%trial)).write_text(json.dumps(configs,indent=2),encoding='utf-8')
        return dict(success=bool(configs[0]['is_success']),trials=trials,memory=configs[0]['memory'])
    env = environment_for(template,task_id,directory,transport)
    puzzle = env.get_case()
    prompt_file = author.REACT_FILE if method=='react' or mode=='react' else author.ACTIONLY_FILE
    prompt = (PACKAGE/'vendor/ga/prompts/game24'/prompt_file).read_text(encoding='utf-8')
    if method == 'react':
        model = author.Local_llm.__new__(author.Local_llm)
        model.tokenizer = tokenizer
        model.generation_pipe = PipelineTransport(transport)
        reward = author.game24_run(puzzle,env,prompt,model,to_print=False)
    elif method == 'ga_rollback':
        model = GARollbackTransport(transport)
        reward = author.game24_run(puzzle,env,prompt,model,model,6,to_print=True)
    else:
        raise ValueError('unsupported Game24 method')
    return dict(success=bool(reward),reward=reward)
