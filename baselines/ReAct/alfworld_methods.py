"""Native GA ALFWorld controllers with exact task IDs and isolated environments."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace

from resident_transport import PipelineTransport,GARollbackTransport,append

PACKAGE=Path(__file__).resolve().parent
ENV_LOCK=RLock()
PREFIXES={'pick_and_place':'put','pick_clean_then_place':'clean','pick_heat_then_place':'heat',
          'pick_cool_then_place':'cool','look_at_obj':'examine','pick_two_obj':'puttwo'}


def load_authors():
    folder=PACKAGE/'vendor/ga/test_alfworld'
    os.chdir(folder);sys.path.insert(0,str(folder))
    return {key:importlib.import_module(name) for key,name in
            [('react','alf_react'),('reflexion','alf_reflexion'),('ga_rollback','alf_rollback')]}


def load_inputs(req):
    import yaml
    from importlib.metadata import version
    if version('alfworld')!='0.3.5':raise ValueError('ALFWorld version mismatch')
    games=json.loads(Path(req['manifest']).read_text())['test']
    if len(games)!=134 or len(set(games))!=134:raise ValueError('expected exact134 manifest')
    config=yaml.safe_load((PACKAGE/'vendor/ga/test_alfworld/base_config.yaml').read_text())
    config['general']['random_seed']=req['environment_seed']
    return dict(games=games,config=config)


class Environment:
    def __init__(self,gamefile,config,directory,transport):
        from alfworld.agents.environment import AlfredTWEnv
        self.gamefile,self.directory,self.transport=gamefile,directory,transport
        provider=object.__new__(AlfredTWEnv)
        provider.config=copy.deepcopy(config)
        provider.train_eval='eval_out_of_distribution'
        provider.game_files=[gamefile];provider.num_games=1
        with ENV_LOCK:self.raw=provider.init_env(batch_size=1)
        self.batch_env=SimpleNamespace(reset=self.replay_reset)
        self.won=False;self.calls=0;self.resets=0

    def reset(self):
        with ENV_LOCK:result=self.raw.reset()
        if result[1]['extra.gamefile'][0]!=self.gamefile:raise ValueError('reset changed task')
        self.won=False;self.resets+=1
        append(self.directory/'events.jsonl',dict(event='environment_reset',trial=self.transport.trial,
            gamefile=self.gamefile,observation=result[0]))
        return result

    def replay_reset(self):
        with ENV_LOCK:result=self.raw.batch_env.reset()
        self.won=False;self.resets+=1
        append(self.directory/'events.jsonl',dict(event='author_rollback_reset',trial=self.transport.trial,
            gamefile=self.gamefile,observation=result[0]))
        return result

    def step(self,actions):
        with ENV_LOCK:result=self.raw.step(actions)
        self.calls+=1;self.won=bool(result[3]['won'][0])
        append(self.directory/'events.jsonl',dict(event='environment_step',trial=self.transport.trial,
            action=actions[0],observation=result[0][0],done=bool(result[2][0]),won=self.won))
        return result

    def close(self):
        with ENV_LOCK:self.raw.close()


def execute(method,task_id,template,authors,tokenizer,transport,directory,mode='act'):
    author=authors[method];gamefile=template['games'][task_id]
    family=[value for prefix,value in PREFIXES.items() if Path(gamefile).parent.parent.name.startswith(prefix)]
    if len(family)!=1:raise ValueError('unknown ALFWorld task family')
    prompt_name='alfworld_ssr_base_prompts.json' if method=='ga_rollback' else 'alfworld_base_prompts.json'
    demos=json.loads((PACKAGE/'vendor/ga/prompts/alfworld'/prompt_name).read_text())
    actor_mode='react' if method=='react' else mode
    base='Interact with a household to solve a task. Here are two examples.\n'+demos[f'{actor_mode}_{family[0]}_1']+demos[f'{actor_mode}_{family[0]}_0']
    env=Environment(gamefile,template['config'],directory,transport)
    try:
        if method=='reflexion':
            model=author.Local_llm.__new__(author.Local_llm)
            model.tokenizer=tokenizer;model.generation_pipe=PipelineTransport(transport)
            configs=[dict(name='env_'+str(task_id),memory=[],is_success=False,skip=False)]
            trials=[]
            for trial in range(2):
                if configs[0]['is_success']:break
                transport.trial,transport.role=trial,'actor'
                ob,info=env.reset();ob='\n'.join(ob[0].split('\n\n')[1:])
                history,success=author.alfworld_run(env,base,configs[0]['memory'],model,to_print=False,ob=ob)
                if bool(success)!=env.won:raise ValueError('author done-as-success differs from native won')
                configs[0]['is_success']=bool(success)
                trials.append(dict(trial=trial,success=bool(success)))
                path=directory/f'trial_{trial}.log'
                path.write_text(f'\n#####\n\nEnvironment #0:\n{history}\n\nSTATUS: {"OK" if success else "FAIL"}\n\n#####\n',encoding='utf-8')
                transport.role='reflection';configs=author.update_memory(str(path),configs,model)
                (directory/f'trial_{trial}.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
            result=dict(success=configs[0]['is_success'],reward=int(configs[0]['is_success']),memory=configs[0]['memory'],trials=trials)
        else:
            ob,info=env.reset();ob='\n'.join(ob[0].split('\n\n')[1:])
            if method=='react':
                reward=author.alfworld_run(env,base+'\nHere is the task.\n',tokenizer,PipelineTransport(transport),
                    to_print=False,ob=ob,local_path='resident')
                result=dict(success=bool(reward),reward=int(bool(reward)))
            elif method=='ga_rollback':
                model=GARollbackTransport(transport)
                reward,rollbacks=author.alfworld_run(env,base,model,model,False,to_print=False,ob=ob,max_rollback_num=6,wait_k=6)
                result=dict(success=bool(reward),reward=int(bool(reward)),rollbacks=rollbacks)
            else:raise ValueError('unsupported ALFWorld method')
        result.update(environment_calls=env.calls,environment_resets=env.resets,gamefile=gamefile)
        return result
    finally:env.close()
