"""Dataset branches for explicitly adapted ToT/LATS, using real public receipts."""
import copy
import json
from pathlib import Path

from resident_transport import append

PACKAGE=Path(__file__).resolve().parent


class BranchWorld:
    def __init__(self,dataset,task_id,template,authors,transport,directory):
        self.dataset,self.task_id,self.directory=dataset,task_id,directory
        self.cache={};self.snapshots={};self.replay_calls=0;self.transition_calls=0
        self.transport=transport
        if dataset=='webshop':
            from webshop_methods import RecordedEnvironment
            self.env=RecordedEnvironment(authors['react'].webshopEnv,template['website'],task_id,directory,transport)
            self.horizon=15;self.examples=authors['react'].prompt1
            observation,reward,done=self.env.step(self.env.session,'reset')
        elif dataset=='alfworld':
            from alfworld_methods import Environment,PREFIXES
            game=template['games'][task_id]
            family=[v for k,v in PREFIXES.items() if Path(game).parent.parent.name.startswith(k)]
            if len(family)!=1:raise ValueError('unknown ALFWorld task family')
            demos=json.loads((PACKAGE/'vendor/ga/prompts/alfworld/alfworld_base_prompts.json').read_text())
            self.examples='Interact with a household to solve a task. Here are two examples.\n'+demos['react_'+family[0]+'_1']+demos['react_'+family[0]+'_0']
            self.env=Environment(game,template['config'],directory,transport);self.horizon=49
            obs,_=self.env.reset();observation='\n'.join(obs[0].split('\n\n')[1:]);reward=0;done=False
        elif dataset=='game24':
            from game24_methods import environment_for
            self.env=environment_for(template,task_id,directory,transport);self.horizon=12
            self.examples=(PACKAGE/'vendor/ga/prompts/game24'/authors['react'].REACT_FILE).read_text()
            observation=self.env.get_case();reward=0;done=False
        else:raise ValueError('unsupported search environment')
        self.cache[()]=dict(observation=observation,reward=reward,terminal=done)
        self.snapshots[()]=self.snapshot()

    def snapshot(self):
        if self.dataset=='webshop':return copy.deepcopy(self.env.raw.sessions)
        if self.dataset=='game24':return dict(idx=self.env.idx,step_cnt=self.env.step_cnt,value_cache=list(self.env.value_cache))
        return None

    def restore(self,actions):
        if self.dataset=='webshop':self.env.raw.sessions=copy.deepcopy(self.snapshots[actions])
        elif self.dataset=='game24':
            for k,v in self.snapshots[actions].items():setattr(self.env,k,copy.deepcopy(v))
        else:
            obs,_=self.env.reset()
            if '\n'.join(obs[0].split('\n\n')[1:])!=self.cache[()]['observation']:
                raise ValueError('ALFWorld replay root differs from original observation')
            for i,action in enumerate(actions):
                receipt=self.native_step(action);self.replay_calls+=1
                expected=self.cache[actions[:i+1]]
                if receipt['observation']!=expected['observation'] or receipt['reward']!=expected['reward']:
                    raise ValueError('ALFWorld branch replay differs from original receipt')
        append(self.directory/'branches.jsonl',dict(event='restore',actions=actions,replay_calls=self.replay_calls))

    def native_step(self,action):
        if self.dataset=='webshop':
            try:observation,reward,done=self.env.step(self.env.session,action)
            except AssertionError:
                # The author WebShop methods expose this same environment rejection.
                observation,reward,done='Invalid action!',0,False
        elif self.dataset=='alfworld':
            obs,_,done,info=self.env.step([action])
            observation,reward,done=obs[0],int(info['won'][0]),bool(done[0])
        else:
            receipt=self.env.step(action)
            observation,reward=receipt['ob'],receipt['r']
            done=action.startswith('answer') or bool(reward)
        return dict(observation=observation,reward=reward,terminal=done)

    def at(self,actions):
        actions=tuple(actions)
        if actions in self.cache:return self.cache[actions]
        parent=actions[:-1]
        self.at(parent)
        if self.cache[parent]['terminal']:raise ValueError('cannot extend a terminal branch')
        self.restore(parent)
        receipt=self.native_step(actions[-1]);self.transition_calls+=1
        receipt['terminal']=receipt['terminal'] or len(actions)>=self.horizon
        self.cache[actions]=receipt;self.snapshots[actions]=self.snapshot()
        append(self.directory/'branches.jsonl',dict(event='transition',actions=actions,receipt=receipt))
        return receipt

    def trajectory(self,actions):
        text=self.cache[()]['observation']
        for i,action in enumerate(actions):
            text+='\nAction: '+action+'\nObservation: '+self.at(actions[:i+1])['observation']
        return text

    def close(self):
        if self.dataset!='game24':self.env.close()


class LatsEnvironment:
    """LATS stores branch action prefixes; restoration is delegated to the dataset."""
    def __init__(self,world):self.world=world;self.sessions={}
    def clone_state(self):return copy.deepcopy(self.sessions)
    def step(self,session,action):
        if session!='fixed_'+str(self.world.task_id):raise ValueError('search changed task ID')
        if action=='reset':actions=()
        else:actions=tuple(self.sessions[session]['actions'])+(action,)
        receipt=self.world.at(actions)
        self.sessions={session:dict(actions=actions)}
        return receipt['observation'],receipt['reward'],receipt['terminal']
