"""Dataset branches for adapted LATS, using real public receipts."""
import copy
import json
from pathlib import Path
from threading import local

from .resident_transport import append

PACKAGE=Path(__file__).resolve().parents[1]
_WORKER_ENVS={}
_GAME24_WORKER=local()


def run_isolated_game24_candidate(puzzle, checkpoint, action):
    """A worker owns its arithmetic environment; siblings share no mutable state."""
    from .game24_env import Game24Environment
    env=getattr(_GAME24_WORKER,'env',None)
    if env is None or env.puzzle!=puzzle:
        env=Game24Environment(puzzle)
        _GAME24_WORKER.env=env
    env.restore(checkpoint)
    receipt=env.execute(action)
    observation=receipt.detail
    if not receipt.done:
        observation+='; numbers left: '+' '.join(receipt.after['numbers'])
    return (dict(observation=observation,reward=receipt.reward or 0,
                 terminal=receipt.done), env.checkpoint())


def _execute_alfworld_candidate(env, root_observation, cached_receipts, actions, directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    env.directory=directory
    obs,_=env.reset()
    root='\n'.join(obs[0].split('\n\n')[1:])
    if root!=root_observation:
        raise ValueError('isolated ALFWorld replay root differs from original observation')
    receipt=None
    for i,action in enumerate(tuple(actions)):
        obs,_,done,info=env.step([action])
        receipt=dict(observation=obs[0],reward=int(info['won'][0]),terminal=bool(done[0]))
        expected=cached_receipts.get(tuple(actions[:i+1]))
        if expected is not None and receipt!=expected:
            raise ValueError('isolated ALFWorld branch replay differs from cached receipt')
        if receipt['terminal'] and i+1<len(actions):
            raise ValueError('isolated ALFWorld branch extends a terminal receipt')
    if receipt is None:
        raise ValueError('isolated ALFWorld candidate has no action')
    return receipt


def run_isolated_alfworld_candidate(gamefile, config, root_observation,
                                    cached_receipts, actions, directory):
    """Run one candidate in a separate process with its own TextWorld parser."""
    from .datasets import Environment
    env=Environment(gamefile,config,Path(directory),None)
    try:
        return _execute_alfworld_candidate(env,root_observation,cached_receipts,actions,directory)
    finally:
        env.close()


def _worker_environment(gamefile, config, directory):
    # Called only by a process-pool worker, which executes one job at a time.
    env=_WORKER_ENVS.get(gamefile)
    if env is None:
        from .datasets import Environment
        env=Environment(gamefile,config,Path(directory),None)
        _WORKER_ENVS[gamefile]=env
    env.directory=Path(directory)
    return env


def run_isolated_alfworld_root(gamefile, config, directory):
    env=_worker_environment(gamefile,config,directory)
    obs,_=env.reset()
    return '\n'.join(obs[0].split('\n\n')[1:])


def run_isolated_alfworld_job(job):
    action,gamefile,config,root_observation,cached_receipts,actions,directory=job
    env=_worker_environment(gamefile,config,directory)
    return action,_execute_alfworld_candidate(
        env,root_observation,cached_receipts,actions,directory)


class BranchWorld:
    def __init__(self,dataset,task_id,template,authors,transport,directory):
        self.dataset,self.task_id,self.directory=dataset,task_id,directory
        self.cache={};self.snapshots={};self.replay_calls=0;self.transition_calls=0
        self.transport=transport
        if dataset=='alfworld':
            from .datasets import PREFIXES
            from .search_adaptations import get_alfworld_candidate_pool
            self.environment_pool=get_alfworld_candidate_pool()
            game=template['games'][task_id]
            self.gamefile=game;self.env_config=template['config']
            family=[v for k,v in PREFIXES.items() if Path(game).parent.parent.name.startswith(k)]
            if len(family)!=1:raise ValueError('unknown ALFWorld task family')
            demos=json.loads((PACKAGE/'vendor/ga/prompts/alfworld/alfworld_base_prompts.json').read_text(encoding='utf-8'))
            self.examples='Interact with a household to solve a task. Return one action at a time. Here are two examples.\n'+demos['act_'+family[0]+'_1']+demos['act_'+family[0]+'_0']
            self.horizon=50
            observation=self.environment_pool.submit(
                run_isolated_alfworld_root,game,self.env_config,directory).result()
            reward=0;done=False
        elif dataset=='game24':
            from .game24_env import Game24Environment
            self.env=Game24Environment(template['puzzles'][task_id]);self.horizon=self.env.max_actions
            self.examples=(PACKAGE/'prompts/game24/game24_base_actionly.txt').read_text(encoding='utf-8')
            observation=self.env.puzzle;reward=0;done=False
        else:raise ValueError('unsupported search environment')
        self.cache[()]=dict(observation=observation,reward=reward,terminal=done)
        self.snapshots[()]=self.snapshot()

    def isolated_at(self, actions, candidate_id):
        """Execute one ALFWorld candidate in a private environment instance."""
        actions=tuple(actions)
        if self.dataset!='alfworld':
            return self.at(actions)
        job=(actions[-1],self.gamefile,self.env_config,self.cache[()]['observation'],
             self.cache,actions,self.directory/'candidate_envs'/str(candidate_id))
        return self.environment_pool.submit(run_isolated_alfworld_job,job).result()[1]

    def record_isolated(self, actions, receipt, snapshot=None):
        actions=tuple(actions)
        if actions in self.cache:
            if self.cache[actions]!=receipt:
                raise ValueError('isolated candidate conflicts with cached receipt')
            if self.dataset=='game24' and self.snapshots[actions]!=snapshot:
                raise ValueError('isolated candidate conflicts with cached checkpoint')
            return self.cache[actions]
        self.cache[actions]=receipt
        self.snapshots[actions]=snapshot
        self.transition_calls+=1
        append(self.directory/'branches.jsonl',dict(event='isolated_transition',actions=actions,receipt=receipt))
        return receipt

    def snapshot(self):
        if self.dataset=='game24':return self.env.checkpoint()
        return None

    def restore(self,actions):
        if self.dataset=='game24':
            self.env.restore(self.snapshots[actions])
        else:
            raise RuntimeError('ALFWorld replay must execute as an isolated process job')
        append(self.directory/'branches.jsonl',dict(event='restore',actions=actions,replay_calls=self.replay_calls))

    def native_step(self,action):
        if self.dataset=='alfworld':
            raise RuntimeError('ALFWorld steps must execute as an isolated process job')
        else:
            receipt=self.env.execute(action)
            numbers=' '.join(receipt.after['numbers'])
            observation=f'{receipt.detail}; numbers left: {numbers}'
            if receipt.done: observation=receipt.detail
            reward=receipt.reward or 0
            done=receipt.done
        return dict(observation=observation,reward=reward,terminal=done)

    def at(self,actions):
        actions=tuple(actions)
        if actions in self.cache:return self.cache[actions]
        parent=actions[:-1]
        self.at(parent)
        if self.cache[parent]['terminal']:raise ValueError('cannot extend a terminal branch')
        if self.dataset=='alfworld':
            receipt=self.isolated_at(actions,'main_'+str(self.transition_calls))
            self.replay_calls+=len(parent)
            return self.record_isolated(actions,receipt)
        self.restore(parent)
        receipt=self.native_step(actions[-1]);self.transition_calls+=1
        self.cache[actions]=receipt;self.snapshots[actions]=self.snapshot()
        append(self.directory/'branches.jsonl',dict(event='transition',actions=actions,receipt=receipt))
        return receipt

    def trajectory(self,actions):
        text=self.cache[()]['observation']
        for i,action in enumerate(actions):
            text+='\nAction: '+action+'\nObservation: '+self.at(actions[:i+1])['observation']
        return text

    def close(self):
        # The shard owns and shuts down the persistent environment process pool.
        pass


class LatsEnvironment:
    """LATS stores branch action prefixes; restoration is delegated to the dataset."""
    def __init__(self,world):self.world=world;self.sessions={}
    def clone_state(self):return copy.deepcopy(self.sessions)
    def step(self,session,action):
        if session!='fixed_'+str(self.world.task_id):raise ValueError('search changed task ID')
        if action=='reset':actions=()
        else:actions=tuple(self.sessions[session]['actions'])+(action,)
        receipt=self.world.at(actions)
        # Preserve the post-action state object for the child node.
        self.sessions[session]=dict(actions=actions)
        return receipt['observation'],receipt['reward'],receipt['terminal']
