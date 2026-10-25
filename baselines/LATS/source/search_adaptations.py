"""Existing ALFWorld/Game24 adaptations of native LATS MCTS."""
import json
import logging
import multiprocessing
import random
import re
from .executors import ProcessPoolExecutor
from threading import RLock
from types import SimpleNamespace

from .author_binding import bind_module
from .score_protocol import recorded_score
from .search_environments import LatsEnvironment, run_isolated_alfworld_job
from .round_budget import install_round_search
from .candidate_execution import candidate_executor, execute_candidates
from .search_environments import run_isolated_game24_candidate
import copy


_ALFWORLD_POOL_LOCK=RLock()
_ALFWORLD_CANDIDATE_POOL=None


def configure_alfworld_candidate_pool(max_workers):
    global _ALFWORLD_CANDIDATE_POOL
    with _ALFWORLD_POOL_LOCK:
        if _ALFWORLD_CANDIDATE_POOL is None:
            _ALFWORLD_CANDIDATE_POOL=ProcessPoolExecutor(
                max_workers=max_workers,mp_context=multiprocessing.get_context('spawn'))
        return _ALFWORLD_CANDIDATE_POOL


def shutdown_alfworld_candidate_pool():
    global _ALFWORLD_CANDIDATE_POOL
    with _ALFWORLD_POOL_LOCK:
        pool=_ALFWORLD_CANDIDATE_POOL; _ALFWORLD_CANDIDATE_POOL=None
    if pool is not None:
        pool.shutdown(wait=True,cancel_futures=True)


def get_alfworld_candidate_pool():
    with _ALFWORLD_POOL_LOCK:
        if _ALFWORLD_CANDIDATE_POOL is None:
            raise RuntimeError('ALFWorld process pool must be configured before creating tasks')
        return _ALFWORLD_CANDIDATE_POOL


def parse_score_line(outputs, lower, upper, label):
    from .score_protocol import parse_score
    return parse_score(outputs, lower, upper)


class DatasetLatsTask:
    def __init__(self,world,gpt,native_task):
        self.world,self.gpt,self.native_task=world,gpt,native_task
        self.value_cache={}
        self.score_protocol_failures=[]
        self.examples=world.examples
    def cot_prompt_wrap(self,x,y='',reflections=()):
        memory='\n'.join(r['trajectory']+'\nReflection: '+r['reflection'] for r in reflections)
        return (self.examples+'\n\nPrevious failed search paths and reflections:\n'+memory+
                '\n\nContinue the current branch with exactly one action. '
                'Return only the action content on one line. Do not output think, '
                'an observation, or a second action.\n\nCurrent branch:\n'+x+y+'\nAction:')
    def value_prompt_wrap(self,x,y,failures=(),reflections=()):
        from .score_protocol import value_prompt
        memory='\n'.join(r['trajectory']+'\nReflection: '+r['reflection'] for r in reflections)
        return value_prompt(self.world.dataset, y, memory)
    def value_outputs_unwrap(self,outputs):
        return recorded_score(outputs, 1, 10, self.score_protocol_failures, 'LATS')
    def generate_self_reflection(self,failures,question):
        result=[]
        for failure in random.sample(failures,min(3,len(failures))):
            trajectory=failure['trajectory']+'\nReward: '+str(failure['r'])
            prompt=('This attempt failed. Diagnose the observed mistake and give a concise plan '
                    'for a new attempt. Do not assume unobserved facts.\n'+trajectory+'\nReflection:')
            result.append(dict(question=question,trajectory=trajectory,reflection=self.gpt(prompt)[0]))
        return result


class ActionOnlyFeedback:
    """Keep LATS failure reflection while changing only the actor protocol."""
    generate_self_reflection = DatasetLatsTask.generate_self_reflection

    def cot_prompt_wrap(self, x, y='', reflections=()):
        memory = '\n'.join(r['trajectory']+'\nReflection: '+r['reflection']
                           for r in reflections)
        return ('Previous failed search paths and reflections:\n'+memory+'\n\n' if memory else '')+x+y


def record_binary_failure(core, node):
    # The author records one failed terminal per distinct reward. Binary
    # environments use zero instead of WebShop's partial terminal reward.
    if node.is_terminal and node.reward == 0 and not any(
            item['r'] == 0 for item in core.failed_trajectories):
        core.failed_trajectories.append(dict(
            trajectory=core.collect_trajectory(node),
            final_answer=node.state['action'], r=0))


class Game24ActionOnlyTask(ActionOnlyFeedback):
    """LATS task adapter using the same action-only actor contract as STAR."""
    def __init__(self, world, gpt):
        self.world, self.gpt = world, gpt
        self.score_protocol_failures=[]
        self.examples = world.examples

    @staticmethod
    def actions(node):
        rows=[]
        while node is not None and node.parent is not None:
            rows.append((node.state.get('action',''), node.state.get('observation','')))
            node=node.parent
        return list(reversed(rows))

    @staticmethod
    def parse_action(raw):
        """Select the first complete Game24 action from the full actor response."""
        if not isinstance(raw, str):
            raise ValueError('Game24 action output must be text')
        from .game24_env import ANSWER_RE, OPERATION_RE

        for line in raw.splitlines():
            action_marker = re.search(r'\bAction\s*:\s*(.*)$', line, flags=re.I)
            candidate = action_marker.group(1).strip() if action_marker else line.strip()
            candidate = re.sub(r'^(?:Act|Action)\s*\d*\s*[>:]\s*', '', candidate,
                               count=1, flags=re.I).strip()
            if OPERATION_RE.fullmatch(candidate) or ANSWER_RE.fullmatch(candidate):
                return candidate
        # An unparseable proposal is an attempted invalid action, not a runner
        # error. The public environment rejects it and the search scores the
        # resulting unchanged state before selecting its next branch.
        return raw.strip()

    def action_prompt(self, node):
        history=''.join(f'Act {i}> {action}\nObs {i}> {observation}\n'
                        for i,(action,observation) in enumerate(self.actions(node),1))
        step=len(self.actions(node))+1
        return (self.examples+'\n\nContinue with exactly one public action, with no introduction, '
                'reasoning, or Markdown.\nReturn only one action on the next line. A binary action '
                'is exactly <number> <+|-|*|/> <number> = <number> and must use two current '
                'number occurrences. A final action must begin with answer: and use each ORIGINAL '
                'input occurrence exactly once. Check the arithmetic before emitting it.\n'
                f'# Here is the task:\nInput: {self.world.env.puzzle}\n{history}Act {step}>')

    def value_prompt_wrap(self, node):
        from .score_protocol import value_prompt
        history=''.join(f'Act {i}> {action}\nObs {i}> {observation}\n'
                        for i,(action,observation) in enumerate(self.actions(node),1))
        return value_prompt('game24', f'Input: {self.world.env.puzzle}\n{history}')

    def value_outputs_unwrap(self, outputs):
        return recorded_score(outputs, 0, 10, self.score_protocol_failures, 'LATS action-only Game24')


class AlfWorldActionOnlyTask(ActionOnlyFeedback):
    """LATS task adapter using STAR's native `>` first-line action contract."""
    def __init__(self, world, gpt):
        self.world, self.gpt = world, gpt
        self.score_protocol_failures=[]
        self.examples = world.examples

    @staticmethod
    def actions(node):
        rows=[]
        while node is not None and node.parent is not None:
            rows.append((node.state.get('action',''), node.state.get('observation','')))
            node=node.parent
        return list(reversed(rows))

    @staticmethod
    def parse_action(raw):
        """Parse one action after protocol whitespace, preserving strict framing."""
        if not isinstance(raw,str):
            raise ValueError('ALFWorld action-only actor output must be text')
        text=raw.lstrip('\ufeff \t\r\n')
        line=text.split('\n',1)[0].strip()
        marker=line.find('>')
        if marker > 2:
            marker=-1
        action=line[marker+1:].strip()
        if len(action)>1:
            action=action[0].lower()+action[1:]
        if not action:
            return None
        return action

    def action_prompt(self, node):
        history=''.join('> '+action+'\n'+observation+'\n'
                        for action,observation in self.actions(node))
        return (self.examples+'\n\nFor the current task, output exactly one valid ALFWorld action '
                'on the next line. Do not repeat an example, describe the room, explain your '
                'reasoning, or output an observation.\nHere is the task:\n'+
                self.world.cache[()]['observation']+'\n'+history+'>')

    def value_prompt_wrap(self, node):
        from .score_protocol import value_prompt
        history=''.join('> '+action+'\n'+observation+'\n'
                        for action,observation in self.actions(node))
        return value_prompt('alfworld', self.world.cache[()]['observation']+'\n'+history)

    def value_outputs_unwrap(self, outputs):
        return recorded_score(outputs, 0, 10, self.score_protocol_failures, 'LATS action-only ALFWorld')


def execute_alfworld_actionly(world, model, config, author):
    """Run native LATS search with STAR's action-only ALFWorld interface."""
    source,_=author; gpt=model.for_role('evaluation')
    logger=logging.getLogger(str(world.directory));logger.propagate=False;logger.setLevel(logging.INFO)
    handler=logging.FileHandler(world.directory/'search.log',encoding='utf-8');logger.addHandler(handler)
    log=SimpleNamespace(INFO=logging.INFO,info=logger.info,warning=logger.warning,debug=logger.debug,basicConfig=lambda **kw:None)
    core=bind_module(source,gpt=gpt,env=LatsEnvironment(world),logging=log)
    task=AlfWorldActionOnlyTask(world,model.for_role('reflection'))
    NativeNode=core.Node
    with _ALFWORLD_POOL_LOCK:
        candidate_pool=_ALFWORLD_CANDIDATE_POOL
    owns_pool=candidate_pool is None
    if owns_pool:
        candidate_pool=configure_alfworld_candidate_pool(max(1,config.candidates*2))

    def node_prompt(node): return task.action_prompt(node)

    def generate_new_states(node,args,unused_task,idx,n):
        children=[];seen=set()
        parent_actions=tuple(action for action,_ in task.actions(node))
        outputs=model.for_role('candidate')(node_prompt(node),n=n,stop=None)
        for ordinal,raw in enumerate(outputs):
            action=task.parse_action(raw)
            if action is None:
                from .resident_transport import append
                append(world.directory/'events.jsonl', dict(event='candidate_unparsed',
                    ordinal=ordinal, raw_output=raw, reason='empty_action'))
                continue
            if action in seen: continue
            seen.add(action)
            branch_id='node_'+str(id(node))+'_candidate_'+str(ordinal)
            actions=parent_actions+(action,)
            expected={prefix:world.cache[prefix] for prefix in
                      (actions[:i] for i in range(1,len(actions))) if prefix in world.cache}
            job=(action,world.gamefile,world.env_config,
                 world.cache[()]['observation'],expected,actions,
                 world.directory/'candidate_envs'/branch_id)
            isolated=candidate_pool.submit(run_isolated_alfworld_job,job).result()[1]
            receipt=world.record_isolated(parent_actions+(action,),isolated)
            child=RecordedNode(
                state={'action':action,'observation':receipt['observation']},
                question=node.question,
                env_state=dict(actions=parent_actions+(action,)),parent=node)
            child.reward=receipt['reward'];child.value=receipt['reward']
            child.is_terminal=receipt['terminal']
            children.append(child)
            record_binary_failure(core,child)
        return children

    def evaluate_children(children,args):
        for child in children:
            if child.is_terminal: continue
            child.value=task.value_outputs_unwrap(
                gpt(task.value_prompt_wrap(child), n=args.n_evaluate_sample, stop=None))
        return sum(child.value for child in children)/len(children) if children else 0

    def evaluate_node(node,args,unused_task,idx):
        return evaluate_children(node.children,args)

    def expand_node(node,args,unused_task,idx):
        if node.depth>=core.iteration_depth_limit:return
        count=args.n_generate_sample*(2 if node.depth==0 else 1)
        node.children.extend(generate_new_states(node,args,unused_task,idx,count))

    def rollout(node,args,unused_task,idx,max_depth=15):
        start_depth=node.depth
        while (not node.is_terminal and node.depth<core.iteration_depth_limit
               and node.depth-start_depth<max_depth):
            children=generate_new_states(node,args,unused_task,idx,args.n_generate_sample)
            if not children:
                from .resident_transport import append
                append(world.directory/'events.jsonl', dict(event='rollout_no_progress', depth=node.depth))
                break
            node.children.extend(children);evaluate_children(children,args)
            node=max(children,key=lambda child:child.value)
        return node

    core.generate_new_states=generate_new_states
    core.generate_prompt=node_prompt
    core.evaluate_node=evaluate_node
    core.expand_node=expand_node
    core.rollout=rollout
    install_round_search(core, config.round_actions or world.horizon, world.directory)
    nodes=[]
    class RecordedNode(NativeNode):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);nodes.append(self)
    core.Node=RecordedNode
    args=SimpleNamespace(backend='resident',temperature=model.temperature,
        n_generate_sample=config.candidates,n_evaluate_sample=config.eval_samples,
        prompt_sample='actionly',log=str(world.directory/'search.log'))
    try:
        state,value,reward,em=core.lats_search(args,task,'fixed_'+str(world.task_id),
            iterations=config.iterations,to_print=False)
        ids={id(node):i for i,node in enumerate(nodes)}
        tree=[dict(id=i,parent=None if node.parent is None else ids[id(node.parent)],
            state=node.state,depth=node.depth,visits=node.visits,value=node.value,
            reward=node.reward,is_terminal=node.is_terminal,env_state=node.env_state)
            for i,node in enumerate(nodes)]
        (world.directory/'search_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        return dict(success=reward==1,reward=reward,selected_state=state,
            reflection_map=core.reflection_map,
            value_protocol_failures=task.score_protocol_failures,
            reward_policy='strict ALFWorld terminal receipt; action-only LATS actor',
            model_failure_policy='empty_skip_unscored_v8',
            adaptation='STAR ALFWorld act prompt and native first-line action parser')
    finally:
        if owns_pool:
            shutdown_alfworld_candidate_pool()
        handler.close();logger.removeHandler(handler)


def execute_game24_actionly(world, model, config, author, environment_pool=None):
    """Run the LATS search core with strict STAR Game24 and action-only actors."""
    source,_=author; gpt=model.for_role('evaluation')
    logger=logging.getLogger(str(world.directory));logger.propagate=False;logger.setLevel(logging.INFO)
    handler=logging.FileHandler(world.directory/'search.log',encoding='utf-8');logger.addHandler(handler)
    log=SimpleNamespace(INFO=logging.INFO,info=logger.info,warning=logger.warning,debug=logger.debug,basicConfig=lambda **kw:None)
    core=bind_module(source,gpt=gpt,env=LatsEnvironment(world),logging=log)
    task=Game24ActionOnlyTask(world,model.for_role('reflection'))
    NativeNode=core.Node

    candidate_pool,owns_pool=candidate_executor(environment_pool,config.candidates)

    def node_prompt(node): return task.action_prompt(node)

    def generate_new_states(node,args,unused_task,idx,n):
        prompt=node_prompt(node)
        if len(core.failed_trajectories)>len(core.reflection_map) and len(core.failed_trajectories)<4:
            core.reflection_map=task.generate_self_reflection(core.failed_trajectories,prompt)
        prompt=task.cot_prompt_wrap(prompt,'',core.reflection_map)
        children=[]
        parent_actions=tuple(action for action,_ in task.actions(node))
        if world.cache[parent_actions]['terminal']:
            raise ValueError('cannot extend a terminal branch')
        checkpoint=world.snapshots[parent_actions]
        def submit(ordinal,action):
            return candidate_pool.submit(run_isolated_game24_candidate,
                world.env.puzzle,copy.deepcopy(checkpoint),action)
        outputs=model.for_role('candidate')(prompt,n=n,stop=None)
        results=execute_candidates(list(enumerate(outputs)),
                                   task.parse_action,submit,deduplicate=True)
        for _,action,(receipt,snapshot) in results:
            receipt=world.record_isolated(parent_actions+(action,),receipt,snapshot)
            state={'action':action,'observation':receipt['observation']}
            child=RecordedNode(state=state,question=node.question,
                env_state=dict(actions=parent_actions+(action,)),parent=node)
            child.reward=receipt['reward'];child.value=receipt['reward']
            child.is_terminal=receipt['terminal']
            children.append(child)
            record_binary_failure(core,child)
        if not children: raise ValueError('LATS action-only actor generated no candidates')
        return children

    def evaluate_children(children, args):
        pending_children = [child for child in children if not child.is_terminal]
        if pending_children:
            outputs=[gpt(task.value_prompt_wrap(child),n=args.n_evaluate_sample,stop=None)
                     for child in pending_children]
            for child, output in zip(pending_children, outputs):
                child.value = task.value_outputs_unwrap(output)
        return sum(child.value for child in children)/len(children) if children else 0

    def evaluate_node(node,args,unused_task,idx):
        return evaluate_children(node.children, args)

    def expand_node(node,args,unused_task,idx):
        if node.depth >= core.iteration_depth_limit: return
        count = args.n_generate_sample * (2 if node.depth == 0 else 1)
        node.children.extend(generate_new_states(node,args,unused_task,idx,count))

    def rollout(node,args,unused_task,idx,max_depth=15):
        start_depth=node.depth
        while (not node.is_terminal and node.depth < core.iteration_depth_limit
               and node.depth-start_depth < max_depth):
            children=generate_new_states(node,args,unused_task,idx,args.n_generate_sample)
            if not children: break
            node.children.extend(children)
            evaluate_children(children, args)
            node=max(children,key=lambda child: child.value)
        return node

    core.generate_new_states=generate_new_states
    core.generate_prompt=node_prompt
    core.evaluate_node=evaluate_node
    core.expand_node=expand_node
    core.rollout=rollout
    install_round_search(core, config.round_actions or world.horizon, world.directory)
    nodes=[]
    class RecordedNode(NativeNode):
        def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs); nodes.append(self)
    core.Node=RecordedNode
    args=SimpleNamespace(backend='resident',temperature=model.temperature,
        n_generate_sample=config.candidates,n_evaluate_sample=config.eval_samples,
        prompt_sample='actionly',log=str(world.directory/'search.log'))
    try:
        state,value,reward,em=core.lats_search(args,task,'fixed_'+str(world.task_id),
            iterations=config.iterations,to_print=False)
        tree=[dict(id=i,parent=None if n.parent is None else nodes.index(n.parent),state=n.state,
                   depth=n.depth,visits=n.visits,value=n.value,reward=n.reward,is_terminal=n.is_terminal)
              for i,n in enumerate(nodes)]
        (world.directory/'search_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        return dict(success=reward==1,reward=reward,selected_state=state,
            reflection_map=core.reflection_map,
            value_protocol_failures=task.score_protocol_failures,
            reward_policy='strict Game24 terminal receipt; action-only LATS actor',
            adaptation='STAR Game24Environment with exact public receipts')
    finally:
        if owns_pool:
            candidate_pool.shutdown(wait=True,cancel_futures=True)
        handler.close();logger.removeHandler(handler)


def execute_lats(world,author,model,config,environment_pool=None):
    dataset=getattr(world,'dataset',None)
    if dataset == 'game24':
        return execute_game24_actionly(world,model,config,author,environment_pool)
    if dataset == 'alfworld':
        return execute_alfworld_actionly(world,model,config,author)
    source,native_task=author;gpt=model.gpt
    logger=logging.getLogger(str(world.directory));logger.propagate=False;logger.setLevel(logging.INFO)
    handler=logging.FileHandler(world.directory/'search.log',encoding='utf-8');logger.addHandler(handler)
    log=SimpleNamespace(INFO=logging.INFO,info=logger.info,warning=logger.warning,debug=logger.debug,basicConfig=lambda **kw:None)
    core=bind_module(source,gpt=gpt,env=LatsEnvironment(world),logging=log)
    task=DatasetLatsTask(world,gpt,native_task.WebShopTask)
    generate=core.generate_new_states
    def generate_new_states(node,*args,**kwargs):
        nodes=generate(node,*args,**kwargs)
        if not nodes:raise ValueError('LATS generated no parseable Action line')
        # WebShop source records only partial rewards; binary tasks also fail at 0.
        for child in nodes:
            if child.is_terminal and child.reward==0:
                trajectory=core.generate_prompt(child)
                if not any(f['trajectory']==trajectory for f in core.failed_trajectories):
                    core.failed_trajectories.append(dict(trajectory=trajectory,final_answer=child.state['action'],r=0))
        return nodes
    core.generate_new_states=generate_new_states
    def expand_node(node,args,task,idx):
        if node is None:raise ValueError('LATS selected an exhausted tree')
        if node.depth>=world.horizon:return
        node.children.extend(core.generate_new_states(node,args,task,idx,args.n_generate_sample*(2 if node.depth==0 else 1)))
    core.expand_node=expand_node
    native_rollout=core.rollout
    def rollout(node,args,task,idx,max_depth=15):
        return native_rollout(node,args,task,idx,max_depth=max(0,world.horizon-node.depth))
    core.rollout=rollout
    nodes=[];NativeNode=core.Node
    class RecordedNode(NativeNode):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);nodes.append(self)
    core.Node=RecordedNode
    args=SimpleNamespace(backend='resident',temperature=model.temperature,
        n_generate_sample=config.candidates,n_evaluate_sample=config.eval_samples,prompt_sample='cot',log=str(world.directory/'search.log'))
    try:
        state,value,reward,em=core.lats_search(args,task,'fixed_'+str(world.task_id),iterations=config.iterations,to_print=False)
        ids={id(node):i for i,node in enumerate(nodes)}
        tree=[dict(id=i,parent=None if node.parent is None else ids[id(node.parent)],
            state=node.state,depth=node.depth,visits=node.visits,value=node.value,reward=node.reward,
            is_terminal=node.is_terminal,env_state=node.env_state) for i,node in enumerate(nodes)]
        (world.directory/'search_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        return dict(success=reward==1,reward=reward,selected_state=state,
            reward_policy='author best search-node reward',reflection_map=core.reflection_map,
            adaptation='dataset public receipts and horizon; zero-reward terminal failure reflections')
    finally:handler.close();logger.removeHandler(handler)
