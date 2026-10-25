"""Task-isolated author LATS WebShop search on the resident model and website."""
import copy
import importlib
import inspect
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from threading import Lock,local
from uuid import uuid4

from .author_binding import bind_module
from .resident_transport import append
from .datasets import native_url
from .round_budget import install_round_search, ROUND_ACTIONS
from .candidate_execution import candidate_executor


class WebShopWorkers:
    """Persistent task-local workers with private native globals and HTTP sessions.

    Native fixed-session routing reads the goal index from the final suffix.
    Each worker gets a separate server session for that same goal, in addition
    to its private client dictionary; server purchase/log state cannot race.
    """
    def __init__(self, source, website, directory, log, task_id):
        self.source,self.website,self.directory,self.log=source,website,directory,log
        self.task_id=task_id
        self.local=local();self.lock=Lock();self.http_sessions=[]
        self.session_prefix='lats_'+uuid4().hex
        self.root_observation=None

    def environment(self):
        if hasattr(self.local,'env'):
            return self.local.env
        import requests
        bound=bind_module(self.source,logging=self.log,WEBSHOP_URL=self.website)
        http=requests.Session()
        with self.lock:
            self.http_sessions.append(http)
            self.local.session=f'{self.session_prefix}_{len(self.http_sessions)}_fixed_{self.task_id}'
        native_text=bound.webshop_text
        signature=inspect.signature(native_text)
        current_url=None
        def get(_native_url):
            response=http.get(current_url,timeout=1200)
            with self.lock:
                append(self.directory/'http.jsonl',dict(url=current_url,status=response.status_code))
            response.raise_for_status()
            return response
        bound.requests=SimpleNamespace(get=get)
        def render(*args,**kwargs):
            nonlocal current_url
            call=signature.bind(*args,**kwargs);call.apply_defaults()
            state=dict(call.arguments);state.update(state.pop('kwargs',{}))
            current_url=native_url(self.website,state)
            return native_text(*args,**kwargs)
        bound.webshop_text=render
        self.local.env=bound.webshopEnv()
        self.local.initialized=False
        return self.local.env

    def execute(self, snapshot, session, action):
        if session!='fixed_'+str(self.task_id):
            raise ValueError('LATS task identity changed')
        env=self.environment()
        remote_session=self.local.session
        if not self.local.initialized and action!='reset':
            root=env.step(remote_session,'reset')
            with self.lock:
                append(self.directory/'events.jsonl',dict(event='environment_worker_reset',
                    server_session=remote_session,task_id=self.task_id,receipt=root))
            if root[0]!=self.root_observation or root[1]!=0 or root[2]:
                raise ValueError('isolated WebShop worker reset changed task observation')
            self.local.initialized=True
        before=copy.deepcopy(snapshot)
        env.sessions={remote_session:copy.deepcopy(snapshot[session])} if snapshot else {}
        if env.sessions and 'session' in env.sessions[remote_session]:
            env.sessions[remote_session]['session']=remote_session
        invalid=False
        try:
            receipt=env.step(remote_session,action)
        except AssertionError:
            if action=='reset':
                raise
            # Native LATS explicitly treats an invalid environment action as
            # observation "Invalid action!", reward -1, nonterminal. Preserve
            # that method behavior and the original exception evidence.
            import traceback
            with self.lock:
                append(self.directory/'events.jsonl',dict(event='environment_invalid_action',
                    action=action,before=before,traceback=traceback.format_exc()))
            receipt=('Invalid action!',-1,False)
            invalid=True
        if action=='reset':
            self.root_observation=receipt[0]
            self.local.initialized=True
        after={session:env.clone_state()[remote_session]}
        if 'session' in after[session]:
            after[session]['session']=session
        return dict(receipt=receipt,before=before,after=after,invalid=invalid,
                    server_session=remote_session)

    def close(self):
        for http in self.http_sessions:
            http.close()


def parse_webshop_action(text):
    # Same first-line framing as the existing WebShop Act-only binding.
    action=re.sub(r'^Action:\s*','',text.strip(),count=1,flags=re.I)
    if not action:
        return None
    if '\n' in action:
        raise ValueError('LATS WebShop actor returned no single-line action candidate')
    return action


def execute_webshop_candidates(outputs, submit, directory):
    if not isinstance(outputs, list):
        raise ValueError('candidate generation must finish before environment execution')
    results=[]
    for ordinal,raw in enumerate(outputs):
        action=parse_webshop_action(raw)
        # The author creates a child only when action_line is nonempty.
        if action is None:
            append(directory/'events.jsonl',dict(event='candidate_unparsed',
                ordinal=ordinal,raw_output=raw,reason='empty_action'))
            continue
        results.append((ordinal,action,submit(ordinal,action).result()))
    return results


def rollout_webshop(bound,node,args,task,idx,max_depth,directory):
    start_depth=node.depth
    while (not node.is_terminal and node.depth<bound.iteration_depth_limit
           and node.depth-start_depth<max_depth):
        children=[]
        children=bound.generate_new_states(node,args,task,idx,args.n_generate_sample)
        if not children:
            append(directory/'events.jsonl',dict(event='rollout_no_progress',
                depth=node.depth,samples=args.n_generate_sample))
            return node
        for child in children:
            if child.is_terminal:
                return child
        prompts=[bound.generate_prompt(child) for child in children]
        values=bound.get_values(task,node.question,prompts,args.n_evaluate_sample)
        if len(values)!=len(children):
            raise ValueError('LATS WebShop rollout scores do not match candidates')
        node=children[values.index(max(values))]
    if not node.is_terminal:
        node.reward=-0.5
    return node

def load_lats():
    return (importlib.import_module('vendor.lats.webshop.lats'),
            importlib.import_module('vendor.lats.webshop.webshop'))


def execute_lats(task_id,author,website,model,directory,config,environment_pool=None):
    core,task_source=author
    logger=logging.getLogger(str(directory));logger.propagate=False;logger.setLevel(logging.INFO)
    handler=logging.FileHandler(directory/'search.log',encoding='utf-8');logger.addHandler(handler)
    log=SimpleNamespace(INFO=logging.INFO,info=logger.info,warning=logger.warning,debug=logger.debug,
                        basicConfig=lambda **kwargs:None)

    gpt=model.for_role('evaluation')
    bound=bind_module(core,gpt=gpt,logging=log,WEBSHOP_URL=website)
    task_module=bind_module(task_source,gpt=model.for_role('reflection'),gpt4=model.for_role('reflection'),tokenizer=model.tokenizer,logging=log)
    task=task_module.WebShopTask()
    from .score_protocol import value_prompt, recorded_score
    score_protocol_failures=[]
    def numeric_value_prompt(x, y, failures=(), reflections=()):
        memory='\n'.join(r['trajectory']+'\nReflection: '+r['reflection'] for r in reflections)
        return value_prompt('webshop', y, memory)
    task.value_prompt_wrap=numeric_value_prompt
    task.value_outputs_unwrap=lambda outputs: recorded_score(
        outputs, 1, 10, score_protocol_failures, 'LATS WebShop')
    action_prompt=(Path(__file__).resolve().parents[1]/'prompts/webshop/webshop_reflexion_actionly_prompt.txt').read_text(encoding='utf-8')
    def action_only_prompt(x,y='',reflection_mapping_list=()):
        memory='\n'.join(r['trajectory']+'\nReflection: '+r['reflection']
                         for r in reflection_mapping_list)
        prefix=('Previous failed search paths and reflections:\n'+memory+'\n\n') if memory else ''
        return prefix+action_prompt+'\n\nHere is the current task:\n'+x+y
    task.cot_prompt_wrap=action_only_prompt
    workers=WebShopWorkers(core,website,directory,log,task_id)
    candidate_pool,owns_pool=candidate_executor(environment_pool,config.candidates)
    calls=[];terminals=[]
    def record(action, result):
        receipt=result['receipt']
        if not result['invalid']:
            calls.append(receipt)
            if receipt[2]:terminals.append(receipt[1])
            append(directory/'events.jsonl',dict(event='environment_step',action=action,
                 before=result['before'],after=result['after'],receipt=receipt,
                 server_session=result['server_session']))
        return receipt

    def generate(node,args,task,idx,n):
        x=bound.generate_prompt(node)
        if len(bound.failed_trajectories)>len(bound.reflection_map) and len(bound.failed_trajectories)<4:
            bound.reflection_map=task.generate_self_reflection(bound.failed_trajectories,x)
        prompt=task.cot_prompt_wrap(x,'\nAction: ',bound.reflection_map)
        def submit(ordinal,action):
            return candidate_pool.submit(workers.execute,copy.deepcopy(node.env_state),idx,action)
        outputs=model.for_role('candidate')(prompt,n=n,stop='\n')
        results=execute_webshop_candidates(outputs,submit,directory)
        unique_states={};added=False
        for _,action,result in results:
            observation,reward,done=record(action,result)
            state=node.state.copy()
            state.update(action=action,observation=observation)
            child=bound.Node(state=state,question=node.question,env_state=result['after'],parent=node)
            child.is_terminal=reward>0 or done
            child.reward=reward;child.value=reward
            # Native LATS executes duplicate actions and retains the last
            # sample's node at the first occurrence's dictionary position.
            unique_states[action]=child
            if child.is_terminal and 0<reward<1 and not added:
                if reward not in [row['r'] for row in bound.failed_trajectories]:
                    added=True
                    bound.failed_trajectories.append(dict(trajectory=bound.collect_trajectory(child),
                                                         final_answer=action,r=reward))
        children=list(unique_states.values())
        return children
    bound.generate_new_states=generate
    def evaluate(node,args,task,idx):
        pending=[child for child in node.children if not child.is_terminal]
        votes=bound.get_values(task,node.question,
            [bound.generate_prompt(child) for child in pending],args.n_evaluate_sample)
        if len(votes)!=len(pending):
            raise ValueError('LATS WebShop expansion scores do not match candidates')
        valid=[value for value in votes if value>=0]
        denominator=max(valid) if valid and max(valid)>0 else 1
        for child,value in zip(pending,votes):
            child.value=value/denominator if value>=0 else -1
        for child in node.children:
            if child.is_terminal:
                child.value=(denominator+1)/denominator
        return sum(child.value for child in node.children)/len(node.children) if node.children else 0
    bound.evaluate_node=evaluate
    def expand(node,args,task,idx):
        if node.depth>=bound.iteration_depth_limit:return
        count=args.n_generate_sample*(2 if node.depth==0 else 1)
        children=bound.generate_new_states(node,args,task,idx,count)
        node.children.extend(children)
    bound.expand_node=expand
    def rollout(node,args,task,idx,max_depth=15):
        return rollout_webshop(bound,node,args,task,idx,max_depth,directory)
    bound.rollout=rollout
    install_round_search(bound,config.round_actions or ROUND_ACTIONS['webshop'],directory)
    env=SimpleNamespace(sessions={});bound.env=env
    def step(session,action):
        result=candidate_pool.submit(workers.execute,copy.deepcopy(env.sessions),session,action).result()
        env.sessions=result['after']
        return record(action,result)
    env.step=step
    nodes=[];NativeNode=bound.Node
    class RecordedNode(NativeNode):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);nodes.append(self)
    bound.Node=RecordedNode
    args=SimpleNamespace(backend='resident',temperature=model.temperature,
        n_generate_sample=config.candidates,n_evaluate_sample=config.eval_samples,prompt_sample='cot',log=str(directory/'search.log'))
    try:
        state,value,reward,em=bound.lats_search(args,task,'fixed_'+str(task_id),iterations=config.iterations,to_print=False)
        ids={id(node):i for i,node in enumerate(nodes)}
        tree=[dict(id=i,parent=None if n.parent is None else ids[id(n.parent)],state=n.state,
            children=[ids[id(c)] for c in n.children],depth=n.depth,visits=n.visits,value=n.value,
            reward=n.reward,is_terminal=n.is_terminal,env_state=n.env_state) for i,n in enumerate(nodes)]
        (directory/'search_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        return dict(success=reward==1,reward=reward,selected_state=state,selected_value=value,
            terminal_rewards=terminals,environment_calls=len(calls),
            value_protocol_failures=score_protocol_failures,
            model_failure_policy='empty_skip_unscored_v8',
            reward_policy='author best search-node reward',
            final_public_reward=calls[-1][1] if calls and calls[-1][2] else None,
            node_count=len(nodes),reflection_map=bound.reflection_map)
    finally:
        if owns_pool:
            candidate_pool.shutdown(wait=True,cancel_futures=True)
        workers.close();handler.close();logger.removeHandler(handler)
