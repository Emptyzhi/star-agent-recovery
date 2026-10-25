"""Disclosed dataset adaptations of native ToT BFS and native LATS MCTS."""
import json
import logging
import random
from types import SimpleNamespace

from author_binding import bind_module
from search_environments import BranchWorld,LatsEnvironment


def chat_generator(tokenizer,transport,cap):
    def gpt(prompt,model=None,temperature=None,max_tokens=None,n=1,stop=None):
        if temperature is not None and temperature!=transport.sampling['temperature']:
            raise ValueError('search sampling differs from declared settings')
        request=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
        outputs=[]
        for _ in range(n):
            raw=transport.generate(request,cap if max_tokens is None else max_tokens)['raw_output']
            if stop is not None:
                stops=[stop] if isinstance(stop,str) else stop
                offsets=[raw.index(s) for s in stops if s in raw]
                if offsets:raw=raw[:min(offsets)]
            outputs.append(raw)
        return outputs
    return gpt


class InteractiveToT:
    def __init__(self,world):
        self.world=world;self.steps=world.horizon;self.stops=['\n']*self.steps;self.value_cache={}
    def get_input(self,idx):return self.world.cache[()]['observation']
    def actions(self,y):return tuple(y.splitlines())
    def cot_prompt_wrap(self,x,y):
        return self.world.examples+'\n\nCurrent branch:\n'+self.world.trajectory(self.actions(y))+'\nReturn one next environment action on one line, without an Action prefix.\nAction:'
    def value_prompt_wrap(self,x,y):
        return ('Assess whether this partial environment trajectory can complete its task. '
                'Use only the observed information. Explain briefly, then finish with exactly '
                'one label: impossible, likely, or sure.\n'+self.world.trajectory(self.actions(y)))
    def value_outputs_unwrap(self,x,y,outputs):
        # The three labels and weights follow native Game24 ToT.
        weights={'impossible':.001,'likely':1,'sure':20}
        labels=[s.split('\n')[-1] for s in outputs]
        return sum(weights[label]*labels.count(label) for label in weights)


def execute_tot(world,author,tokenizer,transport):
    task=InteractiveToT(world);gpt=chat_generator(tokenizer,transport,1000)
    core=bind_module(author,gpt=gpt)
    sample=core.get_samples;value=core.get_value
    def get_samples(task,x,y,*args,**kwargs):
        if world.at(task.actions(y))['terminal']:return [y]
        return [candidate+'\n' for candidate in sample(task,x,y,*args,**kwargs)]
    def get_value(task,x,y,*args,**kwargs):
        receipt=world.at(task.actions(y))
        if receipt['terminal']:return 20*receipt['reward']
        return value(task,x,y,*args,**kwargs)
    core.get_samples,core.get_value=get_samples,get_value
    args=SimpleNamespace(backend='resident',temperature=transport.sampling['temperature'],
        method_generate='sample',method_evaluate='value',method_select='greedy',
        n_generate_sample=5,n_evaluate_sample=3,n_select_sample=5,prompt_sample='cot')
    outputs,tree=core.solve(args,task,world.task_id,to_print=False)
    if not outputs:raise ValueError('ToT search produced no candidate')
    receipts=[world.at(task.actions(y)) for y in outputs]
    (world.directory/'search_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
    return dict(success=receipts[0]['reward']==1,reward=receipts[0]['reward'],
        final_public_reward=receipts[0]['reward'] if receipts[0]['terminal'] else None,
        candidate_rewards=[r['reward'] for r in receipts],outputs=outputs,
        selection='first LLM-ranked final candidate',
        adaptation='interactive ToT: sample5/value3/greedy5; terminal branches retained without extra actions')


class DatasetLatsTask:
    def __init__(self,world,gpt,native_task):
        self.world,self.gpt,self.native_task=world,gpt,native_task
        self.value_cache={}
    def cot_prompt_wrap(self,x,y='',reflections=()):
        memory='\n'.join(r['trajectory']+'\nReflection: '+r['reflection'] for r in reflections)
        return self.world.examples+'\n\nPrevious failed search paths and reflections:\n'+memory+'\n\nCurrent branch:\n'+x+y
    def value_prompt_wrap(self,x,y,failures=(),reflections=()):
        memory='\n'.join(r['trajectory']+'\nReflection: '+r['reflection'] for r in reflections)
        return ('Evaluate this trajectory toward its task using only its observations. '
                'Give a correctness score from 1 to 10, where 10 means the task is solved. '
                'End with: Thus the correctness score is <score>.\n'
                'Earlier failures and reflections:\n'+memory+'\nCurrent trajectory:\n'+y)
    def value_outputs_unwrap(self,outputs):return self.native_task.value_outputs_unwrap(outputs)
    def generate_self_reflection(self,failures,question):
        result=[]
        for failure in random.sample(failures,min(3,len(failures))):
            trajectory=failure['trajectory']+'\nReward: '+str(failure['r'])
            prompt=('This attempt failed. Diagnose the observed mistake and give a concise plan '
                    'for a new attempt. Do not assume unobserved facts.\n'+trajectory+'\nReflection:')
            result.append(dict(question=question,trajectory=trajectory,reflection=self.gpt(prompt)[0]))
        return result


def execute_lats(world,author,tokenizer,transport):
    source,native_task=author;gpt=chat_generator(tokenizer,transport,100)
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
    args=SimpleNamespace(backend='resident',temperature=transport.sampling['temperature'],
        n_generate_sample=5,n_evaluate_sample=1,prompt_sample='cot',log=str(world.directory/'search.log'))
    try:
        state,value,reward,em=core.lats_search(args,task,'fixed_'+str(world.task_id),iterations=30,to_print=False)
        ids={id(node):i for i,node in enumerate(nodes)}
        tree=[dict(id=i,parent=None if node.parent is None else ids[id(node.parent)],
            state=node.state,depth=node.depth,visits=node.visits,value=node.value,reward=node.reward,
            is_terminal=node.is_terminal,env_state=node.env_state) for i,node in enumerate(nodes)]
        (world.directory/'search_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        return dict(success=reward==1,reward=reward,selected_state=state,
            reward_policy='author best search-node reward',reflection_map=core.reflection_map,
            adaptation='dataset public receipts and horizon; zero-reward terminal failure reflections')
    finally:handler.close();logger.removeHandler(handler)


def execute(dataset,method,task_id,template,authors,tokenizer,transport,directory,mode='act'):
    world=BranchWorld(dataset,task_id,template,authors,transport,directory)
    try:
        runner=execute_tot if method=='tot' else execute_lats
        result=runner(world,authors[method],tokenizer,transport)
        result.update(dataset_adaptation=True,environment_horizon=world.horizon,
            unique_environment_transitions=world.transition_calls,environment_replay_steps=world.replay_calls)
        return result
    finally:world.close()
