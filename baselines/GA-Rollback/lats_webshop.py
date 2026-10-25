"""Task-isolated author LATS WebShop search on the resident model and website."""
import copy
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

from author_binding import bind_module
from resident_transport import append
from webshop_methods import native_url

PACKAGE=Path(__file__).resolve().parent


def unbound_gpt(*args,**kwargs):
    raise RuntimeError('LATS model calls must be bound to the task resident')


def load_lats():
    folder=PACKAGE/'vendor/lats/webshop'
    sys.path.insert(0,str(folder))
    if 'models' in sys.modules:raise RuntimeError('conflicting author models module')
    models=ModuleType('models');models.gpt=models.gpt4=unbound_gpt
    sys.modules['models']=models
    os.environ.setdefault('OPENAI_API_KEY','resident-adapter-unused')
    return importlib.import_module('lats'),importlib.import_module('webshop')


def execute_lats(task_id,author,website,tokenizer,transport,directory):
    import requests
    core,task_source=author
    logger=logging.getLogger(str(directory));logger.propagate=False;logger.setLevel(logging.INFO)
    handler=logging.FileHandler(directory/'search.log',encoding='utf-8');logger.addHandler(handler)
    log=SimpleNamespace(INFO=logging.INFO,info=logger.info,warning=logger.warning,debug=logger.debug,
                        basicConfig=lambda **kwargs:None)

    def gpt(prompt,model=None,temperature=None,max_tokens=100,n=1,stop=None):
        if temperature is not None and temperature!=transport.sampling['temperature']:
            raise ValueError('LATS effective temperature mismatch')
        prompt=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
        outputs=[]
        for _ in range(n):
            raw=transport.generate(prompt,max_tokens)['raw_output']
            if stop is not None:
                stops=[stop] if isinstance(stop,str) else stop
                offsets=[raw.index(s) for s in stops if s in raw]
                if offsets:raw=raw[:min(offsets)]
            outputs.append(raw)
        return outputs

    bound=bind_module(core,gpt=gpt,logging=log,WEBSHOP_URL=website)
    task_module=bind_module(task_source,gpt=gpt,gpt4=gpt,tokenizer=tokenizer,logging=log)
    task=task_module.WebShopTask()
    http=requests.Session();native_text=bound.webshop_text
    signature=inspect.signature(native_text)
    current_url=None
    def get(_native_url):
        response=http.get(current_url,timeout=1200)
        append(directory/'http.jsonl',dict(url=current_url,status=response.status_code))
        response.raise_for_status();return response
    bound.requests=SimpleNamespace(get=get)
    def render(*args,**kwargs):
        nonlocal current_url
        call=signature.bind(*args,**kwargs);call.apply_defaults()
        state=dict(call.arguments)
        state.update(state.pop('kwargs',{}))
        current_url=native_url(website,state)
        return native_text(*args,**kwargs)
    bound.webshop_text=render
    env=bound.webshopEnv();bound.env=env
    native_step=env.step;calls=[];terminals=[]
    def step(session,action):
        if session!='fixed_'+str(task_id):raise ValueError('LATS task identity changed')
        before=copy.deepcopy(env.sessions)
        try:receipt=native_step(session,action)
        except AssertionError:
            append(directory/'events.jsonl',dict(event='environment_invalid_action',action=action,before=before))
            raise
        calls.append(receipt)
        if receipt[2]:terminals.append(receipt[1])
        append(directory/'events.jsonl',dict(event='environment_step',action=action,before=before,
             after=copy.deepcopy(env.sessions),receipt=receipt))
        return receipt
    env.step=step
    nodes=[];NativeNode=bound.Node
    class RecordedNode(NativeNode):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);nodes.append(self)
    bound.Node=RecordedNode
    args=SimpleNamespace(backend='resident',temperature=transport.sampling['temperature'],
        n_generate_sample=5,n_evaluate_sample=1,prompt_sample='cot',log=str(directory/'search.log'))
    try:
        state,value,reward,em=bound.lats_search(args,task,'fixed_'+str(task_id),iterations=30,to_print=False)
        ids={id(node):i for i,node in enumerate(nodes)}
        tree=[dict(id=i,parent=None if n.parent is None else ids[id(n.parent)],state=n.state,
            children=[ids[id(c)] for c in n.children],depth=n.depth,visits=n.visits,value=n.value,
            reward=n.reward,is_terminal=n.is_terminal,env_state=n.env_state) for i,n in enumerate(nodes)]
        (directory/'search_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
        return dict(success=reward==1,reward=reward,selected_state=state,selected_value=value,
            terminal_rewards=terminals,environment_calls=len(calls),
            reward_policy='author best search-node reward',
            final_public_reward=calls[-1][1] if calls and calls[-1][2] else None,
            node_count=len(nodes),reflection_map=bound.reflection_map)
    finally:
        http.close();handler.close();logger.removeHandler(handler)
