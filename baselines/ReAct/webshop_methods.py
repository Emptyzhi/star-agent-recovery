"""Author WebShop controllers with recorded, encoded HTTP environment transport."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType,SimpleNamespace
from urllib.parse import quote

from author_binding import bind_module
from resident_transport import PipelineTransport,GARollbackTransport,append

PACKAGE=Path(__file__).resolve().parent


def load_authors():
    folder=PACKAGE/'vendor/ga/test_webshop'
    os.chdir(folder);sys.path.insert(0,str(folder))
    return {key:importlib.import_module(name) for key,name in
            [('react','web_react'),('reflexion','web_reflexion'),('ga_rollback','web_rollback')]}


def native_url(website,state):
    page=state['page_type'];sid=quote(state['session'],safe='')
    query=quote(quote(state.get('query_string',''),safe=''),safe='')
    number=quote(str(state.get('page_num',1)),safe='')
    asin=quote(state.get('asin',''),safe='');options=quote(str(state.get('options',{})),safe='')
    routes={'init':f'{sid}','search':f'search_results/{sid}/{query}/{number}',
            'item':f'item_page/{sid}/{asin}/{query}/{number}/{options}',
            'item_sub':f'item_sub_page/{sid}/{asin}/{query}/{number}/{quote(state.get("subpage",""),safe="")}/{options}',
            'end':f'done/{sid}/{asin}/{options}'}
    return website.rstrip('/')+'/'+routes[page]


class RecordedEnvironment:
    def __init__(self,native_class,website,task_id,directory,transport):
        import requests
        self.raw=native_class(website)
        self.website,self.session,self.directory,self.transport=website,'fixed_'+str(task_id),directory,transport
        self.http=requests.Session();self.calls=0;self.terminal_rewards=[];self.last_receipt=None
        native_text=self.raw.webshop_text

        def render(**state):
            url=native_url(website,state)
            def get(_author_url):
                started=time.perf_counter()
                response=self.http.get(url,timeout=1200)
                append(directory/'http.jsonl',dict(url=url,status=response.status_code,
                    elapsed_seconds=time.perf_counter()-started))
                response.raise_for_status()
                return response
            namespace=dict(native_text.__func__.__globals__,requests=SimpleNamespace(get=get))
            bound=FunctionType(native_text.__func__.__code__,namespace,native_text.__func__.__name__,native_text.__func__.__defaults__)
            return bound(self.raw,**state)
        self.raw.webshop_text=render

    def step(self,session,action):
        # run_trial enumerates one task as fixed_0; map it to its immutable dataset ID.
        if session not in ('fixed_0',self.session):raise ValueError('author selected another task')
        before=copy.deepcopy(self.raw.sessions)
        self.calls+=1
        try:receipt=self.raw.step(self.session,action)
        except AssertionError:
            append(self.directory/'events.jsonl',dict(event='environment_invalid_action',action=action,
                trial=self.transport.trial,before=before,after=copy.deepcopy(self.raw.sessions)))
            raise
        if receipt[2]:self.terminal_rewards.append(receipt[1])
        self.last_receipt=receipt
        append(self.directory/'events.jsonl',dict(event='environment_step',time=time.time(),action=action,
            trial=self.transport.trial,before=before,after=copy.deepcopy(self.raw.sessions),receipt=receipt))
        return receipt

    def close(self):self.http.close()


def execute(method,task_id,template,authors,tokenizer,transport,directory,mode='act'):
    if method=='lats':
        from lats_webshop import execute_lats
        return execute_lats(task_id,authors['lats'],template['website'],tokenizer,transport,directory)
    author=authors[method];website=template['website'];environments=[]
    def factory(url):
        if url!=website:raise ValueError('unexpected website endpoint')
        env=RecordedEnvironment(author.webshopEnv,website,task_id,directory,transport)
        environments.append(env);return env
    try:
        if method=='reflexion':
            bound=bind_module(author,webshopEnv=factory)
            model=author.Local_llm.__new__(author.Local_llm)
            model.tokenizer=tokenizer;model.generation_pipe=PipelineTransport(transport)
            configs=[dict(name='env_'+str(task_id),memory=[],is_success=False,skip=False,reward=0)]
            trials=[]
            for trial in range(2):
                transport.trial,transport.role=trial,'actor'
                path=directory/('trial_%d.log'%trial)
                configs=bound.run_trial(str(path),str(directory/'world.log'),trial,configs,True,model,website)
                trials.append(copy.deepcopy(configs[0]))
                transport.role='reflection';configs=bound.update_memory(str(path),configs,model)
                (directory/('trial_%d.json'%trial)).write_text(json.dumps(configs,indent=2),encoding='utf-8')
            reward=configs[0]['reward']
            result=dict(success=bool(configs[0]['is_success']),reward=reward,
                reward_policy='author best reward across two trials',trials=trials,memory=configs[0]['memory'])
        else:
            env=factory(website)
            if method=='react':
                reward=author.webshop_run(env.session,author.prompt1,env,tokenizer,PipelineTransport(transport),to_print=False)
                result=dict(success=reward==1,reward=reward)
            elif method=='ga_rollback':
                model=GARollbackTransport(transport)
                reward,rollbacks=author.webshop_run(env.session,env,author.BASE_PROMPT,model,model,False,to_print=False,max_rollback_num=6)
                result=dict(success=reward==1,reward=reward,rollbacks=rollbacks)
            else:raise ValueError('unknown method')
        result['terminal_rewards']=[v for env in environments for v in env.terminal_rewards]
        result['environment_calls']=sum(env.calls for env in environments)
        used=[env for env in environments if env.last_receipt is not None]
        if not used:raise ValueError('author produced no environment receipt')
        last=used[-1].last_receipt
        result['final_public_reward']=last[1] if last[2] else None
        return result
    finally:
        for env in environments:env.close()
