"""Run independent author tasks against one existing resident GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import socket
import time
import traceback

from resident_transport import ResidentTransport, ROLE_STOP_MODES
from experiment_contracts import required_output_cap


def save(path, value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)


def run_task(task_id, req, args, template, authors, tokenizer, health, execute, submitted_at):
    directory = args.output_dir/('task_%03d'%task_id)
    directory.mkdir()
    started = time.perf_counter()
    admission_wait = started-submitted_at
    transport = ResidentTransport(req['endpoints'][args.gpu_index],args.lease,req['sampling'],
                                  health,directory)
    identity = dict(task_id=task_id,run_id=req['run_id'],method=req['method'],dataset=req['dataset'],
                    model_path=args.model_path,parent_job_id=args.parent_job_id,node=args.node,
                    step_id=os.environ['SLURM_STEP_ID'],gpu_uuid=health['gpu_uuid'],pid=health['service_pid'])
    save(directory/'identity.json',identity)
    try:
        result = execute(req['method'],task_id,template,authors,tokenizer,transport,directory,req['mode'])
        cost = transport.costs(time.perf_counter()-started)
        cost['task_admission_wait_seconds'] = admission_wait
        summary = dict(complete=True,identity=identity,success=result['success'],
            benchmark_reward=result.get('reward',int(result['success'])),
            final_public_reward=result.get('final_public_reward',result.get('reward',int(result['success']))),
            status='SUCCESS' if result['success'] else 'METHOD_EXHAUSTED',
            cost=cost,method_result=result)
        save(directory/'summary.json',summary)
        print(json.dumps(dict(event='task_complete',task=task_id,success=result['success'])),flush=True)
        return summary
    except Exception as error:
        save(directory/'fatal.json',dict(identity=identity,error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),complete=False,completed_model_records=transport.records,
            elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        transport.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model-path','--local_llm_path',dest='model_path',required=True)
    p.add_argument('--parent-job-id',required=True)
    p.add_argument('--node',required=True)
    p.add_argument('--gpu-index',type=int,choices=(0,1),required=True)
    args = p.parse_args()
    req = json.loads(args.request.read_text())
    req['method'], req['mode'] = req['experiment']['method'], req['experiment']['mode']
    assert req['dataset'] in ('game24','webshop','alfworld') and req['method'] in ('react','reflexion','ga_rollback','tot','lats')
    assert os.environ['SLURM_JOB_ID']==args.parent_job_id and socket.gethostname()==args.node
    assert os.environ['SLURM_STEP_ID'].isdecimal()
    assert req['environment_seed']==1 and req['workers_per_gpu']==40
    assert req['mode'] in ('act','react')
    endpoint = req['endpoints'][args.gpu_index]
    args.lease = req['run_id']+'_g'+str(args.gpu_index)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    import requests
    health_response=requests.get(endpoint+'/health',timeout=30)
    health_response.raise_for_status()
    health=health_response.json()
    assert health['status']=='READY' and health['fatal_error'] is None and health['engine']=='vllm'
    assert health['model_path']==args.model_path and health['max_num_seqs']==40
    assert 'first_line' in health['stop_modes'] and health['chat_template_applied'] is False
    cap = required_output_cap(req['dataset'], req['method'])
    assert health['max_new_tokens']>=cap
    if req['dataset']=='game24':
        from game24_methods import load_authors,execute
        authors=load_authors()
        if req['method']=='tot':
            from tot_game24 import load_tot
            authors['tot']=load_tot(req['csv'])
        template=authors['react'].Game24Task(req['csv'],req['environment_seed'],500)
    elif req['dataset']=='webshop':
        from webshop_methods import load_authors,execute
        authors=load_authors();template=dict(website=req['website'])
        if req['method']=='lats':
            from lats_webshop import load_lats
            authors['lats']=load_lats()
    else:
        from alfworld_methods import load_authors,load_inputs,execute
        authors=load_authors();template=load_inputs(req)
    adapted=(req['method']=='tot' and req['dataset']!='game24') or (req['method']=='lats' and req['dataset']!='webshop')
    if adapted:
        if req['method']=='tot':
            from tot_game24 import load_tot_core
            authors['tot']=load_tot_core()
        else:
            from lats_webshop import load_lats
            authors['lats']=load_lats()
        from functools import partial
        from search_adaptations import execute as execute_adaptation
        execute=partial(execute_adaptation,req['dataset'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=False)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    tasks=req['task_ids'][args.gpu_index::2]
    save(args.output_dir/'run_config.json',dict(request=req,health=health,tasks=tasks,
        dataset_adaptation=adapted,
        author_revision='f6fb280d6c5a03a515e5a7db213585ccd9b0f068',model=args.model_path,
        stop_modes=ROLE_STOP_MODES,postprocessing='unchanged author parser',
        generation_policy='action/candidate first-line during decoding; native reflection/analysis/evaluation',
        environment_seed_scope='once per initialized group'))
    response=requests.post(endpoint+'/begin_run',json=dict(experiment_id=args.lease),timeout=30)
    response.raise_for_status()
    failed=[];results=[]
    try:
        with ThreadPoolExecutor(max_workers=req['workers_per_gpu']) as pool:
            futures={pool.submit(run_task,t,req,args,template,authors,tokenizer,health,execute,time.perf_counter()):t for t in tasks}
            for future in as_completed(futures):
                try: results.append(future.result())
                except Exception as error: failed.append(dict(task=futures[future],error=repr(error)))
        save(args.output_dir/'run_done.json',dict(complete=not failed,tasks=tasks,completed=len(results),
             successes=sum(x['success'] for x in results),failures=failed))
    finally:
        response=requests.post(endpoint+'/end_run',json=dict(experiment_id=args.lease),timeout=30)
        response.raise_for_status()
    return int(bool(failed))


if __name__=='__main__':
    raise SystemExit(main())
