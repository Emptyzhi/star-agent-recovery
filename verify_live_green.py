"""Probe the actual Green oracle/HTTP boundary on a finite WebShop page."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import requests

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from bound_resident_client import BoundResidentClient
from icr_core import Ledger,ProposalOracle,NodeAddress,Phase,SupportKind
from model_backend import load_model_bundle,encode_prompt
from web_v16_episode import WebV16Episode
from web_v16_run import task_limits
from experiment_settings import Sampling


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--node',required=True)
    parser.add_argument('--parent',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert socket.gethostname()==args.node and os.environ['SLURM_JOB_ID']==args.parent
    assert os.environ['SLURM_STEP_ID'].isdecimal()
    model,tokenizer,_=load_model_bundle(args.model,'bfloat16')
    args.out.mkdir(parents=True,exist_ok=False)
    evidence=[]
    for gpu in range(2):
        response=requests.get(f'http://127.0.0.1:{41581+gpu}/health',timeout=30)
        response.raise_for_status();health=response.json()
        binding=dict(parent_job_id=args.parent,step_id=health['slurm_step_id'],hostname=args.node,
            gpu=gpu,gpu_uuid=health['gpu_uuid'],pid=health['service_pid'],port=41581+gpu,
            instance=health['service_instance_id'])
        client=BoundResidentClient(binding,f'www_star_green_probe_{args.node}_{gpu}',model,sampling=Sampling())
        episode=WebV16Episode('http://127.0.0.1:3001',f'www_star_probe_{args.node}_{gpu}_fixed_0',
            0,'',lambda messages:'')
        events=[]
        try:
            client.begin_run()
            episode.reset();episode.step('search[blue cup]')
            assert episode.support().kind is SupportKind.FINITE
            assert episode.ordinary_command_constrained is False
            # The probe tests framing, not whether the model follows a copy instruction.
            prompt=('Write two lines:\nsearch[blue cup]\nObservation: probe only\n')
            episode.prompt=lambda memory:prompt
            oracle=ProposalOracle(client.generate,master_seed=1,
                emit=lambda event,**data:events.append(dict(event=event,**data)),
                token_counter=lambda text:len(encode_prompt(tokenizer,text)['input_ids']),
                context_limit=32768,prompt_limit=32000,request_namespace=f'green_probe_{args.node}_{gpu}',
                generation_versions=health)
            address=NodeAddress(0,0,Phase.SOURCE,0,(),0)
            oracle.ordinary(episode,address,(),Ledger(task_limits()))
            generation=list(oracle.generations.values())[-1]
            request=generation.request;raw=generation.raw_response
            assert request.route=='normal' and request.commands==() and request.stop_mode=='first_line'
            assert raw['route']=='normal' and raw['stop_mode']=='first_line' and raw['finish_reason']=='frame'
            ids=raw['output_token_ids']
            boundaries=[i for i in range(1,len(ids)+1) if '\n' in tokenizer.decode(ids[:i],
                skip_special_tokens=True,clean_up_tokenization_spaces=False)]
            assert boundaries and boundaries[0]==len(ids)
            row=dict(gpu=gpu,finite_page_commands=list(episode.support().commands),
                raw_response=raw,events=events,green_command_trie=False)
            (args.out/f'gpu{gpu}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            evidence.append(dict(gpu=gpu,route=raw['route'],stop_mode=raw['stop_mode'],
                output_tokens=len(ids),first_newline_token_count=boundaries[0],identity=raw['gpu_identity']))
        finally:
            episode.close();client.close()
    (args.out/'PASS.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('GREEN_FREE_FIRSTLINE_VERIFIED_BOTH_GPUS',flush=True)


if __name__=='__main__':main()
