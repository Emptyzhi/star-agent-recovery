"""Resolve startup CLI once and persist the queue's experiment defaults."""
import argparse
import json
from pathlib import Path
from model_backend import add_model_arguments, load_model_bundle
from resident_settings import add_capacity_arguments
from experiment_settings import (Experiment,Sampling,add_experiment_arguments,
                                 add_sampling_arguments,settings_from_args)


def build_parser():
    parser=argparse.ArgumentParser()
    add_model_arguments(parser)
    add_capacity_arguments(parser)
    add_experiment_arguments(parser)
    add_sampling_arguments(parser)
    parser.add_argument('--workers-per-gpu',type=int,default=40)
    parser.add_argument('--output',type=Path,required=True)
    return parser


def main():
    args=build_parser().parse_args()
    experiment,sampling=settings_from_args(args,Experiment),settings_from_args(args,Sampling)
    if not 1 <= args.workers_per_gpu <= args.max_num_seqs:
        raise ValueError('workers-per-gpu must fit max-num-seqs')
    if not 0 < args.prompt_limit <= args.context_limit-args.max_output_tokens:
        raise ValueError('prompt and output capacity exceed context')
    if experiment.required_output_cap()>args.max_output_tokens:
        raise ValueError('role output limits exceed --max-output-tokens')
    model,_,_=load_model_bundle(args.model_path,args.dtype)
    record=dict(model=model,experiment=experiment.as_dict(),sampling=sampling.as_dict(),
                workers_per_gpu=args.workers_per_gpu,context_limit=args.context_limit,
                prompt_limit=args.prompt_limit,max_num_seqs=args.max_num_seqs,
                max_output_tokens=args.max_output_tokens,gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens)
    args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    for value in (model['model_path'],args.dtype,experiment.environment_seed,args.max_num_seqs,
                  args.context_limit,args.prompt_limit,args.max_output_tokens,args.gpu_memory_utilization,
                  args.max_num_batched_tokens,'--do-sample' if sampling.do_sample else '--no-do-sample',
                  sampling.temperature,sampling.top_p,sampling.top_k,sampling.repetition_penalty):
        print(value)


if __name__=='__main__':
    main()
