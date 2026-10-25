"""Model-independent inference settings shared by startup and engine creation."""
import argparse

from model_backend import add_model_arguments
from experiment_settings import add_sampling_arguments


def build_parser():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    add_sampling_arguments(parser)
    parser.add_argument('--state', required=True)
    parser.add_argument('--parent-job-id', required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--gpu-index', required=True, type=int)
    parser.add_argument('--port', required=True, type=int)
    add_capacity_arguments(parser)
    return parser


def add_capacity_arguments(parser):
    parser.add_argument('--max-num-seqs', type=int, default=40)
    parser.add_argument('--context-limit', type=int, default=32768)
    parser.add_argument('--prompt-limit', type=int, default=32000)
    parser.add_argument('--max-output-tokens', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.95)
    parser.add_argument('--max-num-batched-tokens',type=int,default=2048)


def engine_options(model, *, context_limit, max_num_seqs, seed,
                   gpu_memory_utilization=.95, max_num_batched_tokens=2048):
    return dict(model=model['model_path'], tokenizer=model['model_path'],
                dtype=model['torch_dtype'], tensor_parallel_size=1,
                max_model_len=context_limit, max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_utilization, enable_chunked_prefill=True,
                max_num_batched_tokens=max_num_batched_tokens, enforce_eager=True,
                disable_async_output_proc=True, num_scheduler_steps=1,
                enable_prefix_caching=False, seed=seed, trust_remote_code=False,
                generation_config='vllm', guided_decoding_backend='outlines',
                disable_log_stats=True)
