"""One resident vLLM engine, continuous batching, unchanged STAR token framing."""
from collections import deque
from dataclasses import dataclass,field
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
import traceback
import types
from urllib.parse import urlparse,parse_qs
import uuid

from icr_command_trie import CommandTrie
from icr_parameter_language import ParameterLanguage,parameter_vocabulary,validate_spec
from icr_output_framing import STOP_MODES,stop_reason
from model_backend import encode_prompt,configure_tokenizer,load_model_bundle
from resident_settings import engine_options, build_parser
from experiment_settings import Sampling, settings_from_args
from star_vllm_protocol import LeaseBook,ProtocolViolation,validate_request
from star_vllm_framing import apply_boundary
from request_timing import RequestTiming

PACKAGE=Path(__file__).resolve().parent.parent


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


@dataclass
class Pending:
    request:dict
    input_ids:list
    request_id:str
    submitted:float
    mask:object=None
    ready:threading.Event=field(default_factory=threading.Event)
    response:dict|None=None
    error:str|None=None
    stop_reason:str|None=None
    started:float|None=None
    scheduled_steps:int=0
    max_scheduled_batch:int=0
    timing:RequestTiming|None=None


class StarProcessor:
    """Carry stop metadata; mask only the explicitly declared token language."""
    def __init__(self,pending):self.pending=pending
    def __call__(self,prompt_ids,output_ids,logits):
        if self.pending.mask is None:return logits
        import torch
        allowed=self.pending.mask.allowed_next(output_ids)
        keep=torch.tensor(allowed,device=logits.device,dtype=torch.long)
        masked=torch.full_like(logits,float('-inf'));masked[keep]=logits[keep]
        return masked


class Service:
    def __init__(self):
        import torch,vllm
        from vllm import EngineArgs,LLMEngine
        from vllm.sequence import SequenceStatus
        assert vllm.__version__=='0.6.6'
        assert socket.gethostname()==NODE and os.environ['SLURM_JOB_ID']==ARGS.parent_job_id
        assert os.environ['CUDA_VISIBLE_DEVICES']==str(GPU)
        active = subprocess.check_output([
            'nvidia-smi', '--id='+str(GPU), '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError('allocated GPU already has compute processes; reuse or stop its exact old service')
        self.torch=torch
        self.backend, _, eos_ids = load_model_bundle(ARGS.model_path, ARGS.dtype)
        self.default_sampling = settings_from_args(ARGS, Sampling)
        self.seed=secrets.randbits(32)
        gpu_row = subprocess.check_output([
            'nvidia-smi', '--id='+str(GPU), '--query-gpu=uuid,name',
            '--format=csv,noheader'], text=True).strip().split(', ', 1)
        if len(gpu_row) != 2 or not gpu_row[0].startswith('GPU-'):
            raise RuntimeError('cannot verify allocated GPU identity')
        self.identity=dict(parent_job_id=os.environ['SLURM_JOB_ID'],slurm_step_id=os.environ['SLURM_STEP_ID'],
            hostname=socket.gethostname(),cuda_visible_devices=str(GPU),gpu_uuid=gpu_row[0],
            gpu_name=gpu_row[1])
        self.instance=uuid.uuid4().hex;self.started=time.time()
        self.condition=threading.Condition();self.tokenizer_lock=threading.RLock()
        self.book=LeaseBook(self.seed);self.waiting=deque();self.inflight={}
        self.slots=threading.BoundedSemaphore(MAX_SEQS);self.status='LOADING';self.error=None
        self.completed=0;self.step_count=0;self.max_scheduled=0
        self.events=(STATE/'engine_events.jsonl').open('x',buffering=1)
        save(STATE/'engine_config.json',dict(vllm_version=vllm.__version__,torch_version=torch.__version__,
            engine_seed=self.seed,request_seed=None,model_backend=self.backend,max_num_seqs=MAX_SEQS,
            max_model_len=CONTEXT,max_prompt_tokens=PROMPT_CAP,gpu_memory_utilization=ARGS.gpu_memory_utilization,
            max_output_tokens=ARGS.max_output_tokens, sampling=self.default_sampling.as_dict(),
            chunked_prefill=True,max_num_batched_tokens=ARGS.max_num_batched_tokens,enforce_eager=True,async_output=False,
            speculative_decoding=False,model_replicas=1,**self.identity))
        args=EngineArgs(**engine_options(self.backend, context_limit=CONTEXT,
                                        max_num_seqs=MAX_SEQS, seed=self.seed,
                                        gpu_memory_utilization=ARGS.gpu_memory_utilization,
                                        max_num_batched_tokens=ARGS.max_num_batched_tokens))
        self.engine=LLMEngine.from_engine_args(args)
        self.tokenizer=self.engine.get_tokenizer()
        configure_tokenizer(self.tokenizer,self.backend)
        self.eos_ids=list(eos_ids)
        assert all(type(x) is int for x in self.eos_ids) and self.tokenizer.eos_token_id in self.eos_ids
        # Parameter-language masking is a WebShop command-language capability.
        # Do not make unrelated datasets fail at startup when a model tokenizer
        # is valid for ordinary generation but is not byte-level.
        try:
            self.parameter_vocab=parameter_vocabulary(self.tokenizer)
        except ValueError:
            self.parameter_vocab=None
        checker=self.engine.output_processor.stop_checker
        original=checker.maybe_stop_sequence
        def check(instance,seq,new_char_count,sampling_params,lora_req=None):
            original(seq,new_char_count,sampling_params,lora_req)
            processors=[x for x in sampling_params.logits_processors if isinstance(x,StarProcessor)]
            if len(processors)!=1:raise RuntimeError('missing unique STAR request descriptor')
            pending=processors[0].pending
            with self.tokenizer_lock:
                apply_boundary(seq,pending,lambda ids:self.tokenizer.decode(ids,skip_special_tokens=True,
                    clean_up_tokenization_spaces=False),self.eos_ids,time.perf_counter(),SequenceStatus.FINISHED_STOPPED)
        checker.maybe_stop_sequence=types.MethodType(check,checker)
        assert len(self.engine.scheduler)==1
        scheduler=self.engine.scheduler[0];schedule=scheduler.schedule
        def tracked_schedule(instance):
            metadata,output,allow_async=schedule()
            if allow_async:raise RuntimeError('asynchronous output processing violates token-boundary contract')
            ids=[m.request_id for m in metadata]
            self.scheduled_this_step=ids
            self.max_scheduled=max(self.max_scheduled,len(ids))
            for rid in ids:
                p=self.inflight[rid]
                if p.started is None:p.started=self.step_started
                p.scheduled_steps+=1;p.max_scheduled_batch=max(p.max_scheduled_batch,len(ids))
            self.events.write(json.dumps(dict(event='engine_schedule',time=time.time(),request_ids=ids,
                scheduled_sequences=len(ids),batched_tokens=output.num_batched_tokens))+'\n')
            return metadata,output,allow_async
        scheduler.schedule=types.MethodType(tracked_schedule,scheduler)
        self.runtime=dict(service_pid=os.getpid(),service_instance_id=self.instance,service_start_time=self.started,
            engine='vllm',engine_version=vllm.__version__,torch_version=torch.__version__,engine_seed=self.seed,
            model_path=self.backend['model_path'],model_class=self.backend['model_class'],
            model_type=self.backend['model_type'],torch_dtype=self.backend['torch_dtype'],
            model_backend=self.backend,model_replicas=1,loaded_once=True,model_context_tokens=CONTEXT,
            rng_scheme='vllm engine RNG; no request reseeding',per_task_reseed=False,per_request_reseed=False,
            chat_template_applied=False,prompt_truncation=False,stop_boundary='original STAR predicate after each token',
            bitwise_hf_equivalence=False,**self.identity)
        self.status='READY';save(STATE/'health.json',self.health())
        self.worker=threading.Thread(target=self.loop,name='vllm-continuous-batcher',daemon=True);self.worker.start()

    def decoding(self,limit,sampling=None):
        selected = self.default_sampling if sampling is None else sampling
        return dict(**selected.as_dict(),effective_temperature=selected.engine_parameters()['temperature'],
            max_new_tokens=limit,num_beams=1,request_seed=None,engine_seed=self.seed,engine='vllm',
            min_tokens=0,ignore_eos=False,eos_token_id=self.eos_ids)

    def health(self,experiment=None):
        with self.condition:
            return dict(self.runtime,status=self.status,fatal_error=self.error,max_num_seqs=MAX_SEQS,
                max_prompt_tokens=PROMPT_CAP,max_new_tokens=ARGS.max_output_tokens,max_forward_batch_size=MAX_SEQS,
                request_sampling_supported=True, flexible_output_limits=True,
                text_generation_supported=True,normal_route_preserved=True,command_trie_supported=True,
            parameter_language_supported=self.parameter_vocab is not None,stop_modes=list(STOP_MODES),request_deadlines=True,
                sampling_run=None if experiment is None else self.book.metadata(experiment),
                queued_requests=len(self.waiting),active_requests=len(self.inflight),completed_requests=self.completed,
                max_scheduled_sequences=self.max_scheduled,decoding=self.decoding(ARGS.max_output_tokens))

    def control(self,request,end=False):
        if type(request) is not dict or set(request)!={'experiment_id'}:raise ProtocolViolation('invalid lease control')
        if self.status!='READY':raise ProtocolViolation('service is not healthy',503)
        return self.book.end(request['experiment_id']) if end else self.book.begin(request['experiment_id'])

    def generate(self,request):
        validate_request(request, ARGS.max_output_tokens)
        if self.status!='READY':raise ProtocolViolation('service is not healthy',503)
        preprocessing_started=time.perf_counter()
        with self.tokenizer_lock:
            preprocessing_queue=time.perf_counter()-preprocessing_started
            ids=encode_prompt(self.tokenizer,request['prompt'])['input_ids'];mask=None
            if len(ids)>PROMPT_CAP or len(ids)+request['max_new_tokens']>CONTEXT:
                raise ProtocolViolation('complete prompt exceeds the declared context limit',413)
            if request['route']=='command_trie':
                sequences=tuple(tuple(self.tokenizer.encode(c,add_special_tokens=False)) for c in request['allowed_commands'])
                for command,tokens in zip(request['allowed_commands'],sequences):
                    if (not tokens or len(tokens)+1>request['max_new_tokens']
                            or any(t in self.tokenizer.all_special_ids for t in tokens)
                            or self.tokenizer.decode(tokens,clean_up_tokenization_spaces=False)!=command):
                        raise ProtocolViolation('command encoding does not preserve the declared language')
                mask=CommandTrie(sequences,self.tokenizer.eos_token_id)
            if 'parameter_language' in request:
                if self.parameter_vocab is None:
                    raise ValueError('parameter-language masking is unavailable for this tokenizer')
                validate_spec(request['parameter_language'])
                mask=ParameterLanguage(request['parameter_language'],self.parameter_vocab,
                    lambda s:self.tokenizer.encode(s,add_special_tokens=False),self.tokenizer.eos_token_id)
        submitted=time.perf_counter()
        if not self.slots.acquire(blocking=False):raise ProtocolViolation('all request slots are occupied',429)
        rid=uuid.uuid4().hex;acquired=False;completed=False
        try:
            self.book.acquire(request['experiment_id'],rid);acquired=True
            pending=Pending(dict(request),ids,rid,submitted,mask=mask,
                timing=RequestTiming(submitted,preprocessing_queue_seconds=preprocessing_queue))
            with self.condition:
                if self.status!='READY':raise ProtocolViolation('service is not healthy',503)
                self.waiting.append(pending);self.condition.notify()
            pending.ready.wait()
            if pending.error is not None:raise RuntimeError(pending.error)
            if pending.response is None:raise RuntimeError('request completed without a response')
            completed=True;return pending.response
        finally:
            if acquired:self.book.release(request['experiment_id'],rid,completed)
            self.slots.release()

    def loop(self):
        from vllm import SamplingParams
        try:
            while True:
                with self.condition:
                    if not self.inflight:self.condition.wait_for(lambda:bool(self.waiting))
                    new=list(self.waiting);self.waiting.clear()
                    for p in new:self.inflight[p.request_id]=p
                for p in new:
                    self.events.write(json.dumps(dict(event='request_enqueued',request_id=p.request_id,
                        experiment_id=p.request['experiment_id'],submitted_monotonic=p.submitted,
                        preprocessing_queue_seconds=p.timing.preprocessing_queue_seconds))+'\n')
                    sampling = Sampling(**p.request['sampling']) if 'sampling' in p.request else self.default_sampling
                    params=SamplingParams(n=1,**sampling.engine_parameters(),
                        presence_penalty=0,frequency_penalty=0,min_p=0,seed=None,
                        max_tokens=p.request['max_new_tokens'],min_tokens=0,ignore_eos=False,
                        stop_token_ids=self.eos_ids,stop=None,detokenize=False,logprobs=0,
                        logits_processors=[StarProcessor(p)])
                    self.engine.add_request(p.request_id,{'prompt_token_ids':p.input_ids},params)
                self.scheduled_this_step=[];self.step_started=time.perf_counter()
                outputs=self.engine.step();step_ended=time.perf_counter();self.step_count+=1
                for rid in self.scheduled_this_step:
                    self.inflight[rid].timing.scheduled_step(self.step_started,step_ended)
                self.events.write(json.dumps(dict(event='engine_step_completed',step=self.step_count,
                    started_monotonic=self.step_started,ended_monotonic=step_ended,
                    request_ids=self.scheduled_this_step))+'\n')
                for out in outputs:
                    if not out.finished:continue
                    p=self.inflight[out.request_id]
                    response=self.response(p,out)
                    with self.condition:
                        del self.inflight[p.request_id];self.completed+=1
                        p.response=response;p.ready.set()
        except Exception as error:
            traceback.print_exc()
            with self.condition:
                self.error=type(error).__name__+': '+str(error);self.status='FAILED'
                save(STATE/'engine_error.json',dict(error=self.error,traceback=traceback.format_exc()))
                for p in [*self.waiting,*self.inflight.values()]:p.error=self.error;p.ready.set()
                self.waiting.clear();self.condition.notify_all()

    def response(self,p,out):
        if len(out.outputs)!=1 or list(out.prompt_token_ids)!=p.input_ids:raise RuntimeError('vLLM changed request token identity')
        result=out.outputs[0];tokens=list(result.token_ids)
        if not tokens or len(tokens)>p.request['max_new_tokens']:raise RuntimeError('invalid generated token count')
        if result.logprobs is None or len(result.logprobs)!=len(tokens):raise RuntimeError('missing chosen-token probabilities')
        probabilities=[math.exp(probs[token].logprob) for token,probs in zip(tokens,result.logprobs)]
        with self.tokenizer_lock:text=self.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        reason=p.stop_reason
        if reason=='eos' and len(tokens)==p.request['max_new_tokens']:reason='eos_at_limit'
        if reason is None:
            if len(tokens)!=p.request['max_new_tokens']:raise RuntimeError('generation stopped without a declared boundary')
            reason='length'
        sampling = Sampling(**p.request['sampling']) if 'sampling' in p.request else self.default_sampling
        route=p.request['route'];decoding=self.decoding(p.request['max_new_tokens'],sampling)
        decoding.update(command_language_size=len(p.request['allowed_commands']) if route=='command_trie' else 0,
            command_constraint='token-prefix mask before top-k sampling' if route=='command_trie' else None,
            probability_semantics='vllm 0.6.6 processed logprobs after temperature/top-k/top-p')
        response=dict(route=route,text=text,raw_output=text,input_tokens=len(p.input_ids),output_tokens=len(tokens),
            output_token_ids=tokens,selected_probabilities=probabilities,average_prob=sum(probabilities)/len(probabilities),
            task_seed=None,call_index=0,effective_seed=None,sampling_run=self.book.metadata(p.request['experiment_id']),
            request_id=p.request_id,finish_reason=reason,final_token_id=tokens[-1],stop_mode=p.request.get('stop_mode'),
            budget_seconds=p.request.get('budget_seconds'),parameter_language=p.request.get('parameter_language'),
            output_token_limit=p.request['max_new_tokens'],decoding=decoding,model_path=self.backend['model_path'],
            gpu_identity=self.identity,runtime=self.runtime,
            usage=dict(prompt_tokens=len(p.input_ids),completion_tokens=len(tokens),total_tokens=len(p.input_ids)+len(tokens),
                max_scheduled_batch=p.max_scheduled_batch,scheduled_steps=p.scheduled_steps,max_num_seqs=MAX_SEQS),
            wall_time_seconds=time.perf_counter()-p.submitted-p.timing.engine_queue_seconds,
            queue_time_seconds=p.timing.queue_seconds,
            engine_queue_seconds=p.timing.engine_queue_seconds,
            preprocessing_queue_seconds=p.timing.preprocessing_queue_seconds,
            model_active_step_seconds=p.timing.active_step_seconds,
            timing_policy='all engine scheduling waits plus tokenizer-lock wait; shared-step latency separately recorded')
        self.events.write(json.dumps(dict(event='request_completed',time=time.time(),request_id=p.request_id,
            experiment_id=p.request['experiment_id'],input_tokens=len(p.input_ids),output_tokens=len(tokens),
            finish_reason=reason,max_scheduled_batch=p.max_scheduled_batch,
            queue_time_seconds=p.timing.queue_seconds,model_active_step_seconds=p.timing.active_step_seconds))+'\n')
        return response


class Handler(BaseHTTPRequestHandler):
    service:Service
    def reply(self,status,value):
        body=json.dumps(value,allow_nan=False).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path)
        if path.path!='/health':return self.reply(404,dict(error='unknown endpoint'))
        try:
            params=parse_qs(path.query);experiment=params['experiment_id'][0] if 'experiment_id' in params else None
            self.reply(200,self.service.health(experiment))
        except ProtocolViolation as e:self.reply(e.status,dict(error=str(e)))
    def do_POST(self):
        try:
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path=='/generate_text':result=self.service.generate(request)
            elif self.path=='/begin_run':result=self.service.control(request)
            elif self.path=='/end_run':result=self.service.control(request,end=True)
            else:raise ProtocolViolation('unknown endpoint',404)
            self.reply(200,result)
        except (json.JSONDecodeError,ProtocolViolation) as e:self.reply(e.status if isinstance(e,ProtocolViolation) else 400,dict(error=str(e)))
        except Exception as e:
            traceback.print_exc();self.reply(500,dict(error=type(e).__name__+': '+str(e)))
    def log_message(self,*args):pass


class Server(ThreadingHTTPServer):
    request_queue_size=128
    daemon_threads=True


def main(args):
    global ARGS, STATE, NODE, GPU, MAX_SEQS, PROMPT_CAP, CONTEXT, PORT
    ARGS=args
    STATE=Path(args.state)
    NODE=args.node
    GPU=args.gpu_index
    MAX_SEQS=args.max_num_seqs
    PROMPT_CAP=args.prompt_limit
    CONTEXT=args.context_limit
    PORT=args.port
    if (socket.gethostname() != NODE or os.environ.get('SLURM_JOB_ID') != args.parent_job_id
            or not os.environ.get('SLURM_STEP_ID', '').isdecimal()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(GPU)):
        raise RuntimeError('model startup requires the declared Slurm compute step and GPU')
    if (GPU < 0 or MAX_SEQS < 1 or not 1 <= PORT <= 65535
            or args.max_output_tokens < 1 or not 0 < PROMPT_CAP <= CONTEXT-args.max_output_tokens
            or not 0 < args.gpu_memory_utilization <= 1 or args.max_num_batched_tokens < 1):
        raise ValueError('invalid service capacity or context settings')
    STATE.mkdir(parents=True, exist_ok=False)
    Handler.service=Service()
    with Server(('127.0.0.1',PORT),Handler) as server:
        save(STATE/'READY.json',Handler.service.health())
        server.serve_forever()


if __name__=='__main__':
    main(build_parser().parse_args())
