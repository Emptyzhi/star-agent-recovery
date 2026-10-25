from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from .native_protocol import (render_native_messages, render_plain_continuation,
    recovery_command_messages,decision_chat_messages, RUNTIME_PREFIX)
from .candidate_context import project_candidate_history
from .framing import render_sectioned_continuation,one_line_recovery
from urllib.parse import urlparse
from urllib.request import Request,urlopen,build_opener,ProxyHandler


class ModelContextLimit(ValueError):
    """The complete request cannot fit; no model request was submitted."""


class ModelDeadlineExpired(TimeoutError):
    """Prompt preparation exhausted the budget before model submission."""


def resident_binding_step(args):
    step=args.step_id if args.resident_step_id is None else args.resident_step_id
    if type(step) is not str or not step.isdecimal():
        raise ValueError('resident step must be an explicit numeric Slurm step')
    return step


def resident_chat_messages(messages, model_type):
    if model_type != 'mistral':
        return messages
    # Mistral v0.3 permits one initial system block and alternating dialogue.
    systems, dialogue = [], []
    for message in messages:
        if set(message) != {'role','content'} or type(message['content']) is not str:
            raise ValueError('Mistral text transport requires literal role/content messages')
        role,content = message['role'],message['content']
        if role=='system':
            systems.append(content)
        elif role in ('user','assistant'):
            if dialogue and dialogue[-1]['role']==role:
                if role!='user':
                    raise ValueError('consecutive assistant turns are not part of this protocol')
                dialogue[-1]['content'] += '\n\n'+content
            else:
                dialogue.append(dict(role=role,content=content))
        else:
            raise ValueError('unsupported role in Mistral text transport')
    if not dialogue or dialogue[0]['role']!='user' or dialogue[-1]['role']!='user':
        raise ValueError('Mistral generation must start and end with a user turn')
    return ([dict(role='system',content='\n\n'.join(systems))] if systems else [])+dialogue


def render_resident_request(messages, purpose, tokenizer, model_type,history_format='flat',prompt_format='native-chat',candidate_history='full',interface_format='native',recovery_prompt_policy='full',text_history_format='sectioned'):
    if purpose not in ('action','recovery'):
        raise ValueError('unknown generation purpose')
    if candidate_history == 'once':
        if prompt_format != 'plain' or history_format != 'dialogue':
            raise ValueError('candidate references require plain recorded dialogue')
        messages, _ = project_candidate_history(messages)
    elif candidate_history != 'full':
        raise ValueError('unknown candidate history presentation')
    if prompt_format == 'plain':
        if interface_format not in ('native','sectioned','json-tools'):raise ValueError('unknown interface format')
        if interface_format=='json-tools':
            from .json_tools import actor_only_request
            actor_only=actor_only_request(messages)
            return render_sectioned_continuation(messages,history_format,json_tools=True,actor_only=actor_only,
                recovery_prompt_policy=recovery_prompt_policy), None, (
                'native_act_only_plain_actor_only' if actor_only else 'native_act_only_plain_json_tool_call')
        if interface_format=='sectioned':
            if text_history_format=='native':
                from .native_history import render_native_history
                prompt=render_native_history(messages,recovery_prompt_policy)
            elif text_history_format=='sectioned':
                prompt=render_sectioned_continuation(messages,history_format,
                    recovery_prompt_policy=recovery_prompt_policy)
            else:raise ValueError('unknown text history format')
        else:
            prompt=render_plain_continuation(messages,history_format)
        policy = ('native_act_only_plain_recovery_command' if purpose=='recovery'
                  else 'native_act_only_plain_action_or_rewind')
        return prompt, None, policy
    if prompt_format != 'native-chat':
        raise ValueError('unknown model prompt format')
    if purpose == 'action':
        command_phase=any(m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX)
            and '\nOutput phase: recovery-command.' in m['content'] for m in messages)
        if command_phase:
            encoded=resident_chat_messages(recovery_command_messages(messages,history_format),model_type)
            prompt=tokenizer.apply_chat_template(encoded,tokenize=False,add_generation_prompt=True)
            return prompt,encoded,'native_model_chat_recovery_command'
        if any(m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX) for m in messages):
            encoded=resident_chat_messages(decision_chat_messages(messages,history_format),model_type)
            prompt=tokenizer.apply_chat_template(encoded,tokenize=False,add_generation_prompt=True)
            return prompt,encoded,'native_model_chat_action_or_rewind'
        return render_native_messages(messages), messages, 'native_act_only_plain_continuation'
    if purpose == 'recovery':
        encoded = resident_chat_messages(recovery_command_messages(messages,history_format),model_type)
        prompt = tokenizer.apply_chat_template(encoded,tokenize=False,add_generation_prompt=True)
        return prompt, encoded, 'native_model_chat_recovery_command'
    raise ValueError('unknown generation purpose')


@dataclass(frozen=True)
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    queue_seconds: float | None
    raw: dict
    stop_reason: str

    def __post_init__(self):
        if not isinstance(self.text,str):
            raise ValueError('model returned non-text content')
        if self.stop_reason not in ('stop', 'frame', 'length', 'eos', 'eos_at_limit', 'time'):
            raise ValueError('unknown model stop reason')
        if any(type(v) is not int or v < 0 for v in (self.input_tokens,self.output_tokens)):
            raise ValueError('model response must supply actual token usage')
        if self.queue_seconds is not None and (not math.isfinite(self.queue_seconds) or self.queue_seconds < 0):
            raise ValueError('invalid model queue time')


class OpenAIModel:
    def __init__(self,args,settings,task_namespace):
        parsed = urlparse(args.endpoint)
        if parsed.scheme not in ('http','https') or parsed.username or parsed.password:
            raise ValueError('endpoint must be an HTTP URL without embedded credentials')
        self.plain = settings.prompt_format == 'plain'
        self.url = args.endpoint.rstrip('/')+('/completions' if self.plain else '/chat/completions')
        # Node-local model traffic must stay local even when the host uses a VPN proxy.
        self.http = build_opener(ProxyHandler({})) if parsed.hostname in ('localhost','127.0.0.1','::1') else build_opener()
        self.settings = settings
        self.key_variable = args.api_key_env
        self.identity = dict(backend='openai',model=settings.model,endpoint=args.endpoint,
                             queue_time_accounting='unavailable')

    def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
        s = self.settings
        if self.plain:
            prompt, _, _ = render_resident_request(messages, purpose, None, None,
                s.history_format, s.prompt_format, s.candidate_history,s.interface_format,s.recovery_prompt_policy,s.text_history_format)
            model_input = dict(prompt=prompt)
        else:
            encoded=(decision_chat_messages(messages,s.history_format) if any(m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX)
                     for m in messages) else messages)
            model_input = dict(messages=encoded)
        payload = dict(model=s.model,**model_input,max_tokens=max_tokens,temperature=s.temperature,
                       top_p=s.top_p,stream=False,
                       top_k=s.top_k,repetition_penalty=s.repetition_penalty)
        if purpose not in ('action','recovery'):
            raise ValueError('unknown generation purpose')
        if s.interface_format=='json-tools':
            from .json_tools import actor_only_request
            if actor_only_request(messages):payload['stop']=['\n']
            else:payload['response_format'] = {'type':'json_object'}
        elif purpose == 'action' or (s.interface_format=='sectioned' and one_line_recovery(messages)):
            payload['stop'] = ['\n']
        headers = {'Content-Type':'application/json'}
        if self.key_variable:
            headers['Authorization'] = 'Bearer '+os.environ[self.key_variable]
        request = Request(self.url,data=json.dumps(payload).encode('utf-8'),headers=headers,method='POST')
        on_submit()
        with self.http.open(request,timeout=timeout) as response:
            raw = json.load(response)
        if len(raw['choices']) != 1 or raw['choices'][0]['finish_reason'] not in ('stop','length'):
            raise ValueError('incomplete/multiple model choices; original response: '+json.dumps(raw,ensure_ascii=False))
        choice,usage = raw['choices'][0],raw['usage']
        if self.plain:
            raw = dict(raw, client_prompt=prompt, client_model_messages=None,
                client_prompt_policy='native_act_only_plain_'+purpose)
        try:
            return Reply(choice['text'] if self.plain else choice['message']['content'],usage['prompt_tokens'],usage['completion_tokens'],None,raw,
                         choice['finish_reason'])
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError('invalid model response; original response: '+json.dumps(raw,ensure_ascii=False)) from error

    def close(self):
        pass


class ResidentModel:
    def __init__(self,args,settings,task_namespace,bundle):
        from bound_resident_client import BoundResidentClient
        from experiment_settings import Sampling
        model,tokenizer,health = bundle
        self.model_type = model['model_type']
        self.history_format = settings.history_format
        self.prompt_format = settings.prompt_format
        self.candidate_history = settings.candidate_history
        self.interface_format = settings.interface_format
        self.recovery_prompt_policy = settings.recovery_prompt_policy
        self.text_output_boundary = settings.text_output_boundary
        self.text_history_format = settings.text_history_format
        self.tokenizer = tokenizer
        self.prompt_limit = args.prompt_limit
        self.context_limit = args.context_limit
        self.context_overflow = settings.context_overflow
        self.count = 0
        self.response_grace_seconds = settings.response_grace_seconds
        port = urlparse(args.endpoint).port
        binding = dict(parent_job_id=args.parent_job_id,step_id=resident_binding_step(args),hostname=args.node,
                       gpu=args.gpu_index,gpu_uuid=health['gpu_uuid'],pid=health['service_pid'],
                       instance=health['service_instance_id'],port=port)
        sampling = Sampling(temperature=settings.temperature,top_p=settings.top_p,top_k=settings.top_k,
                            repetition_penalty=settings.repetition_penalty)
        self.client = BoundResidentClient(binding,task_namespace,model,sampling=sampling,
                                          request_timeout=settings.request_timeout)
        actual = self.client.begin_run()
        if settings.text_output_boundary=='command':
            from .resident_text_frame import STOP_MODE
            if STOP_MODE not in actual['stop_modes']:
                self.client.close()
                raise ValueError('resident lacks the declared text-command boundary; no request submitted')
            if settings.text_history_format=='native' and 'first_nonempty_line' not in actual['stop_modes']:
                self.client.close()
                raise ValueError('resident lacks the declared native Actor line boundary; no request submitted')
        if settings.max_output_tokens > actual['max_new_tokens']:
            self.client.close()
            raise ValueError('requested output limit exceeds the loaded resident capacity')
        self.identity = dict(backend='resident',model=model,binding=binding,health=actual,
            execution=dict(parent_job_id=args.parent_job_id,step_id=args.step_id,hostname=args.node,pid=os.getpid()))

    def generate(self,messages,max_tokens,timeout,on_submit,*,purpose='action'):
        from model_backend import encode_prompt
        prepared_at = time.perf_counter()
        prompt,encoded_messages,policy = render_resident_request(
            messages,purpose,self.tokenizer,self.model_type,self.history_format,self.prompt_format,self.candidate_history,self.interface_format,self.recovery_prompt_policy,self.text_history_format)
        length = len(encode_prompt(self.tokenizer,prompt)['input_ids'])
        compression = None
        budget = min(self.prompt_limit,self.context_limit-max_tokens)
        if length > budget:
            if self.context_overflow=='stop':
                raise ModelContextLimit('model context exhausted; no prompt truncation is applied')
            from .context_window import fit_context,ContextWindowTooSmall
            def render(view):
                return render_resident_request(view,purpose,self.tokenizer,self.model_type,
                    self.history_format,self.prompt_format,self.candidate_history,self.interface_format,
                    self.recovery_prompt_policy,self.text_history_format)
            try:
                fitted,compression = fit_context(messages,render,
                    lambda text:len(encode_prompt(self.tokenizer,text)['input_ids']),budget,length)
            except ContextWindowTooSmall as error:
                raise ModelContextLimit('Explicit context projection failed: '+str(error)) from error
            prompt,encoded_messages,policy = fitted
        remaining = timeout - (time.perf_counter() - prepared_at)
        if remaining <= 0:
            raise ModelDeadlineExpired('prompt preparation exhausted the generation budget; no request submitted')
        self.count += 1
        if purpose not in ('action','recovery'):
            raise ValueError('unknown generation purpose')
        from .json_tools import actor_only_request
        request = SimpleNamespace(request_id='agentrewind_'+str(self.count),route='reflection',
                                  prompt=prompt,output_limit=max_tokens,
                                  stop_mode=('first_nonempty_line' if self.text_output_boundary=='command' and
                                    self.text_history_format=='native' and purpose=='action' else
                                    'agentrewind_text_command' if self.text_output_boundary=='command' else
                                    'first_line' if self.interface_format=='json-tools' and actor_only_request(messages) else
                                    'json_object' if self.interface_format=='json-tools' else
                                    'first_line' if purpose=='action' or (self.interface_format=='sectioned' and one_line_recovery(messages)) else None),parameter_language=None)
        self.client.request_timeout = remaining + self.response_grace_seconds
        on_submit()
        generation = self.client.generate(request,budget_seconds=remaining)
        raw = dict(generation.raw_response,client_messages=messages,client_prompt=prompt,
                   client_model_messages=encoded_messages,client_prompt_policy=policy)
        if compression is not None:
            raw['client_context_compression'] = compression
        return Reply(generation.text,generation.input_tokens,generation.output_tokens,
                     generation.raw_response['queue_time_seconds'],raw,generation.raw_response['finish_reason'])

    def close(self):
        self.client.close()


def prepare_resident(args,settings):
    import socket
    if os.environ.get('SLURM_JOB_ID') != args.parent_job_id or socket.gethostname() != args.node:
        raise RuntimeError('resident evaluation must run in its declared compute allocation')
    if os.environ.get('SLURM_STEP_ID') != args.step_id or not args.step_id.isdecimal():
        raise RuntimeError('resident evaluation must run in its declared workload step')
    url = urlparse(args.endpoint)
    if url.scheme != 'http' or url.hostname not in ('localhost','127.0.0.1') or url.port is None:
        raise ValueError('resident endpoint must be node-local')
    sys.path.insert(0,str(args.native_adapter_root/'source'))
    sys.path.insert(0,str(args.native_adapter_root))
    from model_backend import load_model_bundle,check_model_health
    model,tokenizer,eos = load_model_bundle(settings.model,args.dtype)
    with build_opener(ProxyHandler({})).open(args.endpoint.rstrip('/')+'/health',timeout=30) as response:
        health = json.load(response)
    check_model_health(health,model)
    if health['status'] != 'READY' or health['fatal_error'] is not None:
        raise RuntimeError('resident is unhealthy')
    if health['slurm_step_id'] != resident_binding_step(args):
        raise RuntimeError('resident is outside its declared model-service step')
    if health['max_num_seqs'] < args.workers or health['max_new_tokens'] < settings.max_output_tokens:
        raise ValueError('requested concurrency/output limit exceeds resident capacity')
    if health['model_context_tokens'] != args.context_limit or args.prompt_limit > health['max_prompt_tokens']:
        raise ValueError('requested context limits differ from resident capacity')
    if tuple(health['decoding']['eos_token_id']) != eos:
        raise ValueError('resident EOS differs from selected tokenizer')
    return model,tokenizer,health
