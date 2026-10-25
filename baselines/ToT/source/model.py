"""Synchronous complete-list model calls over raw resident HTTP."""
import time
from .resident_transport import append


def validate_health(health, req):
    if health['status'] != 'READY' or health['fatal_error'] is not None or health['engine'] != 'vllm':
        raise ValueError('resident engine is not ready')
    if health['model_path'] != req['model_path']:
        raise ValueError('selected model path differs from loaded resident model')
    if health['max_num_seqs'] != req['max_num_seqs']:
        raise ValueError('requested engine concurrency differs from the resident')
    if health['max_new_tokens'] < req['experiment']['search']['max_output_tokens']:
        raise ValueError('resident output capacity is below the requested search token limit')
    if health['chat_template_applied'] or 'first_line' not in health['stop_modes']:
        raise ValueError('raw prompt and generation-time first-line framing are required')
    if 'tot_label' not in health['stop_modes']:
        raise ValueError('resident must stop when a complete ToT label is generated')


def unbound_gpt(*args, **kwargs):
    raise RuntimeError('model calls must be bound to a task resident')


class Model:
    def __init__(self, tokenizer, transport, config, temperature, directory, request_pool=None):
        self.tokenizer, self.transport, self.config = tokenizer, transport, config
        self.temperature, self.directory = temperature, directory
        self.role, self.logical_calls = 'candidate', 0
        if request_pool is not None:
            raise ValueError('candidate request pools are excluded from this experiment')

    def gpt(self, prompt, model=None, temperature=None, max_tokens=None, n=1, stop=None):
        if temperature is not None and temperature != self.temperature:
            raise ValueError('search changed the configured sampling temperature')
        cap = 256 if self.role == 'reflection' else self.config.max_output_tokens
        if max_tokens is not None and max_tokens != cap:
            raise ValueError('search changed the configured token limit')
        if type(n) is not int or n < 1:
            raise ValueError('invalid generation count')
        call_id = self.logical_calls
        self.logical_calls += 1
        append(self.directory/'logical_calls.jsonl', dict(call_id=call_id, role=self.role,
            samples=n, stop=stop, time=time.time(), transport='synchronous complete candidate list'))
        outputs=[]
        self.transport.role=self.role
        for sample in range(n):
            raw=self.transport.generate(prompt,cap)['raw_output']
            text=raw
            # Preserve author parser delimiters after token-level framing.
            if stop is not None:
                stops=[stop] if isinstance(stop,str) else stop
                offsets=[text.index(s) for s in stops if s in text]
                if offsets:text=text[:min(offsets)]
            append(self.directory/'samples.jsonl',dict(call_id=call_id,sample=sample,
                role=self.role,raw_output=raw,framed_output=text))
            outputs.append(text)
        return outputs

    def for_role(self, role):
        def call(*args, **kwargs):
            previous=self.role
            self.role=role
            try:return self.gpt(*args,**kwargs)
            finally:self.role=previous
        return call

    def close(self):
        pass
