"""OpenAI SDK Chat Completions with strict resident identity and usage receipts."""
import time
from openai import APIStatusError, APITimeoutError, OpenAI
from .resident_transport import ResidentTransport, append


class ChatCompletionTransport(ResidentTransport):
    def close(self):
        # Borrowed shard clients outlive individual tasks and close with the shard.
        with self._lock:
            owned = list(self._owned_sessions)
            self._owned_sessions.clear()
        for client in owned:
            client.close()

    def _session(self):
        if self.http is not None:
            return self.http
        client = getattr(self._local, 'session', None)
        if client is None:
            client = OpenAI(base_url=self.endpoint+'/v1', api_key='local-resident',
                            timeout=self.timeout, max_retries=0)
            self._local.session = client
            with self._lock:
                self._owned_sessions.append(client)
        return client

    def generate(self, prompt, max_tokens):
        role, trial = self.role, self.trial
        extra = dict(top_k=self.sampling['top_k'],
                     repetition_penalty=self.sampling['repetition_penalty'],
                     experiment_id=self.lease,
                     route='reflection' if role in ('reflection', 'analysis') else 'normal')
        payload = dict(model=self.identity['model_path'],
                       messages=[dict(role='user', content=prompt)],
                       max_tokens=max_tokens, temperature=self.sampling['temperature'],
                       top_p=self.sampling['top_p'], n=1, stream=False, **extra)
        with self._lock:
            ordinal = self._next_ordinal
            self._next_ordinal += 1
            append(self.directory/'requests.jsonl', dict(ordinal=ordinal, role=role, trial=trial,
                   time=time.time(), api='chat.completions', payload=payload))
        started = time.perf_counter()
        def request():
            return self._session().chat.completions.create(
                model=payload['model'], messages=payload['messages'], max_tokens=max_tokens,
                temperature=payload['temperature'], top_p=payload['top_p'],
                n=1, stream=False, extra_body=extra)
        try:
            if self.request_limiter is None:
                response = request()
            else:
                with self.request_limiter:
                    response = request()
        except APIStatusError as error:
            with self._lock:
                append(self.directory/'http_errors.jsonl', dict(ordinal=ordinal,
                       status=error.status_code, body=error.response.text,
                       elapsed_seconds=time.perf_counter()-started))
            raise
        except APITimeoutError as error:
            with self._lock:
                append(self.directory/'http_errors.jsonl', dict(ordinal=ordinal,
                       error=repr(error), elapsed_seconds=time.perf_counter()-started))
            raise TimeoutError('HTTP request timed out via OpenAI SDK') from error
        raw = response.model_extra['resident_response']
        record = dict(ordinal=ordinal, role=role, trial=trial,
                      http_round_trip_seconds=time.perf_counter()-started,
                      response=raw, chat_completion=response.model_dump(exclude={'resident_response'}))
        with self._lock:
            append(self.directory/'responses.jsonl', record)
        if len(response.choices) != 1 or response.choices[0].message.content != raw['raw_output']:
            raise ValueError('Chat Completions content differs from model output')
        if response.choices[0].finish_reason != raw['openai_finish_reason']:
            raise ValueError('Chat Completions finish reason differs from model evidence')
        if (response.usage.prompt_tokens, response.usage.completion_tokens) != (raw['input_tokens'], raw['output_tokens']):
            raise ValueError('Chat Completions token accounting mismatch')
        if raw['runtime']['chat_template_applied'] is not True:
            raise ValueError('native chat template was not applied')
        for key in ('parent_job_id', 'hostname', 'gpu_uuid'):
            if raw['gpu_identity'][key] != self.identity[key]:
                raise ValueError('resident identity changed: '+key)
        for key in ('service_pid', 'service_instance_id'):
            if raw['runtime'][key] != self.identity[key]:
                raise ValueError('resident model process changed: '+key)
        if str(raw['gpu_identity']['slurm_step_id']) != str(self.identity['slurm_step_id']):
            raise ValueError('resident compute step changed')
        if raw['effective_seed'] is not None or raw['task_seed'] is not None:
            raise ValueError('request reseeded continuous model RNG')
        if raw['text'] != raw['raw_output'] or raw['output_token_limit'] != max_tokens:
            raise ValueError('service changed output protocol')
        if raw['output_tokens'] != len(raw['output_token_ids']):
            raise ValueError('incomplete output token accounting')
        if any(raw['decoding'][k] != v for k, v in self.sampling.items()):
            raise ValueError('effective sampling mismatch')
        with self._lock:
            self.records.append(record)
        return raw
