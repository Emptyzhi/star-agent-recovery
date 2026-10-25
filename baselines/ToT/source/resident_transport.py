"""Recorded resident HTTP transport; no task parser or recovery policy."""
import json
import math
import time
from threading import RLock, local

ROLE_STOP_MODES={'actor':'first_line','proposal':'first_line','candidate':'first_line',
                 'reflection':None,'analysis':None,'evaluation':'tot_label'}


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')


class ResidentTransport:
    def __init__(self, endpoint, lease, sampling, identity, directory, timeout=1200, session=None,
                 request_limiter=None):
        self._lock = RLock()
        self._local = local()
        self._owned_sessions = []
        self._next_ordinal = 0
        self.http = session
        self.endpoint, self.lease, self.sampling = endpoint, lease, sampling
        self.identity, self.directory, self.timeout = identity, directory, timeout
        self.records = []
        self.request_limiter = request_limiter
        self.role, self.trial = 'actor', 0

    def _session(self):
        if self.http is not None:
            return self.http
        cached = getattr(self._local, 'session', None)
        if cached is None:
            import requests
            cached = requests.Session()
            self._local.session = cached
            with self._lock:
                self._owned_sessions.append(cached)
        return cached

    def generate(self, prompt, max_tokens):
        role, trial = self.role, self.trial
        if role not in ROLE_STOP_MODES:
            raise ValueError('undeclared generation role: '+role)
        stop_mode=ROLE_STOP_MODES[role]
        payload = dict(route='reflection' if role in ('reflection','analysis') else 'normal',
                       prompt=prompt, experiment_id=self.lease, task_seed=None, call_index=0,
                       max_new_tokens=max_tokens, stop_mode=stop_mode, sampling=self.sampling)
        with self._lock:
            ordinal = self._next_ordinal
            self._next_ordinal += 1
            append(self.directory/'requests.jsonl', dict(ordinal=ordinal, role=role, trial=trial,
                                                        time=time.time(), payload=payload))
        started = time.perf_counter()
        if self.request_limiter is None:
            response = self._session().post(self.endpoint+'/generate_text', json=payload, timeout=self.timeout)
        else:
            with self.request_limiter:
                response = self._session().post(self.endpoint+'/generate_text', json=payload, timeout=self.timeout)
        elapsed = time.perf_counter()-started
        if not response.ok:
            with self._lock:
                append(self.directory/'http_errors.jsonl', dict(ordinal=ordinal, status=response.status_code,
                                                               body=response.text, elapsed_seconds=elapsed))
        response.raise_for_status()
        raw = response.json()
        record = dict(ordinal=ordinal, role=role, trial=trial,
                      http_round_trip_seconds=elapsed, response=raw)
        with self._lock:
            append(self.directory/'responses.jsonl', record)
        for key in ('parent_job_id','hostname','gpu_uuid'):
            if raw['gpu_identity'][key] != self.identity[key]:
                raise ValueError('resident identity changed: '+key)
        if raw['runtime']['service_pid'] != self.identity['service_pid']:
            raise ValueError('resident model process changed')
        if raw['runtime']['service_instance_id'] != self.identity['service_instance_id']:
            raise ValueError('resident model instance changed')
        if str(raw['gpu_identity']['slurm_step_id']) != str(self.identity['slurm_step_id']):
            raise ValueError('resident compute step changed')
        if raw['effective_seed'] is not None or raw['task_seed'] is not None:
            raise ValueError('request reseeded continuous model RNG')
        if raw['text'] != raw['raw_output'] or raw['output_token_limit'] != max_tokens:
            raise ValueError('service changed output protocol')
        if raw['stop_mode']!=stop_mode:
            raise ValueError('service did not apply the declared output boundary')
        if raw['output_tokens'] != len(raw['output_token_ids']):
            raise ValueError('incomplete output token accounting')
        if any(raw['decoding'][k] != v for k,v in self.sampling.items()):
            raise ValueError('effective sampling mismatch')
        with self._lock:
            self.records.append(record)
        return raw

    def costs(self, elapsed):
        rows = [r['response'] for r in self.records]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid task elapsed time')
        for row in rows:
            for key in ('queue_time_seconds', 'model_active_step_seconds'):
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError('invalid request duration: '+key)
        queue = sum(r['queue_time_seconds'] for r in rows)
        if queue>elapsed:
            raise ValueError('sequential queue accounting exceeds task elapsed time')
        return dict(model_requests=len(rows), input_tokens=sum(r['input_tokens'] for r in rows),
                    output_tokens=sum(r['output_tokens'] for r in rows), elapsed_seconds=elapsed,
                    model_queue_seconds=queue, queue_excluded_elapsed_seconds=elapsed-queue,
                    queue_time_semantics='sum of measured model waits; sequential per-task requests',
                    model_active_step_seconds=sum(r['model_active_step_seconds'] for r in rows))

    def close(self):
        sessions = [self.http] if self.http is not None else []
        with self._lock:
            sessions += self._owned_sessions
            self._owned_sessions = []
        for session in sessions:
            session.close()


class PipelineTransport:
    """Expose Transformers' pipeline return shape to unchanged author parsers."""
    def __init__(self, transport, reflection_tokens=256):
        self.transport, self.reflection_tokens = transport, reflection_tokens

    def __call__(self, prompt, **kwargs):
        if kwargs['return_full_text'] is not False:
            raise ValueError('author pipeline must request generated text only')
        cap = self.reflection_tokens if self.transport.role == 'reflection' else kwargs['max_new_tokens']
        raw = self.transport.generate(prompt, cap)
        return [{'generated_text': raw['raw_output']}]


class GARollbackTransport:
    """Author GA text/probability return contract over vLLM processed logprobs."""
    def __init__(self, transport):
        self.transport = transport

    def _generate(self, prompt, stop=None, max_new_tokens=100, gen_logits=False):
        self.transport.role = 'analysis' if gen_logits else 'actor'
        raw = self.transport.generate(prompt, max_new_tokens)
        text = raw['raw_output']
        result = dict(text=None, average_prob=None, logsumexp_prob=None)
        if gen_logits:
            probabilities = raw['selected_probabilities']
            if len(probabilities) != raw['output_tokens'] or not probabilities:
                raise ValueError('missing generated-token probabilities for GA confidence gate')
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError('invalid generated-token probability')
            result['average_prob'] = sum(probabilities)/len(probabilities)
            peak = max(probabilities)
            result['logsumexp_prob'] = peak+math.log(sum(math.exp(p-peak) for p in probabilities))
        if stop:
            # Literal author behavior, including IndexError for all-empty output.
            text = [part for part in text.split(stop) if part][0]
        result['text'] = text.strip()
        return result
