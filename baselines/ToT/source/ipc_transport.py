import time
from .ipc_protocol import exchange
from .resident_transport import ResidentTransport, append


class IPCTransport(ResidentTransport):
    def generate(self, prompt, max_tokens):
        role, trial = self.role, self.trial
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        payload = dict(prompt=prompt, max_tokens=max_tokens,
                       sampling=self.sampling, experiment_id=self.lease)
        append(self.directory/'requests.jsonl', dict(ordinal=ordinal, role=role, trial=trial,
               api='vllm_unix_ipc_raw_text', time=time.time(), payload=payload))
        started = time.perf_counter()
        raw = exchange(self.endpoint, 'generate', payload)
        record = dict(ordinal=ordinal, role=role, trial=trial, response=raw,
                      ipc_round_trip_seconds=time.perf_counter()-started)
        append(self.directory/'responses.jsonl', record)
        for key in ('parent_job_id', 'hostname', 'gpu_uuid', 'slurm_step_id'):
            if str(raw['gpu_identity'][key]) != str(self.identity[key]):
                raise ValueError('resident identity changed: '+key)
        for key in ('service_pid', 'service_instance_id'):
            if raw['runtime'][key] != self.identity[key]:
                raise ValueError('resident process changed: '+key)
        if raw['runtime']['chat_template_applied'] is not False or raw['output_token_limit'] != max_tokens:
            raise ValueError('resident changed raw-text protocol or output limit')
        if raw['text'] != raw['raw_output'] or len(raw['output_token_ids']) != raw['output_tokens']:
            raise ValueError('resident output receipt is inconsistent')
        if raw['decoding'] != self.sampling:
            raise ValueError('sampling changed')
        self.records.append(record)
        return raw

    def close(self):
        pass

    def costs(self, elapsed):
        rows = [record['response'] for record in self.records]
        return dict(model_requests=len(rows), input_tokens=sum(r['input_tokens'] for r in rows),
            output_tokens=sum(r['output_tokens'] for r in rows), elapsed_seconds=elapsed,
            model_queue_seconds=sum(r['queue_time_seconds'] for r in rows),
            queue_excluded_elapsed_seconds=None, model_active_step_seconds=None,
            queue_time_semantics='IPC admission request-seconds; internal vLLM wait not separately measured',
            queue_excluded_elapsed_unavailable_reason='engine scheduling critical path not measured')
