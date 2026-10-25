"""WebShop client bound to one continuously sampled resident vLLM instance."""

from copy import deepcopy
import math
import time

import requests

from icr_core import Generation, ProtocolError
from model_backend import check_model_health
from experiment_settings import Sampling


class BoundResidentClient:
    def __init__(self, binding, experiment_id, model, *, sampling=None, request_timeout=1200):
        self.binding = deepcopy(binding)
        self.experiment_id = experiment_id
        self.model = dict(model)
        self.sampling = Sampling() if sampling is None else sampling
        self.request_timeout = request_timeout
        self.endpoint = 'http://127.0.0.1:%d' % binding['port']
        self.http = requests.Session()
        self.active = False
        self.identity = None
        self.health_record = None
        self.stop_modes = ()

    def _check_identity(self, value):
        binding = self.binding
        if (value['parent_job_id'] != binding['parent_job_id']
                or str(value['slurm_step_id']) != str(binding.get('resident_step_id', binding['step_id']))
                or value['hostname'] != binding['hostname']
                or value['cuda_visible_devices'] != str(binding['gpu'])
                or value['gpu_name'] != 'NVIDIA A100-PCIE-40GB'
                or value['gpu_uuid'] != binding['gpu_uuid']):
            raise ProtocolError('response does not match the selected resident GPU/run identity')

    def health(self):
        response = self.http.get(self.endpoint + '/health',
                                 params={'experiment_id': self.experiment_id}, timeout=30)
        response.raise_for_status()
        health = response.json()
        self._check_identity(health)
        check_model_health(health, self.model)
        if (health['status'] != 'READY' or health['engine'] != 'vllm'
                or health['engine_version'] != '0.6.6'
                or health['loaded_once'] is not True
                or health['max_num_seqs'] < 1
                or health['chat_template_applied'] is not False
                or health['prompt_truncation'] is not False):
            raise ProtocolError('resident differs from the WebShop vLLM contract')
        lease = health['sampling_run']
        if (type(lease) is not dict or lease['experiment_id'] != self.experiment_id
                or lease['request_seed'] is not None
                or lease['per_task_reseed'] is not False
                or lease['per_request_reseed'] is not False):
            raise ProtocolError('resident RNG lease differs from continuous-run policy')
        decoding = health['decoding']
        if (decoding['request_seed'] is not None or
                (not health.get('request_sampling_supported',False) and not self.sampling.matches(decoding))):
            raise ProtocolError('resident lacks per-request sampling and differs from requested settings')
        identity = {key: health[key] for key in
                    ('service_instance_id', 'service_pid', 'service_start_time', 'gpu_uuid')}
        if self.binding['pid'] is not None and identity['service_pid'] != self.binding['pid']:
            raise ProtocolError('resident model PID changed after dispatch')
        if self.binding['instance'] is not None and identity['service_instance_id'] != self.binding['instance']:
            raise ProtocolError('resident model instance changed after dispatch')
        self.identity = identity
        self.health_record = health
        self.stop_modes = tuple(health['stop_modes'])
        return health

    def begin_run(self):
        response = self.http.post(self.endpoint + '/begin_run',
                                  json={'experiment_id': self.experiment_id}, timeout=30)
        response.raise_for_status()
        self.active = True
        return self.health()

    def end_run(self):
        if not self.active:
            raise ProtocolError('resident experiment lease is not active')
        response = self.http.post(self.endpoint + '/end_run',
                                  json={'experiment_id': self.experiment_id}, timeout=30)
        response.raise_for_status()
        self.active = False
        return response.json()

    def generate(self, request, *, budget_seconds=None):
        if not self.active or self.identity is None:
            raise ProtocolError('model client is not health-bound')
        output_limit = request.output_limit
        if type(output_limit) is not int or not 1 <= output_limit <= self.health_record['max_new_tokens']:
            raise ProtocolError('invalid output token limit')
        if (not self.health_record.get('flexible_output_limits',False)
                and request.route != 'reflection' and output_limit != 100):
            raise ProtocolError('legacy resident requires 100-token action limits')
        payload = dict(route=request.route, prompt=request.prompt,
                       experiment_id=self.experiment_id, task_seed=None,
                       call_index=0, max_new_tokens=output_limit)
        if budget_seconds is not None:
            if type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
                raise ProtocolError('generation budget must be positive and finite')
            payload['budget_seconds'] = budget_seconds
        if self.health_record.get('request_sampling_supported',False):
            payload['sampling'] = self.sampling.as_dict()
        if request.stop_mode is not None:
            if request.stop_mode not in self.stop_modes:
                raise ProtocolError('resident does not support the requested stop mode')
            payload['stop_mode'] = request.stop_mode
        if request.parameter_language is not None:
            payload['parameter_language'] = request.parameter_language
        if request.route == 'command_trie':
            payload['allowed_commands'] = list(request.commands)
        started = time.perf_counter()
        response = self.http.post(self.endpoint + '/generate_text', json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        raw = response.json()
        self._check_identity(raw['gpu_identity'])
        if {key: raw['runtime'][key] for key in self.identity} != self.identity:
            raise ProtocolError('resident service instance changed during generation')
        decoding = raw['decoding']
        if not self.sampling.matches(decoding) or decoding['request_seed'] is not None:
            raise ProtocolError('generation changed the requested sampling configuration')
        if (raw['route'] != request.route or raw['task_seed'] is not None
                or raw['call_index'] != 0 or raw['effective_seed'] is not None
                or raw['output_token_limit'] != output_limit
                or raw['budget_seconds'] != budget_seconds
                or raw['input_tokens'] + output_limit > self.health_record['model_context_tokens']
                or raw['output_tokens'] != len(raw['output_token_ids'])
                or raw['text'] != raw['raw_output']):
            raise ProtocolError('vLLM response violates the declared request/accounting contract')
        for key in ('queue_time_seconds', 'model_active_step_seconds', 'http_round_trip_seconds'):
            value = (time.perf_counter() - started if key == 'http_round_trip_seconds'
                     else raw[key])
            if not math.isfinite(value) or value < 0:
                raise ProtocolError('invalid model timing: ' + key)
        return Generation(request, raw['text'], raw['input_tokens'],
                          raw['output_tokens'], raw)

    def close(self):
        if self.active:
            self.end_run()
        self.http.close()
