"""Explicit request/lease contract for concurrent STAR inference."""
import math
import threading
from icr_output_framing import STOP_MODES
from experiment_settings import Sampling


class ProtocolViolation(ValueError):
    def __init__(self,message,status=400):
        super().__init__(message);self.status=status


def experiment_name(value):
    if type(value) is not str or not value.strip():
        raise ProtocolViolation('experiment_id must be a nonempty string')
    return value


def validate_request(request, max_output_tokens=256):
    required={'route','prompt','experiment_id','task_seed','call_index','max_new_tokens'}
    optional={'stop_mode','budget_seconds','parameter_language','sampling'}
    if type(request) is not dict:raise ProtocolViolation('request must be an object')
    route=request.get('route')
    if route=='command_trie':required.add('allowed_commands')
    if route not in ('normal','reflection','command_trie') or not required<=set(request) or set(request)-required-optional:
        raise ProtocolViolation('request does not match the declared STAR schema')
    experiment_name(request['experiment_id'])
    if type(request['prompt']) is not str or not request['prompt'].strip():
        raise ProtocolViolation('prompt must be a nonempty string')
    if request['task_seed'] is not None or type(request['call_index']) is not int or request['call_index']!=0:
        raise ProtocolViolation('per-task and per-request seeds are disabled')
    limit=request['max_new_tokens']
    if type(limit) is not int or not 1<=limit<=max_output_tokens:
        raise ProtocolViolation('invalid route output budget')
    if 'sampling' in request:
        try:
            Sampling(**request['sampling'])
        except (TypeError,ValueError) as error:
            raise ProtocolViolation('invalid sampling settings: '+str(error)) from error
    mode=request.get('stop_mode')
    if mode is not None and mode not in STOP_MODES:raise ProtocolViolation('unknown stop mode')
    seconds=request.get('budget_seconds')
    if seconds is not None and (type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=3600):
        raise ProtocolViolation('invalid time budget')
    if route=='command_trie':
        commands=request['allowed_commands']
        if (type(commands) is not list or not 1<=len(commands)<=1024
                or any(type(c) is not str or not c or '\n' in c or '\r' in c for c in commands)
                or len(set(commands))!=len(commands)):
            raise ProtocolViolation('invalid complete-command language')
    if 'parameter_language' in request and (route=='command_trie' or mode is not None):
        raise ProtocolViolation('parameter language cannot mix with another constraint')


class LeaseBook:
    def __init__(self,engine_seed):
        self.engine_seed=engine_seed;self.leases={};self.lock=threading.RLock()

    def _require(self,name):
        experiment_name(name)
        if name not in self.leases:raise ProtocolViolation('unknown experiment lease',409)
        return self.leases[name]

    def begin(self,name):
        with self.lock:
            experiment_name(name)
            if name in self.leases:raise ProtocolViolation('experiment already active',409)
            self.leases[name]={'active':set(),'completed':0}
            return self.metadata(name)

    def acquire(self,name,request_id):
        with self.lock:
            lease=self._require(name)
            if request_id in lease['active']:raise ProtocolViolation('duplicate request identity',409)
            lease['active'].add(request_id)

    def release(self,name,request_id,completed):
        with self.lock:
            lease=self._require(name)
            if request_id not in lease['active']:raise RuntimeError('request lease was lost')
            lease['active'].remove(request_id)
            if completed:lease['completed']+=1

    def metadata(self,name):
        with self.lock:
            lease=self._require(name)
            return dict(experiment_id=name,engine_seed=self.engine_seed,request_seed=None,
                per_task_reseed=False,per_request_reseed=False,active_requests=len(lease['active']),
                completed_requests=lease['completed'],rng_scheme='vllm engine RNG; no request reseeding')

    def end(self,name):
        with self.lock:
            lease=self._require(name)
            if lease['active']:raise ProtocolViolation('experiment still has active requests',409)
            record=self.metadata(name);del self.leases[name];return record
