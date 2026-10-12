"""CLI and queue settings shared by every WebShop execution path."""
import argparse
from dataclasses import asdict, dataclass, fields
import math

ABLATIONS = ('full', 'without_yellow', 'without_red', 'without_event_subtree')


@dataclass(frozen=True)
class Sampling:
    do_sample: bool = True
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: int = 20
    repetition_penalty: float = 1.05

    def __post_init__(self):
        if type(self.do_sample) is not bool:
            raise ValueError('do_sample must be boolean')
        for name in ('temperature','top_p','repetition_penalty'):
            value = getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError(name+' must be finite')
        if self.temperature < 0 or (self.do_sample and self.temperature == 0):
            raise ValueError('sampling requires positive temperature; use --no-do-sample for greedy')
        if not 0 < self.top_p <= 1 or self.repetition_penalty <= 0:
            raise ValueError('invalid top_p or repetition_penalty')
        if type(self.top_k) is not int or (self.top_k != -1 and self.top_k < 1):
            raise ValueError('top_k must be -1 (unlimited) or a positive integer')

    def as_dict(self):
        return asdict(self)

    def engine_parameters(self):
        return dict(temperature=self.temperature if self.do_sample else 0.0,
                    top_p=self.top_p,top_k=self.top_k,repetition_penalty=self.repetition_penalty)

    def matches(self, decoding):
        return all(decoding.get(k) == v for k,v in self.as_dict().items())


@dataclass(frozen=True)
class Experiment:
    ablation: str = 'full'
    environment_seed: int = 1
    initial_actions: int = 15
    recovery_actions: int = 15
    max_recoveries: int = 1
    yellow_actions: int = 1
    yellow_entries: int = 2
    actor_output_tokens: int = 100
    yellow_diagnosis_tokens: int = 100
    yellow_output_tokens: int = 100
    red_reflection_tokens: int = 256
    red_selection_tokens: int = 100
    model_request_limit: int | None = None
    environment_call_limit: int | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    task_timeout: int = 7200
    request_timeout: float = 1200

    @classmethod
    def for_dataset(cls, dataset):
        defaults = {
            'webshop': {},
            'game24': dict(initial_actions=12, recovery_actions=12, model_request_limit=64),
            'alfworld': dict(initial_actions=49, recovery_actions=49, yellow_actions=4,
                             model_request_limit=112, environment_call_limit=1000),
        }
        if dataset not in defaults:
            raise ValueError('unknown dataset')
        return cls(**defaults[dataset])

    def __post_init__(self):
        for field in fields(self):
            name, value = field.name, getattr(self,field.name)
            if name == 'ablation':
                if value not in ABLATIONS:
                    raise ValueError('unknown STAR ablation')
            elif name == 'request_timeout':
                if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(name+' must be positive and finite')
            elif name in ('model_request_limit','environment_call_limit','total_input_tokens','total_output_tokens') and value is None:
                continue
            elif type(value) is not int or value < (0 if name in ('environment_seed','yellow_entries','max_recoveries') else 1):
                raise ValueError(name+' has an invalid budget or seed')

    def as_dict(self):
        return asdict(self)

    def required_output_cap(self):
        return max(self.actor_output_tokens,self.yellow_diagnosis_tokens,self.yellow_output_tokens,
                   self.red_reflection_tokens,self.red_selection_tokens)

    def ledger_limits(self, prompt_limit):
        from icr_core import Limits
        ordinary = max(self.actor_output_tokens,self.yellow_diagnosis_tokens,
                       self.yellow_output_tokens,self.red_selection_tokens)
        return Limits(environment_calls=(self.environment_call_limit if self.environment_call_limit is not None
                                         else 1+self.initial_actions+self.max_recoveries*(1+self.recovery_actions)),
            model_requests=self.model_request_limit,
            input_tokens=self.total_input_tokens if self.total_input_tokens is not None else
                None if self.model_request_limit is None else self.model_request_limit*prompt_limit,
            output_tokens=self.total_output_tokens if self.total_output_tokens is not None else
                None if self.model_request_limit is None else
                (self.model_request_limit-self.max_recoveries)*ordinary+self.max_recoveries*self.red_reflection_tokens,
            elapsed_seconds=self.task_timeout)


def add_sampling_arguments(parser):
    parser.add_argument('--do-sample', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--temperature',type=float,default=.1)
    parser.add_argument('--top-p','--top_p',dest='top_p',type=float,default=1.)
    parser.add_argument('--top-k','--top_k',dest='top_k',type=int,default=20)
    parser.add_argument('--repetition-penalty',type=float,default=1.05)


def add_experiment_arguments(parser):
    for field in fields(Experiment):
        flag = '--'+field.name.replace('_','-')
        if field.name == 'ablation':
            parser.add_argument(flag, choices=ABLATIONS, default=field.default)
            continue
        aliases = [flag,'--seed'] if field.name == 'environment_seed' else [flag]
        parser.add_argument(*aliases,dest=field.name,type=float if field.name == 'request_timeout' else int,
                            default=field.default)


def settings_from_args(args, cls):
    return cls(**{f.name:getattr(args,f.name) for f in fields(cls)})


def settings_flags(settings):
    flags = []
    for name,value in settings.as_dict().items():
        if value is None:
            continue
        if name == 'do_sample':
            flags.append('--do-sample' if value else '--no-do-sample')
        else:
            flags.extend(['--'+name.replace('_','-'),str(value)])
    return flags


def validate_runtime_settings(health, experiment, sampling, prompt_limit):
    if (experiment.required_output_cap() > health['max_new_tokens']
            or prompt_limit > health['max_prompt_tokens']
            or prompt_limit + experiment.required_output_cap() > health['model_context_tokens']):
        raise ValueError('requested token limits exceed resident capacity')
    if not health.get('request_sampling_supported',False) and not sampling.matches(health['decoding']):
        raise ValueError('legacy resident cannot change sampling per request')
    if not health.get('flexible_output_limits',False) and any(value != 100 for value in (
            experiment.actor_output_tokens,experiment.yellow_output_tokens,experiment.red_selection_tokens)):
        raise ValueError('legacy resident only supports 100-token action/selection requests')
