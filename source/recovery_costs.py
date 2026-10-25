"""Passive recovery cost boundaries; never change controller decisions."""
from copy import deepcopy
import math
import time


COST_KEYS = ('model_requests', 'input_tokens', 'output_tokens', 'model_queue_seconds')
BOUNDARIES = ('red_preparation_started', 'red_checkpoint_restored', 'recovery_stage_finished')


class RecoveryCostRecorder:
    def __init__(self, started, *, clock=time.monotonic):
        self.started, self.clock = started, clock
        self.cost = dict.fromkeys(COST_KEYS, 0)
        self.boundaries = []

    def record_generation(self, generation):
        for name in ('input_tokens', 'output_tokens'):
            value = getattr(generation, name)
            if type(value) is not int or value < 0:
                raise ValueError('recovery costs require measured nonnegative token counts')
        queue = generation.raw_response['queue_time_seconds']
        if type(queue) not in (int, float) or not math.isfinite(queue) or queue < 0:
            raise ValueError('recovery costs require measured queue time')
        self.cost['model_requests'] += 1
        self.cost['input_tokens'] += generation.input_tokens
        self.cost['output_tokens'] += generation.output_tokens
        self.cost['model_queue_seconds'] += queue

    def snapshot(self, elapsed_seconds):
        return dict(self.cost, elapsed_seconds=elapsed_seconds,
                    queue_excluded_elapsed_seconds=elapsed_seconds-self.cost['model_queue_seconds'])

    def observe(self, event):
        value = self.snapshot(self.clock()-self.started)
        if event in BOUNDARIES:
            self.boundaries.append(dict(event=event, cost=value))
        return dict(task_elapsed_seconds=value['elapsed_seconds'], completed_model_cost=value)

    def report(self, elapsed_seconds):
        total = self.snapshot(elapsed_seconds)
        previous = self.snapshot(0)
        previous.update({name: 0 for name in previous})
        phase, recovery = 'execution', 0
        segments = []

        def close(end, reason):
            increment = {name: end[name]-previous[name] for name in end}
            if any(increment[name] < 0 for name in (*COST_KEYS, 'elapsed_seconds')):
                raise ValueError('recovery cost boundaries must be monotone')
            segments.append(dict(phase=phase, recovery_count=recovery, end_reason=reason,
                                 start=deepcopy(previous), end=deepcopy(end), cost=increment))

        for boundary in self.boundaries:
            event, end = boundary['event'], boundary['cost']
            if event == 'red_preparation_started':
                if phase != 'execution':
                    raise ValueError('recovery preparation began before the previous restore')
                close(end, event)
                previous = end
                recovery += 1
                phase = 'recovery_preparation'
            elif event == 'red_checkpoint_restored':
                if phase != 'recovery_preparation':
                    raise ValueError('restore has no recorded preparation boundary')
                close(end, event)
                previous, phase = end, 'execution'
        close(total, 'task_finished' if phase == 'execution' else 'task_ended_before_restore')
        return dict(schema='star_recovery_cost_segments_v1', totals=total,
                    semantics='Disjoint measured intervals: initial execution, recovery preparation including restoration, and post-restore execution. Completed model requests only; fatal records are not promoted to complete tasks.',
                    boundaries=deepcopy(self.boundaries), segments=segments)
