"""Separate engine scheduling waits from shared-batch processing intervals."""
from dataclasses import dataclass


@dataclass
class RequestTiming:
    last_active_end:float
    preprocessing_queue_seconds:float=0.0
    engine_queue_seconds:float=0.0
    active_step_seconds:float=0.0

    def scheduled_step(self,start,end):
        if start<self.last_active_end or end<start:
            raise RuntimeError('non-monotonic request timing')
        self.engine_queue_seconds+=start-self.last_active_end
        self.active_step_seconds+=end-start
        self.last_active_end=end

    @property
    def queue_seconds(self):
        return self.preprocessing_queue_seconds+self.engine_queue_seconds


def task_accounting(records,wall_seconds,complete):
    queue=sum(r['queue_time_seconds'] for r in records)
    if wall_seconds<0 or queue<0 or queue>wall_seconds:
        raise RuntimeError('task/model timing intervals are inconsistent')
    inputs=sum(r['input_tokens'] for r in records)
    outputs=sum(r['output_tokens'] for r in records)
    return dict(accounting_complete=complete,model_requests=len(records),input_tokens=inputs,output_tokens=outputs,
        total_tokens=inputs+outputs,wall_elapsed_seconds=wall_seconds,model_queue_seconds=queue,
        elapsed_seconds=wall_seconds-queue,elapsed_seconds_excluding_model_queue=wall_seconds-queue,
        model_active_step_seconds=sum(r['model_active_step_seconds'] for r in records),
        elapsed_seconds_definition='task wall time minus tokenizer-lock and all engine scheduling waits',
        model_active_step_time_definition='shared batch-step latency; not exclusive per-request GPU compute')
