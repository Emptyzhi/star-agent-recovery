"""Explicit projection of rejected proposals within one candidate-selection phase."""
from copy import deepcopy
import json


LEDGER_PREFIX = '[AGENTREWIND_REJECTED_HISTORY]\n'


class RejectedHistory:
    def __init__(self, keep):
        if type(keep) is not int or keep not in (0, 2):
            raise ValueError('rejected-history-keep must be 0 or 2')
        self.keep = keep
        self.clear()

    def clear(self):
        self.base = None
        self.stage = None
        self.entries = []

    def begin(self, messages, *, trace, candidate_record_uid, continuation_record_uid=None):
        if not self.keep:
            return None
        if self.base is not None:
            raise ValueError('candidate-selection phase is already active')
        self.base = deepcopy(messages)
        self.stage = dict(trace=trace, candidate_record_uid=candidate_record_uid)
        if continuation_record_uid is not None:
            self.stage['continuation_record_uid'] = continuation_record_uid
        self.entries = []
        return dict(stage=self.stage, base_messages=deepcopy(messages), keep=self.keep)

    def register(self, messages, *, trace, record_uid, raw_reply, tool, target,
                 error_type, error):
        if not self.keep or self.base is None:
            raise ValueError('rejection registration requires an active compression phase')
        expected = deepcopy(self.base)
        for entry in self.entries:
            expected.extend(entry['messages'])
        if messages[:-2] != expected or messages[-2] != dict(role='assistant', content=raw_reply):
            raise ValueError('rejected response does not extend the exact phase history')
        if messages[-1]['role'] != 'user' or not isinstance(error, str):
            raise ValueError('rejection must have its actual error receipt')
        entry = dict(source=dict(trace=trace, record_uid=record_uid), tool=tool,
                     target=target, error_type=error_type, error=error,
                     messages=deepcopy(messages[-2:]))
        self.entries.append(entry)
        return dict(stage=self.stage, entry=deepcopy(entry))

    def project(self, messages):
        if not self.keep or self.base is None:
            return deepcopy(messages), None
        expected = deepcopy(self.base)
        for entry in self.entries:
            expected.extend(entry['messages'])
        if expected != messages:
            raise ValueError('candidate-selection history changed outside registered rejections')
        count = len(self.entries) - self.keep
        if count <= 0:
            return deepcopy(messages), None
        groups = {}
        for entry in self.entries[:count]:
            key = (entry['tool'], entry['target'], entry['error_type'], entry['error'])
            if key not in groups:
                groups[key] = dict(tool=entry['tool'], target=entry['target'],
                                   error_type=entry['error_type'], error=entry['error'],
                                   count=0, first_source=entry['source'], last_source=entry['source'])
            groups[key]['count'] += 1
            groups[key]['last_source'] = entry['source']
        ledger = dict(controller_generated=True, stage=self.stage,
                      meaning='Earlier rejected proposal text is omitted from this model input. '
                              'These attempts executed no environment action or restoration. '
                              'Full original replies and receipts remain in the trace. '
                              'The most recent two rejected proposals and receipts follow verbatim.',
                      omitted_proposals=count, groups=list(groups.values()))
        marker = dict(role='system', content=LEDGER_PREFIX + json.dumps(ledger, ensure_ascii=False))
        projected = deepcopy(self.base)
        index = next(i for i, message in enumerate(projected) if message['role'] == 'user')
        projected.insert(index, marker)
        for entry in self.entries[count:]:
            projected.extend(deepcopy(entry['messages']))
        return projected, dict(stage=self.stage, keep=self.keep, omitted_proposals=count,
                               ledger=ledger, source_message_count=len(messages),
                               projected_message_count=len(projected))
