"""Explicit settings and shared active-history window policy."""
import json
from pathlib import Path

from icr_core import ProtocolError


CONFIG = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
if (set(CONFIG) != {'tree_mode', 'history_policy', 'history_window_tokens', 'reflection_output_tokens'}
        or CONFIG['tree_mode'] not in ('none', 'event', 'full')
        or CONFIG['history_policy'] != 'full_retained_action_observation'
        or CONFIG['history_window_tokens'] != 32768
        or CONFIG['reflection_output_tokens'] != 256):
    raise ProtocolError('invalid matched-history experiment configuration')


def render_active_history(history, *, omitted_records=0):
    lines = ['[ACTIVE_EXECUTION_HISTORY]\n']
    if omitted_records:
        lines.append('[HISTORY_TRUNCATED omitted_records=%d policy=first_and_recent_complete_receipts]\n' % omitted_records)
    for receipt in history:
        action, observation = receipt['action'], receipt['observation']
        if type(action) is not str or type(observation) is not str:
            raise ProtocolError('history action and observation must be actual text')
        lines.append(f'Action: {action}\nObservation: {observation}\n\n')
    lines.append('[ACTIVE_EXECUTION_HISTORY_END]')
    return ''.join(lines)
