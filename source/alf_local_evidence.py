"""Public evidence for bounded local repair, separate from the historical ETT."""
from tree_history import canonical


def is_query(action):
    return action in {'look', 'inventory'} or action.startswith(('go to ', 'examine ', 'think:'))


def decision_key(history, state, initial_observation):
    # Rejection text is not a new physical state. An accepted operation starts
    # a new evidence epoch, allowing revisits after any public physical effect.
    accepted = [r for r in history if r['native_accepted'] and not r['action'].startswith('think:')]
    return canonical(dict(
        observation=accepted[-1]['observation'] if accepted else initial_observation,
        available_actions=state['available_actions'],
        physical_effect_epoch=sum(not is_query(r['action']) for r in accepted)))


def new_accepted_evidence(receipt, earlier_receipts, initial_receipt):
    # The adapter's thought acknowledgement is not an environment observation.
    if not receipt['native_accepted'] or receipt['action'].startswith('think:'):
        return False
    def view(row):
        return canonical([row['observation'], row['after']['available_actions']])
    known = {view(initial_receipt)}
    known.update(view(r) for r in earlier_receipts
                 if r['native_accepted'] and not r['action'].startswith('think:'))
    return view(receipt) not in known


def trace_text(initial_observation, path, *, indexed=False):
    lines = [initial_observation]
    for index, occurrence in enumerate(path):
        marker = f'ACTION {index}: ' if indexed else '> '
        lines.extend([marker + occurrence.action, occurrence.receipt['observation']])
    return '\n'.join(lines)
